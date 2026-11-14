/* halo_q6k_layout.cpp -- the stored form of ggml Q6_K dense weights ("halo.q6k.*", see
 * halo_q6k.h) and the relayouts that produce it from the weight's canonical planes (spec §4.3).
 * Pure host C++: nothing here includes halo.h or <hip/hip_runtime.h>.
 *
 * The canonical planes are `codes` i6 [N][K] (q - 32, packed 6 bits per value), `scale` i8
 * [N][K/16] and `scale.1` f16 [N][K/256]. The relayout only moves bytes: 16-row by 64-K tiles of
 * codes (each row's four 16-value sub-blocks stay in order), the scales in 16-value sub-block
 * order, and the f16 super-scales in 256-value order. All three hooks are exact inverses; the
 * six-bit code stream is never repacked, so a round trip is byte for byte.
 *
 * Guards. The format addresses K in whole 256-value blocks and N in whole 16-row tiles, so a
 * shape outside that has no tile to hold its last rows or columns and layout, relayout and
 * unrelayout refuse it with RAD_E_SHAPE before touching memory. The destination size is checked
 * against the layout's exactly, the plane extents against N, K, K/16 and K/256, the encoding
 * against the three planes Q6_K has, and N*K is bounded so every offset fits int64 with room.
 */
#include "halo_plugin.h"
#include "halo_enc.h"
#include "halo_q6k.h"

#include <cstdint>
#include <cstring>

enum { Q6K_A = 0, Q6K_ASCALE = 1, Q6K_B = 2, Q6K_BSCALE = 3, Q6K_Y = 4, Q6K_ASUM = 5, Q6K_BREF = 6 };

namespace {

struct QGeom { long long N, K, KB, KG; };

int qgeom(const RadParam* p, int n_p, QGeom* g) {
    g->N = halo_pgeti_or(p, n_p, "N", 0);
    g->K = halo_pgeti_or(p, n_p, "K", 0);
    if (!halo_q6k::shape_ok(g->N, g->K)) return RAD_E_SHAPE;
    if (halo_pgeti_or(p, n_p, "group", halo_q6k::kBlock) != halo_q6k::kBlock) return RAD_E_SHAPE;
    g->KB = g->K / halo_q6k::kSub;
    g->KG = g->K / halo_q6k::kBlock;
    return RAD_OK;
}

/* Q6_K's three planes, in lq_ggml's encoding order. */
bool q6k_enc(const RadEncoding* e) {
    if (!e || !rad_enc_is(e, "affine") || e->transform[0]) return false;
    if (e->n_planes != 3) return false;
    const RadEncPlane* c = &e->plane[0];
    if (std::strncmp(c->role, "codes", RAD_ENC_STR) != 0 || c->dtype != RAD_I6 ||
        c->kind != RAD_PLANE_TILED || c->block[0] != 1 || c->block[1] != 1)
        return false;
    const int s = rad_enc_find(e, "scale");
    if (s < 0 || e->plane[s].dtype != RAD_I8 || e->plane[s].kind != RAD_PLANE_TILED ||
        e->plane[s].block[0] != 1 || e->plane[s].block[1] != halo_q6k::kSub)
        return false;
    const int d = rad_enc_find(e, "scale.1");
    return d >= 0 && e->plane[d].dtype == RAD_F16 && e->plane[d].kind == RAD_PLANE_TILED &&
           e->plane[d].block[0] == 1 && e->plane[d].block[1] == halo_q6k::kBlock;
}

void qfill(RadLayout* out, const char* tag, uint32_t dt, long long d0, long long d1,
           long long bytes) {
    std::memset(out, 0, sizeof(*out));
    out->tag = tag;
    out->dtype = dt;
    out->rank = 2;
    out->shape[0] = d0;
    out->shape[1] = d1;
    out->align = 16;
    out->bytes = bytes;
}

int q6k_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
               const RadTensor* pl, int n, RadLayout* out, QGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    QGeom g;
    const int rc = qgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (!q6k_enc(enc)) return RAD_E_DTYPE;
    if (operand == Q6K_B) {
        if (!halo_enc::selects(enc, sel, n, "codes") || !halo_enc::dims(pl[0], g.N, g.K))
            return RAD_E_SHAPE;
        qfill(out, "halo.q6k.w.tile16x64", RAD_I6, g.N, g.K, halo_q6k::code_bytes(g.N, g.K));
    } else if (operand == Q6K_BSCALE) {
        if (!halo_enc::selects(enc, sel, n, "scale") || !halo_enc::dims(pl[0], g.N, g.KB))
            return RAD_E_SHAPE;
        qfill(out, "halo.q6k.s.tile16x16", RAD_I8, g.N, g.KB, halo_q6k::scale_bytes(g.N, g.K));
    } else if (operand == Q6K_BREF) {
        if (!halo_enc::selects(enc, sel, n, "scale.1") || !halo_enc::dims(pl[0], g.N, g.KG))
            return RAD_E_SHAPE;
        qfill(out, "halo.q6k.d.tile16x256", RAD_F16, g.N, g.KG, halo_q6k::d_bytes(g.N, g.K));
    } else {
        return RAD_E_UNSUPPORTED;
    }
    if (gout) *gout = g;
    return RAD_OK;
}

/* Walks every 12-byte sub-block row once: move(stored byte offset, canonical byte offset). The
 * canonical codes row is K*6/8 bytes and sub-block kb starts at 12*kb, so codes never repack. */
template <typename Move>
void walk_codes(const QGeom& g, Move move) {
    const long long NP = g.K / halo_q6k::kPass;
    for (long long n = 0; n < g.N; ++n)
        for (long long kb = 0; kb < g.KB; ++kb) {
            const long long pass = kb / 4, s = kb % 4;
            const long long at = ((n / halo_q6k::kTileRows) * NP + pass) * halo_q6k::kPassBytes +
                                 (n % halo_q6k::kTileRows) * (halo_q6k::kPass * 6 / 8) +
                                 s * halo_q6k::kSubBytes;
            move(at, n * (g.K * 6 / 8) + kb * halo_q6k::kSubBytes);
        }
}
template <typename Move>
void walk_scales(const QGeom& g, Move move) {
    for (long long n = 0; n < g.N; ++n)
        for (long long kb = 0; kb < g.KB; ++kb)
            move(halo_q6k::scale_idx(n, kb, g.K), n * g.KB + kb);
}
template <typename Move>
void walk_d(const QGeom& g, Move move) {
    for (long long n = 0; n < g.N; ++n)
        for (long long kg = 0; kg < g.KG; ++kg)
            move(halo_q6k::d_idx(n, kg, g.K), n * g.KG + kg);
}

}  /* namespace */

extern "C" int halo_rad_layout_q6k(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const RadTensor* pl, int n, RadLayout* out) {
    return q6k_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int halo_rad_relayout_q6k(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, void* dst, int64_t dst_bytes) {
    RadLayout L;
    QGeom g;
    const int rc = q6k_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data || !dst) return RAD_E_SHAPE;
    if (operand == Q6K_BREF) {
        const uint16_t* s = (const uint16_t*)pl[0].data;
        uint16_t* d = (uint16_t*)dst;
        walk_d(g, [&](int64_t at, int64_t from) { d[at] = s[from]; });
    } else if (operand == Q6K_BSCALE) {
        const uint8_t* s = (const uint8_t*)pl[0].data;
        uint8_t* d = (uint8_t*)dst;
        walk_scales(g, [&](int64_t at, int64_t from) { d[at] = s[from]; });
    } else {
        const uint8_t* s = (const uint8_t*)pl[0].data;
        uint8_t* d = (uint8_t*)dst;
        walk_codes(g, [&](int64_t at, int64_t from) {
            std::memcpy(d + at, s + from, halo_q6k::kSubBytes);
        });
    }
    return RAD_OK;
}

extern "C" int halo_rad_unrelayout_q6k(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel, const void* src,
                                       int64_t src_bytes, const RadTensor* pl, int n) {
    RadLayout L;
    QGeom g;
    const int rc = q6k_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data || !src) return RAD_E_SHAPE;
    if (operand == Q6K_BREF) {
        const uint16_t* s = (const uint16_t*)src;
        uint16_t* d = (uint16_t*)pl[0].data;
        walk_d(g, [&](int64_t at, int64_t from) { d[from] = s[at]; });
    } else if (operand == Q6K_BSCALE) {
        const uint8_t* s = (const uint8_t*)src;
        uint8_t* d = (uint8_t*)pl[0].data;
        walk_scales(g, [&](int64_t at, int64_t from) { d[from] = s[at]; });
    } else {
        const uint8_t* s = (const uint8_t*)src;
        uint8_t* d = (uint8_t*)pl[0].data;
        walk_codes(g, [&](int64_t at, int64_t from) {
            std::memcpy(d + from, s + at, halo_q6k::kSubBytes);
        });
    }
    return RAD_OK;
}
