/* halo_q80_layout.cpp -- the stored form of ggml Q8_0 dense weights ("halo.q80.tile16x32", see
 * halo_q80.h) and the relayouts that produce it from the weight's canonical planes (spec §4.3).
 * Pure host C++: nothing here includes halo.h or <hip/hip_runtime.h>.
 *
 * The canonical planes are `codes` i8 [N][K] and `scale` f16 [N][K/32] (affine, scale block 1x32),
 * which are the GGUF block fields unchanged. The relayout only moves bytes: 16-row by 32-K tiles of
 * codes, and the scales in the same tile order. Both hooks are exact inverses.
 *
 * Guards. The format addresses K in whole 32-blocks and N in whole 16-row tiles; a shape outside
 * that has no tile to hold its last rows or columns, so layout, relayout and unrelayout all refuse
 * it with RAD_E_SHAPE before touching memory (an earlier family's guard accepted a K its stored
 * form could not address). The destination size is checked against the layout's exactly, the
 * plane extents against N, K and K/32, and N*K is bounded so every offset fits int64 with room.
 */
#include "halo_plugin.h"
#include "halo_enc.h"
#include "halo_q80.h"

#include <cstdint>
#include <cstring>

enum { Q80_A = 0, Q80_ASCALE = 1, Q80_B = 2, Q80_BSCALE = 3, Q80_Y = 4 };

namespace {

struct QGeom { long long N, K, KB; };

int qgeom(const RadParam* p, int n_p, QGeom* g) {
    g->N = halo_pgeti_or(p, n_p, "N", 0);
    g->K = halo_pgeti_or(p, n_p, "K", 0);
    if (!halo_q80::shape_ok(g->N, g->K)) return RAD_E_SHAPE;
    if (halo_pgeti_or(p, n_p, "group", halo_q80::kBlock) != halo_q80::kBlock) return RAD_E_SHAPE;
    g->KB = g->K / halo_q80::kBlock;
    return RAD_OK;
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

int q80_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
               const RadTensor* pl, int n, RadLayout* out, QGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    QGeom g;
    const int rc = qgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (operand != Q80_B && operand != Q80_BSCALE) return RAD_E_UNSUPPORTED;
    if (!halo_enc::affine(enc, RAD_I8, RAD_F16, 1, halo_q80::kBlock)) return RAD_E_DTYPE;
    if (operand == Q80_BSCALE) {
        if (!halo_enc::selects(enc, sel, n, "scale") || !halo_enc::dims(pl[0], g.N, g.KB))
            return RAD_E_SHAPE;
        qfill(out, "halo.q80.s.tile16x32", RAD_F16, g.N, g.KB, halo_q80::scale_bytes(g.N, g.K));
    } else {
        if (!halo_enc::selects(enc, sel, n, "codes") || !halo_enc::dims(pl[0], g.N, g.K))
            return RAD_E_SHAPE;
        qfill(out, "halo.q80.w.tile16x32", RAD_I8, g.N, g.K, halo_q80::code_bytes(g.N, g.K));
    }
    if (gout) *gout = g;
    return RAD_OK;
}

/* Walks every element once: move(stored offset, canonical offset), in elements of `esz` bytes. A
 * tile row is 32 contiguous codes (or one scale), so codes move 32 bytes at a time. */
template <typename Move>
void walk_codes(const QGeom& g, Move move) {
    for (long long n = 0; n < g.N; ++n)
        for (long long kb = 0; kb < g.KB; ++kb)
            move(halo_q80::code_off(n, kb * halo_q80::kBlock, g.K), (n * g.KB + kb) * halo_q80::kBlock);
}
template <typename Move>
void walk_scales(const QGeom& g, Move move) {
    for (long long n = 0; n < g.N; ++n)
        for (long long kb = 0; kb < g.KB; ++kb)
            move(halo_q80::scale_idx(n, kb, g.K), n * g.KB + kb);
}

}  /* namespace */

extern "C" int halo_rad_layout_q80(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const RadTensor* pl, int n, RadLayout* out) {
    return q80_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int halo_rad_relayout_q80(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, void* dst, int64_t dst_bytes) {
    RadLayout L;
    QGeom g;
    const int rc = q80_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data || !dst) return RAD_E_SHAPE;
    if (operand == Q80_BSCALE) {
        const uint16_t* s = (const uint16_t*)pl[0].data;
        uint16_t* d = (uint16_t*)dst;
        walk_scales(g, [&](int64_t at, int64_t from) { d[at] = s[from]; });
    } else {
        const uint8_t* s = (const uint8_t*)pl[0].data;
        uint8_t* d = (uint8_t*)dst;
        walk_codes(g, [&](int64_t at, int64_t from) { std::memcpy(d + at, s + from, halo_q80::kBlock); });
    }
    return RAD_OK;
}

extern "C" int halo_rad_unrelayout_q80(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel, const void* src,
                                       int64_t src_bytes, const RadTensor* pl, int n) {
    RadLayout L;
    QGeom g;
    const int rc = q80_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data || !src) return RAD_E_SHAPE;
    if (operand == Q80_BSCALE) {
        const uint16_t* s = (const uint16_t*)src;
        uint16_t* d = (uint16_t*)pl[0].data;
        walk_scales(g, [&](int64_t at, int64_t from) { d[from] = s[at]; });
    } else {
        const uint8_t* s = (const uint8_t*)src;
        uint8_t* d = (uint8_t*)pl[0].data;
        walk_codes(g, [&](int64_t at, int64_t from) { std::memcpy(d + from, s + at, halo_q80::kBlock); });
    }
    return RAD_OK;
}
