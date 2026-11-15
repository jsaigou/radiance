/* halo_moe_unrot_bench -- the three unrotated expert forms (W4 6/7/8) against the rotated codebook
 * forms (W4 2/4/5) at Flash-Next expert geometry: layout-hook checks, a numeric check against a
 * host decode, and decode/prefill timings with GB/s of expert bytes.
 *
 * Not part of the plugin and not built by CMake; build it beside a built libhalo:
 *
 *   hipcc -x hip --offload-arch=gfx1151 -O2 -std=gnu++17 -ffp-contract=off -I abi -I libhalo \
 *       libhalo/halo_moe_unrot_bench.hip -o halo_moe_unrot_bench -ldl -lamdhip64
 *
 *   halo_moe_unrot_bench <libhalo.so> layout
 *   halo_moe_unrot_bench <libhalo.so> check      [reps]
 *   halo_moe_unrot_bench <libhalo.so> bench      [reps [trials]]
 *
 * Formats, in the op's own numbering:
 *
 *   iq4nl32a16  W4 6   u4 codes into llama.cpp's 16-level kvalues_iq4nl, f16 per 32, bf16 A
 *   q50a16      W4 7   i5 two's-complement codes, f16 per 32, bf16 A
 *   q20a16      W4 8   u2 codes naming -1/0/+1/+2, f16 per 64, bf16 A
 *   w4nl64a8h   W4 2   the w4nl codebook, E4M3 per 64 under kW4G64Scale, rotated A8
 *   w5nl64a8h   W4 4   the w5nl codebook (32 levels), same scale grid, rotated A8
 *   w3nl64a8h   W4 5   the w3nl codebook (8 levels), same scale grid, rotated A8
 *
 * Shapes: a projected-expert slice at the production two-rank geometry, gate_up (N 640, K 5120)
 * and down (N 5120, K 1280), and the single-rank whole expert (gate_up 1280 x 2560, down 2560 x
 * 640). 512 experts, top_k 10. M = 1, 4, 32 are decode-band token counts; a spread routing puts
 * 1-2 rows on every one of the 512 experts at T 760 (still the decode band); M = 128 and 512 are
 * prefill chunks. `bench` prints us, GB/s of expert bytes and the active-expert count per cell.
 *
 * Weights are one expert's stored plane repeated into 512 distinct addresses, so the read traffic
 * is real (values do not affect the kernel's arithmetic or traffic). `peak` (also run by `bench`)
 * measures a plain streaming read on the same device for the DRAM ceiling.
 */
#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_args.h"
#include "rad_encoding.h"

#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

static int g_fail = 0;
static void ok(const std::string& what, bool c, const std::string& d = "") {
    if (!c) ++g_fail;
    std::printf("  %s  %s%s%s\n", c ? "ok  " : "FAIL", what.c_str(), d.empty() ? "" : " -- ", d.c_str());
}

/* ------------------------------------------------------------------ the plugin */
struct Plugin {
    void* h = nullptr;
    int (*kernel_count)(void) = nullptr;
    const RadKernelInfo* (*kernel_at)(int) = nullptr;
    void (*pin_moe)(int) = nullptr;
    void (*pin_fk)(int) = nullptr;
    void (*pin_ddec)(int) = nullptr;
    void (*pin_dtile)(int) = nullptr;
};
template <class F> static bool sym(void* h, const char* n, F* out) {
    void* p = dlsym(h, n);
    if (!p) return false;
    std::memcpy(out, &p, sizeof p);
    return true;
}
static bool load_plugin(Plugin* pl, const char* path) {
    pl->h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!pl->h) { std::printf("dlopen(%s): %s\n", path, dlerror()); return false; }
    bool k = sym(pl->h, "rad_kernel_count", &pl->kernel_count) &&
             sym(pl->h, "rad_kernel_at", &pl->kernel_at);
    sym(pl->h, "halo_moe_gemm_pin_prefill", &pl->pin_moe);
    sym(pl->h, "halo_moe_gemm_pin_fk", &pl->pin_fk);
    sym(pl->h, "halo_moe_gemm_pin_down_dec", &pl->pin_ddec);
    sym(pl->h, "halo_moe_gemm_pin_down_tile", &pl->pin_dtile);
    return k;
}
static const RadKernelInfo* find(const Plugin& pl, const char* name) {
    for (int k = 0; k < pl.kernel_count(); ++k)
        if (!std::strcmp(pl.kernel_at(k)->name, name)) return pl.kernel_at(k);
    return nullptr;
}

/* ------------------------------------------------------------------ small helpers */
static RadParam PI(const char* k, long long v) {
    RadParam p; std::memset(&p, 0, sizeof p); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p;
}
static RadParam PS(const char* k, const char* v) {
    RadParam p; std::memset(&p, 0, sizeof p); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p;
}
static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = d; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t acc = 1;
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
    return t;
}
static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static float h_e4m3(uint8_t b) {
    const float sign = (b & 0x80u) ? -1.0f : 1.0f;
    const uint32_t ex = (uint32_t)(b >> 3) & 0x0Fu, mn = (uint32_t)b & 0x07u;
    if (ex == 0) return sign * (float)mn * (1.0f / 512.0f);
    return sign * (1.0f + (float)mn / 8.0f) * std::ldexp(1.0f, (int)ex - 7);
}
static uint16_t h_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float h_bf16(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f;
}
static float h_f16(uint16_t h) { return rad_load_f32(&h, RAD_F16, 0); }
static void* dalloc(size_t bytes) {
    void* p = nullptr;
    HCK(hipMalloc(&p, bytes ? bytes : 16));
    return p;
}
static void* dup_to_dev(const void* src, size_t bytes) {
    void* d = dalloc(bytes);
    HCK(hipMemcpy(d, src, bytes, hipMemcpyHostToDevice));
    return d;
}

/* ------------------------------------------------------------------ the six formats */
enum FKind { K_IQ4, K_Q50, K_Q20, K_W4, K_W5, K_W3 };
struct Fmt {
    const char* name;
    FKind       kind;
    uint32_t    codes;     /* the canonical code plane's dtype */
    uint32_t    wdt;       /* the stored weight plane's dtype the kernel is handed */
    uint32_t    sdt;       /* the stored scale plane's dtype */
    int         bits;      /* bits a canonical code occupies */
    int         gs;        /* K elements a scale covers */
    int         nt;        /* table entries (0: no table) */
    const float* ftab;     /* the rotated codebook (null for the unrotated forms) */
    bool        rotated;
};
static const Fmt kIQ4 = { "iq4nl32a16", K_IQ4, RAD_U4, HALO_DT_MOEW4, RAD_F16, 4, 32, 16, nullptr, false };
static const Fmt kQ50 = { "q50a16",     K_Q50, RAD_I5, HALO_DT_MOEW5, RAD_F16, 5, 32,  0, nullptr, false };
static const Fmt kQ20 = { "q20a16",     K_Q20, RAD_U2, RAD_U2,        RAD_F16, 2, 64,  0, nullptr, false };
static const Fmt kW4  = { "w4nl64a8h",  K_W4,  RAD_U4, HALO_DT_MOEW4, RAD_F8E4M3, 4, 64, 16, kW4nlTable, true };
static const Fmt kW5  = { "w5nl64a8h",  K_W5,  RAD_U5, HALO_DT_MOEW5, RAD_F8E4M3, 5, 64, 32, kW5nlTable, true };
static const Fmt kW3  = { "w3nl64a8h",  K_W3,  RAD_U3, HALO_DT_MOEW3, RAD_F8E4M3, 3, 64,  8, kW3nlTable, true };
static const Fmt* const kFmts[6] = { &kIQ4, &kQ50, &kQ20, &kW4, &kW5, &kW3 };

static bool is_iq4(const Fmt& F) { return F.kind == K_IQ4; }
static bool unrot_asis(const Fmt& F) { return F.kind == K_Q20; }
static bool q50signed(const Fmt& F) { return F.kind == K_Q50; }

/* ------------------------------------------------------------------ canonical planes */
struct Planes {
    RadEncoding enc{};
    std::vector<std::vector<uint8_t>> data;
    std::vector<int64_t> rows, cols;
    int find(const char* role) const { return rad_enc_find(&enc, role); }
    int64_t row_bytes(int i) const { return rad_enc_row_bytes(enc.plane[i].dtype, cols[(size_t)i]); }
    RadTensor t(int i) const {
        RadTensor x;
        std::memset(&x, 0, sizeof x);
        x.data = const_cast<uint8_t*>(data[(size_t)i].data());
        x.dtype = enc.plane[i].dtype;
        x.rank = 2;
        x.shape[0] = rows[(size_t)i];
        x.shape[1] = cols[(size_t)i];
        rad_tensor_pack(&x);
        return x;
    }
    void size(int64_t n, int64_t k) {
        data.assign((size_t)enc.n_planes, {});
        rows.assign((size_t)enc.n_planes, 0);
        cols.assign((size_t)enc.n_planes, 0);
        for (int i = 0; i < enc.n_planes; ++i) {
            rad_enc_plane_dims(&enc.plane[i], n, k, &rows[(size_t)i], &cols[(size_t)i]);
            data[(size_t)i].assign((size_t)(rows[(size_t)i] * row_bytes(i)), 0);
        }
    }
};

/* One expert's canonical planes. Rotated forms carry the w4nl/w5nl/w3nl codebook, an E4M3 scale
 * per 64 under the constant second level and the fwht128 transform. The unrotated forms carry the
 * GGUF plane as stored: u4/i5/u2 codes and an f16 scale per 32 or 64, no rotation; IQ4_NL also
 * carries its i8 table. */
static Planes make_planes(const Fmt& F, int64_t N, int64_t K, uint32_t seed) {
    Planes p;
    rad_enc_clear(&p.enc);
    rad_enc_copy_str(p.enc.scheme, "affine");
    if (F.rotated) rad_enc_copy_str(p.enc.transform, "fwht128");
    rad_enc_add_plane(&p.enc, "codes", F.codes, 1, 1);
    rad_enc_add_plane(&p.enc, "scale", F.sdt, 1, F.gs);
    if (F.rotated) {
        rad_enc_add_plane(&p.enc, "scale.1", RAD_F32, 0, 0);
        rad_enc_add_table(&p.enc, "table", RAD_F32, 1, F.nt);
    } else if (is_iq4(F)) {
        rad_enc_add_table(&p.enc, "table", RAD_I8, 1, F.nt);
    }
    p.size(N, K);
    const int ci = p.find("codes"), si = p.find("scale");
    const int64_t crow = p.row_bytes(ci);
    uint8_t* cdat = p.data[(size_t)ci].data();
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t x = mix32((uint32_t)(n * K + k) ^ seed);
            int code = 0;
            switch (F.kind) {
                case K_IQ4: case K_W4: code = (int)(x % 16u); break;
                case K_Q50: case K_W5: code = (int)(x % 32u) - 16; break;
                case K_W3:  code = (int)(x % 8u); break;
                case K_Q20: code = (int)(x % 4u); break;
            }
            rad_store_code(cdat + n * crow, F.codes, k, code);
        }
    }
    const int64_t ng = K / F.gs, srow = p.row_bytes(si);
    uint8_t* sdat = p.data[(size_t)si].data();
    uint16_t* s16 = (uint16_t*)sdat;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < ng; ++g) {
            const uint32_t x = mix32((uint32_t)(n * ng + g) ^ (seed * 7u + 1u));
            if (F.rotated) sdat[(size_t)(n * srow + g)] = (uint8_t)(0x30u + x % 32u);
            else           s16[(size_t)(n * ng + g)] = (uint16_t)(0x2C00u + (x & 0x0FFFu));
        }
    if (F.rotated) {
        std::memcpy(p.data[(size_t)p.find("scale.1")].data(), &kW4G64Scale, sizeof kW4G64Scale);
        std::memcpy(p.data[(size_t)p.find("table")].data(), F.ftab, sizeof(float) * (size_t)F.nt);
    } else if (is_iq4(F)) {
        std::memcpy(p.data[(size_t)p.find("table")].data(), kIq4NlTable, 16);
    }
    return p;
}

static std::vector<RadParam> moe_lp(const Fmt& F, int64_t N, int64_t K) {
    return { PI("N", N), PI("K", K), PI("N_odd", N), PI("K_odd", K), PI("N_1", N), PI("K_1", K),
             PI("N_2", N), PI("K_2", K), PI("N_3", N), PI("K_3", K), PI("group", 128),
             PS("dtype", F.name) };
}

static int stored(const RadKernelInfo* r, const std::vector<RadParam>& lp, int operand,
                  const Planes& p, std::initializer_list<const char*> roles,
                  std::vector<uint8_t>* out) {
    int sel[RAD_ENC_MAX_PLANES];
    RadTensor pt[RAD_ENC_MAX_PLANES];
    int n = 0;
    for (const char* role : roles) {
        sel[n] = p.find(role);
        if (sel[n] < 0) return RAD_E_SHAPE;
        pt[n] = p.t(sel[n]);
        ++n;
    }
    RadLayout L;
    std::memset(&L, 0, sizeof L);
    const int rc = r->layout(lp.data(), (int)lp.size(), operand, &p.enc, sel, pt, n, &L);
    if (rc != RAD_OK) return rc;
    out->assign((size_t)L.bytes, 0);
    return r->relayout(lp.data(), (int)lp.size(), operand, &p.enc, sel, pt, n, out->data(), L.bytes);
}

/* One expert's stored weight and scale planes. The hook gives the kernel's packing for IQ4_NL,
 * Q5_0 and the rotated forms; Q2_0's canonical u2 plane and every unrotated f16 scale plane are
 * stored as they are (the hook answers RAD_E_UNSUPPORTED, which is the "as stored" answer here). */
static bool stored_one(const Plugin& pl, const Fmt& F, int64_t N, int64_t K, uint32_t seed,
                       const RadKernelInfo* r, Planes* canon, std::vector<uint8_t>* w,
                       std::vector<uint8_t>* s, bool verbose) {
    (void)pl;
    *canon = make_planes(F, N, K, seed);
    const std::vector<RadParam> lp = moe_lp(F, N, K);
    const int ci = canon->find("codes"), si = canon->find("scale");
    if (F.rotated) {
        const int rw = stored(r, lp, 2, *canon, { "codes", "table" }, w);
        const int rs = stored(r, lp, 3, *canon, { "scale", "scale.1" }, s);
        if (rw != RAD_OK || rs != RAD_OK) {
            if (verbose) std::printf("  %s relayout rc %d/%d\n", F.name, rw, rs);
            return false;
        }
    } else if (is_iq4(F)) {
        const int rw = stored(r, lp, 2, *canon, { "codes", "table" }, w);
        const int rs = stored(r, lp, 3, *canon, { "scale" }, s);
        if (rw != RAD_OK) { if (verbose) std::printf("  %s weight relayout rc %d\n", F.name, rw); return false; }
        if (rs != RAD_E_UNSUPPORTED) { if (verbose) std::printf("  %s scale relayout rc %d\n", F.name, rs); return false; }
        *s = canon->data[(size_t)si];
    } else if (q50signed(F)) {
        const int rw = stored(r, lp, 2, *canon, { "codes" }, w);
        const int rs = stored(r, lp, 3, *canon, { "scale" }, s);
        if (rw != RAD_OK) { if (verbose) std::printf("  %s weight relayout rc %d\n", F.name, rw); return false; }
        if (rs != RAD_E_UNSUPPORTED) { if (verbose) std::printf("  %s scale relayout rc %d\n", F.name, rs); return false; }
        *s = canon->data[(size_t)si];
    } else if (unrot_asis(F)) {
        const int rw = stored(r, lp, 2, *canon, { "codes" }, w);
        const int rs = stored(r, lp, 3, *canon, { "scale" }, s);
        if (rw != RAD_E_UNSUPPORTED) { if (verbose) std::printf("  %s weight layout rc %d\n", F.name, rw); return false; }
        if (rs != RAD_E_UNSUPPORTED) { if (verbose) std::printf("  %s scale layout rc %d\n", F.name, rs); return false; }
        *w = canon->data[(size_t)ci];
        *s = canon->data[(size_t)si];
    }
    const int64_t wbytes = F.rotated ? (F.kind == K_W5 ? N * (K / 8) * 5
                                                      : F.kind == K_W3 ? N * (K / 8) * 3
                                                                       : N * K / 2)
                           : F.kind == K_Q50 ? N * (K / 8) * 5
                           : F.kind == K_Q20 ? N * K / 4
                                             : N * K / 2;
    const int64_t sbytes = F.rotated ? N * (K / 64) : N * (K / F.gs) * 2;
    if ((int64_t)w->size() != wbytes || (int64_t)s->size() != sbytes) {
        if (verbose)
            std::printf("  %s stored bytes %zu/%zu, want %lld/%lld\n", F.name, w->size(), s->size(),
                        (long long)wbytes, (long long)sbytes);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ routing */
static void route(int64_t M, int64_t top_k, int64_t ne, uint64_t seed, bool drops,
                  std::vector<int32_t>& off, std::vector<int32_t>& sorted) {
    const int64_t Tn = M * top_k;
    std::vector<int32_t> ids((size_t)Tn, -1);
    uint64_t s = seed;
    auto rnd = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(s >> 33); };
    for (int64_t m = 0; m < M; ++m)
        for (int64_t k = 0; k < top_k; ++k) {
            int32_t e = 0;
            bool dup = true;
            while (dup) {
                const uint32_t x = rnd();
                e = (int32_t)((x >> 1) % (uint32_t)ne);
                dup = false;
                for (int64_t q = 0; q < k; ++q) dup = dup || ids[(size_t)(m * top_k + q)] == e;
            }
            ids[(size_t)(m * top_k + k)] = e;
        }
    if (drops)
        for (int64_t i = 5; i < Tn; i += 37) ids[(size_t)i] = -1;
    std::vector<int32_t> cnt((size_t)ne, 0);
    for (const int32_t e : ids) if (e >= 0) ++cnt[(size_t)e];
    off.assign((size_t)(ne + 1), 0);
    for (int64_t e = 0; e < ne; ++e) off[(size_t)(e + 1)] = off[(size_t)e] + cnt[(size_t)e];
    sorted.assign((size_t)Tn, -1);
    std::vector<int32_t> at(off.begin(), off.end() - 1);
    for (int64_t i = 0; i < Tn; ++i)
        if (ids[(size_t)i] >= 0) sorted[(size_t)(at[(size_t)ids[(size_t)i]]++)] = (int32_t)i;
}

/* Every expert gets one or two rows: slot s names expert s % ne, so a token's ten slots are ten
 * consecutive experts. T 760 stays inside the decode band (2T <= 3 ne). */
static void route_spread(int64_t T, int64_t top_k, int64_t ne, std::vector<int32_t>& off,
                         std::vector<int32_t>& sorted) {
    std::vector<std::vector<int32_t>> by((size_t)ne);
    for (int64_t s = 0; s < T; ++s) by[(size_t)(s % ne)].push_back((int32_t)s);
    off.assign((size_t)(ne + 1), 0);
    for (int64_t e = 0; e < ne; ++e) off[(size_t)(e + 1)] = off[(size_t)e] + (int32_t)by[(size_t)e].size();
    sorted.clear();
    for (int64_t e = 0; e < ne; ++e) sorted.insert(sorted.end(), by[(size_t)e].begin(), by[(size_t)e].end());
}

/* ------------------------------------------------------------------ a projection's device state */
struct Bank {
    const Fmt* F = nullptr;
    int64_t N = 0, K = 0, ne = 0, top_k = 10;
    bool sorted_op = false;
    int64_t nec[2] = { 0, 0 };
    void* d_w[2] = {}; void* d_s[2] = {};
    size_t wbytes = 0, sbytes = 0;
    const RadKernelInfo* row = nullptr;
    double bytes_expert = 0.0;
};

/* `replicate`: one stored expert copied into every class slot (real addresses, real traffic). When
 * false the caller has already supplied each expert's stored bytes in `w`/`s` (the check bank). */
static bool build_bank(const Plugin& pl, Bank& B, const Fmt& F, int64_t N, int64_t K, int64_t ne,
                       int64_t top_k, bool sorted_op, bool replicate,
                       const std::vector<std::vector<uint8_t>>* wq = nullptr,
                       const std::vector<std::vector<uint8_t>>* sq = nullptr) {
    B.F = &F; B.N = N; B.K = K; B.ne = ne; B.top_k = top_k; B.sorted_op = sorted_op;
    B.row = find(pl, "moe_gemm_w4a8");
    if (!B.row) return false;
    B.nec[0] = (ne + 1) / 2;
    B.nec[1] = ne / 2;
    Planes canon;
    std::vector<uint8_t> w, s;
    if (replicate) {
        if (!stored_one(pl, F, N, K, 0x77u + (uint32_t)N, B.row, &canon, &w, &s, true)) return false;
        B.wbytes = w.size();
        B.sbytes = s.size();
    } else {
        if (!wq || !sq) return false;
        B.wbytes = wq[0][0].size();
        B.sbytes = sq[0][0].size();
    }
    B.bytes_expert = (double)(B.wbytes + B.sbytes);
    for (int c = 0; c < 2; ++c) {
        std::vector<uint8_t> wc((size_t)B.nec[c] * B.wbytes);
        std::vector<uint8_t> sc((size_t)B.nec[c] * B.sbytes);
        for (int64_t x = 0; x < B.nec[c]; ++x) {
            const uint8_t* wp = replicate ? w.data() : wq[c][(size_t)x].data();
            const uint8_t* sp = replicate ? s.data() : sq[c][(size_t)x].data();
            std::memcpy(wc.data() + (size_t)x * B.wbytes, wp, B.wbytes);
            std::memcpy(sc.data() + (size_t)x * B.sbytes, sp, B.sbytes);
        }
        B.d_w[c] = dup_to_dev(wc.data(), wc.size());
        B.d_s[c] = dup_to_dev(sc.data(), sc.size());
    }
    return true;
}
static void free_bank(Bank& B) {
    for (int c = 0; c < 2; ++c) { if (B.d_w[c]) hipFree(B.d_w[c]); if (B.d_s[c]) hipFree(B.d_s[c]); B.d_w[c] = B.d_s[c] = nullptr; }
}

/* ------------------------------------------------------------------ calls */
struct Call {
    std::vector<RadTensor> ts;
    std::vector<RadParam> ps;
    RadArgs a;
    void* d_a = nullptr; void* d_as = nullptr; void* d_sorted = nullptr; void* d_off = nullptr;
    void* d_y = nullptr; void* d_map = nullptr;
    int64_t Tr = 0, M = 0, active = 0;
};
static void make_call(const Bank& B, int64_t M, const std::vector<int32_t>& off,
                      const std::vector<int32_t>& sorted, const std::vector<uint8_t>& abytes,
                      const std::vector<float>& ascales, Call& c) {
    const Fmt& F = *B.F;
    const int64_t Tr = M * B.top_k, rows = B.sorted_op ? Tr : M;
    c.Tr = Tr; c.M = M;
    c.d_a = dup_to_dev(abytes.data(), abytes.size());
    if (!ascales.empty()) c.d_as = dup_to_dev(ascales.data(), ascales.size() * 4);
    c.d_sorted = dup_to_dev(sorted.data(), sorted.size() * 4);
    c.d_off = dup_to_dev(off.data(), off.size() * 4);
    c.d_y = dalloc((size_t)(Tr * B.N * 2));
    const int64_t cap = B.ne + Tr / 16 + 8;
    c.d_map = dalloc((size_t)((1 + 2 * cap) * 4 + 4096));
    c.active = 0;
    for (int64_t e = 0; e < B.ne; ++e) c.active += off[(size_t)(e + 1)] > off[(size_t)e];
    auto codes_t = [&](int q) { return T(B.d_w[q], F.wdt, { B.nec[q], B.N, B.K }); };
    auto scale_t = [&](int q) { return T(B.d_s[q], F.sdt, { B.nec[q], B.N, B.K / F.gs }); };
    if (F.rotated) {
        c.ts = { T(c.d_a, RAD_F8E4M3, { rows, B.K }), T(c.d_as, RAD_F32, { rows, B.K / 128 }),
                 codes_t(0), scale_t(0), T(c.d_sorted, RAD_I32, { Tr }), T(c.d_off, RAD_I32, { B.ne + 1 }),
                 codes_t(1), scale_t(1), T(c.d_y, RAD_BF16, { Tr, B.N }) };
    } else {
        c.ts = { T(c.d_a, RAD_BF16, { rows, B.K }), T(nullptr, RAD_F32, { 1, 1 }),
                 codes_t(0), scale_t(0), T(c.d_sorted, RAD_I32, { Tr }), T(c.d_off, RAD_I32, { B.ne + 1 }),
                 codes_t(1), scale_t(1), T(c.d_y, RAD_BF16, { Tr, B.N }) };
    }
    c.ps = { PI("M", M), PI("N", B.N), PI("K", B.K), PI("N_odd", B.N), PI("K_odd", B.K),
             PI("parts", 1), PI("n_expert", B.ne), PI("top_k", B.top_k), PI("group", 128),
             PS("a_order", B.sorted_op ? "sorted" : "token"), PS("dtype", F.name) };
    std::memset(&c.a, 0, sizeof c.a);
    c.a.t = c.ts.data(); c.a.n_t = (int)c.ts.size();
    c.a.p = c.ps.data(); c.a.n_p = (int)c.ps.size();
    c.a.scratch = c.d_map; c.a.scratch_bytes = (1 + 2 * cap) * 4 + 4096;
}
static void free_call(Call& c) {
    for (void* p : { c.d_a, c.d_as, c.d_sorted, c.d_off, c.d_y, c.d_map }) if (p) hipFree(p);
    c = Call{};
}

/* ------------------------------------------------------------------ activations */
/* Unrotated forms read bf16 rows. Small integers times 1/4 keep every product exact in f16, which
 * is only for the check; the bench uses arbitrary bf16 patterns. */
static void draw_act_bf16(int64_t rows, int64_t K, uint32_t seed, std::vector<uint8_t>* out) {
    out->resize((size_t)(rows * K) * 2);
    uint16_t* v = (uint16_t*)out->data();
    for (int64_t m = 0; m < rows; ++m)
        for (int64_t k = 0; k < K; ++k)
            v[(size_t)(m * K + k)] = h_to_bf16((float)(((mix32((uint32_t)(m * K + k) ^ seed)) % 97u) - 48) * 0.25f);
}
/* Rotated forms read E4M3 rows plus an f32 scale per 128. E4M3 bytes with the exponent field 6..9
 * are 0.5..7.5; the scale grid is the real one. */
static void draw_act_e4m3(int64_t rows, int64_t K, uint32_t seed, std::vector<uint8_t>* aq,
                          std::vector<float>* as) {
    const int64_t kb = K / 128;
    aq->resize((size_t)(rows * K));
    as->resize((size_t)(rows * kb));
    for (int64_t m = 0; m < rows; ++m) {
        for (int64_t g = 0; g < kb; ++g)
            (*as)[(size_t)(m * kb + g)] = 0.011f + 0.003f * (float)g + 0.0007f * (float)(m % 11);
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t x = mix32((uint32_t)(m * K + k) ^ seed);
            const uint32_t ex = 6 + (x >> 8) % 4, mn = (x >> 4) & 7u;
            (*aq)[(size_t)(m * K + k)] = (uint8_t)(((x & 1u) ? 0x80u : 0u) | (ex << 3) | mn);
        }
    }
}

/* ------------------------------------------------------------------ the host model */
/* One output element, the way the kernel forms it: the canonical level times the activation, the
 * weight scale folded per its grid (and, for the rotated forms, the activation scale per 128).
 * Grouping differs from the kernel's f32 chain only by f32 summation order. */
static double host_elem(const Fmt& F, const Planes& pl, int64_t n, int64_t K,
                        const uint8_t* ae4, const uint16_t* abf, const float* as) {
    const int ci = pl.find("codes"), si = pl.find("scale");
    const int64_t crow = pl.row_bytes(ci);
    const uint8_t* cdat = pl.data[(size_t)ci].data() + n * crow;
    if (!F.rotated) {
        const int64_t ng = K / F.gs;
        const uint8_t* sdat = pl.data[(size_t)si].data();
        const uint16_t* s16 = (const uint16_t*)sdat;
        double out = 0.0;
        for (int64_t k = 0; k < K; ++k) {
            const int code = (int)rad_load_bits(cdat, F.bits, k);
            double level;
            if (is_iq4(F))      level = (double)kIq4NlTable[code];
            else if (q50signed(F)) level = (double)(code >= 16 ? code - 32 : code);
            else                level = (double)(code - 1);
            out += level * (double)h_bf16(abf[k]) * (double)h_f16(s16[n * ng + k / F.gs]);
        }
        return out;
    }
    const int64_t nkb = K / 128;
    const uint8_t* sdat = pl.data[(size_t)si].data();
    double out = 0.0;
    for (int64_t kb = 0; kb < nkb; ++kb) {
        double v = 0.0;
        for (int64_t q = 0; q < 2; ++q) {
            double raw = 0.0;
            for (int64_t j = 0; j < 64; ++j) {
                const int64_t k = kb * 128 + q * 64 + j;
                raw += (double)F.ftab[(int)rad_load_bits(cdat, F.bits, k)] * (double)h_e4m3(ae4[k]);
            }
            v += raw * (double)h_e4m3(sdat[n * (2 * nkb) + kb * 2 + q]) * (double)kW4G64Scale;
        }
        out += v * (double)as[kb];
    }
    return out;
}

static std::vector<uint16_t> run_launch(const Plugin& pl, Call& c, size_t elems, int form,
                                        const RadKernelInfo* row, int* rc) {
    std::vector<uint16_t> junk(elems, 0x7f7fu);
    HCK(hipMemcpy(c.d_y, junk.data(), elems * 2, hipMemcpyHostToDevice));
    if (pl.pin_moe) pl.pin_moe(form);
    *rc = row->launch(&c.a, nullptr);
    HCK(hipStreamSynchronize(nullptr));
    if (pl.pin_moe) pl.pin_moe(0);
    std::vector<uint16_t> got(elems);
    HCK(hipMemcpy(got.data(), c.d_y, elems * 2, hipMemcpyDeviceToHost));
    return got;
}

/* ------------------------------------------------------------------ check */
static int64_t expert_of(const std::vector<int32_t>& off, int64_t i) {
    for (size_t e = 0; e + 1 < off.size(); ++e)
        if (i >= off[e] && i < off[e + 1]) return (int64_t)e;
    return -1;
}

static void check_case(const Plugin& pl, const Fmt& F, const char* label, int64_t N, int64_t K,
                       bool sorted_op, const std::vector<int64_t>& Ms) {
    const int64_t ne = 8, top_k = 2;
    const int64_t nec[2] = { (ne + 1) / 2, ne / 2 };
    const RadKernelInfo* row = find(pl, "moe_gemm_w4a8");
    if (!row) { ok(std::string(F.name) + " " + label, false, "row absent"); return; }
    std::vector<std::vector<uint8_t>> wq[2], sq[2];
    std::vector<Planes> cp[2];
    for (int c = 0; c < 2; ++c)
        for (int64_t x = 0; x < nec[c]; ++x) {
            const uint32_t e = (uint32_t)(2 * x + c);
            Planes canon;
            std::vector<uint8_t> w, s;
            if (!stored_one(pl, F, N, K, 0x1000u + e, row, &canon, &w, &s, true)) {
                ok(std::string(F.name) + " " + label, false, "build");
                return;
            }
            cp[c].push_back(canon);
            wq[c].push_back(w);
            sq[c].push_back(s);
        }
    Bank B;
    if (!build_bank(pl, B, F, N, K, ne, top_k, sorted_op, false, wq, sq)) {
        ok(std::string(F.name) + " " + label, false, "bank");
        return;
    }
    const RadKernelInfo* r = B.row;
    for (int64_t M : Ms) {
        const int64_t Tr = M * top_k, rows = sorted_op ? Tr : M;
        std::vector<int32_t> off, sorted;
        route(M, top_k, ne, 0x9E3779B97F4A7C15ull ^ (uint64_t)Tr, true, off, sorted);
        const int64_t live = off[(size_t)ne];
        std::vector<uint8_t> abytes;
        std::vector<float> ascales;
        if (F.rotated) draw_act_e4m3(rows, K, 0xABCDu + (uint32_t)M, &abytes, &ascales);
        else           draw_act_bf16(rows, K, 0xABCDu + (uint32_t)M, &abytes);
        const uint16_t* abf = F.rotated ? nullptr : (const uint16_t*)abytes.data();

        std::vector<double> ref((size_t)(Tr * N), 0.0);
        for (int64_t i = 0; i < live; ++i) {
            const int32_t f = sorted[(size_t)i];
            if (f < 0) continue;
            const int64_t e = expert_of(off, i);
            if (e < 0) continue;
            const int64_t arow = sorted_op ? i : f / top_k;
            const Planes& q = cp[e & 1][(size_t)(e / 2)];
            for (int64_t n = 0; n < N; ++n)
                ref[(size_t)(i * N + n)] =
                    host_elem(F, q, n, K, abytes.data() + arow * K, abf ? abf + arow * K : nullptr,
                              ascales.empty() ? nullptr : ascales.data() + arow * (K / 128));
        }
        Call c;
        make_call(B, M, off, sorted, abytes, ascales, c);
        static const char* kForm[] = { "rule", "aligned prefill", "decode (tile, run)", "uniform prefill" };
        std::vector<uint16_t> form0;
        for (int form = 0; form < 4; ++form) {
            int rc = 0;
            const std::vector<uint16_t> got = run_launch(pl, c, (size_t)(Tr * N), form, r, &rc);
            const std::string what = std::string(F.name) + " " + label + " T=" + std::to_string(Tr) + " " + kForm[form];
            if (rc != RAD_OK) { std::printf("  skip  %s -- launch rc=%d\n", what.c_str(), rc); continue; }
            int64_t fdiff = 0;
            if (form0.empty()) form0 = got;
            else for (size_t x = 0; x < got.size(); ++x) fdiff += got[x] != form0[x];
            int64_t bad = 0;
            double worst = 0.0;
            std::string first;
            for (int64_t i = 0; i < Tr; ++i)
                for (int64_t n = 0; n < N; ++n) {
                    const double want = ref[(size_t)(i * N + n)];
                    const double g = (double)h_bf16(got[(size_t)(i * N + n)]);
                    const double err = std::fabs(g - want);
                    const double tol = 0.004 * std::fabs(want) + 1e-4;
                    if (err > tol) {
                        if (!bad)
                            first = ", first at row " + std::to_string(i) + " col " + std::to_string(n) +
                                    ": " + std::to_string(g) + " for " + std::to_string(want);
                        worst = std::max(worst, err / tol);
                        ++bad;
                    }
                }
            ok(what + " (forms agree bitwise)", fdiff == 0,
               std::to_string(fdiff) + " of " + std::to_string(Tr * N) + " differ from the first form");
            ok(what + " vs host decode", bad == 0,
               std::to_string(bad) + " of " + std::to_string(Tr * N) + " outside 0.4% + 1e-4" +
                   (bad ? ", worst err/tol " + std::to_string(worst) + first : ""));
        }
        free_call(c);
    }
    free_bank(B);
}

/* ------------------------------------------------------------------ layout hooks */
/* `group` is the GEMM's 128-element scale block, and wgeom refuses K % group. Every format's own
 * grid (32, 64 or 128) divides 128, so a whole 128 of K is the acceptance rule; the per-format
 * checks below it never add a refusal. */
static bool layout_uses_ok(const Fmt& F, int64_t K) {
    (void)F;
    return K % 128 == 0;
}
static void layout_one(const Plugin& pl, const Fmt& F, int64_t N, int64_t K) {
    const RadKernelInfo* r = find(pl, "moe_gemm_w4a8");
    const std::string tag = std::string(F.name) + " K=" + std::to_string(K);
    if (!r || !r->layout || !r->relayout) { ok(tag + " row or hooks present", false); return; }
    const bool want = layout_uses_ok(F, K);
    Planes base = make_planes(F, N, K, 0x51u);
    const std::vector<RadParam> lp = moe_lp(F, N, K);
    const char* wroles2[] = { "codes", "table" };
    const char* wroles1[] = { "codes" };
    const char** wroles = F.rotated || is_iq4(F) ? wroles2 : wroles1;
    const int wn = (F.rotated || is_iq4(F)) ? 2 : 1;
    const char* sroles2[] = { "scale", "scale.1" };
    const char* sroles1[] = { "scale" };
    const char** sroles = F.rotated ? sroles2 : sroles1;
    const int sn = F.rotated ? 2 : 1;

    for (int which = 0; which < 2; ++which) {
        const char** roles = which ? sroles : wroles;
        const int nr = which ? sn : wn;
        int sel[RAD_ENC_MAX_PLANES];
        RadTensor pt[RAD_ENC_MAX_PLANES];
        for (int i = 0; i < nr; ++i) { sel[i] = base.find(roles[i]); pt[i] = base.t(sel[i]); }
        RadLayout L;
        std::memset(&L, 0, sizeof L);
        const int rl = r->layout(lp.data(), (int)lp.size(), which ? 3 : 2, &base.enc, sel, pt, nr, &L);
        const char* plane = which ? "scale" : "weight";
        if (!want) {
            ok(tag + " " + plane + " layout refused", rl == RAD_E_SHAPE, "rc " + std::to_string(rl));
            continue;
        }
        const bool asis = (!which && unrot_asis(F)) || (which && !F.rotated);
        if (asis) {
            ok(tag + " " + plane + " stored as is", rl == RAD_E_UNSUPPORTED, "rc " + std::to_string(rl));
            continue;
        }
        if (rl != RAD_OK) { ok(tag + " " + plane + " layout accepted", false, "rc " + std::to_string(rl)); continue; }
        const size_t gd = 64;
        std::vector<uint8_t> buf((size_t)L.bytes + 2 * gd, 0xA5);
        const int rr = r->relayout(lp.data(), (int)lp.size(), which ? 3 : 2, &base.enc, sel, pt, nr,
                                   buf.data() + gd, L.bytes);
        bool guard = true;
        for (size_t i = 0; i < gd; ++i)
            guard = guard && buf[i] == 0xA5 && buf[gd + (size_t)L.bytes + i] == 0xA5;
        ok(tag + " " + plane + " relayout stays inside the plane", rr == RAD_OK && guard,
           "rc " + std::to_string(rr));
        if (!r->unrelayout || rr != RAD_OK) continue;
        Planes back = make_planes(F, N, K, 0x99u);
        RadTensor pb[RAD_ENC_MAX_PLANES];
        for (int i = 0; i < nr; ++i) pb[i] = back.t(sel[i]);
        const int ru = r->unrelayout(lp.data(), (int)lp.size(), which ? 3 : 2, &base.enc, sel,
                                     buf.data() + gd, L.bytes, pb, nr);
        bool same = ru == RAD_OK;
        for (int i = 0; i < nr && same; ++i) {
            const size_t n = base.data[(size_t)sel[i]].size();
            same = std::memcmp(back.data[(size_t)sel[i]].data(), base.data[(size_t)sel[i]].data(), n) == 0;
        }
        ok(tag + " " + plane + " relayout then unrelayout restores the plane", same,
           "rc " + std::to_string(ru));
    }
    /* Q2_0's canonical u2 plane is the stored plane: its row is K/4 bytes. */
    if (unrot_asis(F) && want) {
        const int ci = base.find("codes");
        ok(tag + " canonical u2 row is K/4 bytes", base.row_bytes(ci) == K / 4,
           std::to_string(base.row_bytes(ci)) + " bytes");
    }
}

static void layout_tests(const Plugin& pl) {
    const int64_t N = 128;
    for (const Fmt* F : kFmts)
        for (int64_t K : { 32, 48, 96, 128, 160, 256, 1024 }) layout_one(pl, *F, N, K);
}

/* ------------------------------------------------------------------ timing */
static double time_calls(const Plugin& pl, std::vector<Call>& cs, const RadKernelInfo* row, int reps) {
    if (pl.pin_moe) pl.pin_moe(0);
    for (int i = 0; i < 4; ++i) row->launch(&cs[(size_t)i % cs.size()].a, nullptr);
    HCK(hipStreamSynchronize(nullptr));
    hipEvent_t e0, e1;
    HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
    HCK(hipEventRecord(e0, nullptr));
    for (int i = 0; i < reps; ++i) row->launch(&cs[(size_t)i % cs.size()].a, nullptr);
    HCK(hipEventRecord(e1, nullptr));
    HCK(hipEventSynchronize(e1));
    float ms = 0;
    HCK(hipEventElapsedTime(&ms, e0, e1));
    HCK(hipEventDestroy(e0)); HCK(hipEventDestroy(e1));
    return (double)ms / reps;
}

/* A plain streaming read of a large buffer: the DRAM ceiling for the GB/s column. */
__global__ void stream_read(const uint4* __restrict__ p, uint64_t n4, uint32_t* sink) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (; i < n4; i += step) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9E3779B9u) sink[0] = acc;
}
static void peak_bandwidth(int reps) {
    const uint64_t bytes = 512ull << 20;           /* 512 MiB */
    const uint64_t n4 = bytes / 16;
    void* p = dalloc((size_t)bytes);
    HCK(hipMemset(p, 0x5A, (size_t)bytes));
    uint32_t* sink = (uint32_t*)dalloc(64);
    const dim3 grid(40 * 32), block(256);
    stream_read<<<grid, block>>>((const uint4*)p, n4, sink);
    HCK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
    HCK(hipEventRecord(e0, nullptr));
    for (int i = 0; i < reps; ++i) stream_read<<<grid, block>>>((const uint4*)p, n4, sink);
    HCK(hipEventRecord(e1, nullptr));
    HCK(hipEventSynchronize(e1));
    float ms = 0;
    HCK(hipEventElapsedTime(&ms, e0, e1));
    const double gbs = (double)bytes * (double)reps / (ms * 1e-3) / 1e9;
    std::printf("streaming read ceiling: %.1f GB/s (%.0f MiB x %d)\n", gbs, (double)bytes / 1048576.0, reps);
    HCK(hipEventDestroy(e0)); HCK(hipEventDestroy(e1));
    hipFree(p); hipFree(sink);
}

/* ------------------------------------------------------------------ bench */
struct Proj { const char* name; int64_t N, K; bool sorted_op; };
static const Proj kTp2[2] = { { "gate_up", 640, 5120, false }, { "down", 5120, 1280, true } };
static const Proj kTp1[2] = { { "gate_up", 1280, 2560, false }, { "down", 2560, 640, true } };

static void bench_bank(const Plugin& pl, const Fmt& F, const Proj& P, int reps, int trials) {
    const int64_t ne = 512, top_k = 10;
    Bank B;
    if (!build_bank(pl, B, F, P.N, P.K, ne, top_k, P.sorted_op, true)) {
        std::printf("  %s %s: bank build failed\n", F.name, P.name);
        return;
    }
    struct Case { const char* name; int64_t M; bool spread; };
    const Case cases[] = { { "M=1", 1, false }, { "M=4", 4, false }, { "M=32", 32, false },
                           { "spread", 76, true }, { "M=128", 128, false }, { "M=512", 512, false } };
    for (const Case& cs : cases) {
        const int64_t M = cs.M, Tr = M * top_k, rows = P.sorted_op ? Tr : M;
        std::vector<uint8_t> abytes;
        std::vector<float> ascales;
        if (F.rotated) draw_act_e4m3(rows, P.K, 0x55u ^ (uint32_t)M, &abytes, &ascales);
        else           draw_act_bf16(rows, P.K, 0x55u ^ (uint32_t)M, &abytes);
        const int R = 4;
        std::vector<Call> calls((size_t)R);
        double act = 0.0;
        for (int r = 0; r < R; ++r) {
            std::vector<int32_t> off, sorted;
            if (cs.spread) route_spread(Tr, top_k, ne, off, sorted);
            else           route(M, top_k, ne, 0x1234567ull * (uint64_t)(r + 1) + (uint64_t)Tr, false, off, sorted);
            make_call(B, M, off, sorted, abytes, ascales, calls[(size_t)r]);
            act += (double)calls[(size_t)r].active;
        }
        act /= (double)R;
        /* A prefill launch is milliseconds: fewer launches give a median as stable as the
         * decode cells' hundred without stretching the run. */
        const int rp = Tr > 1024 ? std::max(5, reps / 10)
                       : Tr > 320 ? std::max(10, reps / 4) : reps;
        std::vector<double> t;
        for (int tr = 0; tr < trials; ++tr) t.push_back(time_calls(pl, calls, B.row, rp) * 1e3);
        std::sort(t.begin(), t.end());
        const double us = t[t.size() / 2];
        const bool dec = 2 * Tr <= 3 * ne;
        std::printf("%-12s %-8s %-6s %6lld %-7s %9.1f %9.1f %10.0f %8.1f\n", F.name, P.name, cs.name,
                    (long long)Tr, dec ? "decode" : "prefill", us,
                    act * B.bytes_expert / (us * 1e-6) / 1e9, B.bytes_expert, act);
        std::fflush(stdout);
        for (Call& c : calls) free_call(c);
    }
    free_bank(B);
}

static void bench(const Plugin& pl, int reps, int trials) {
    peak_bandwidth(8);
    std::printf("\nFlash-Next expert slices: 512 experts, top_k 10; weights one expert replicated\n");
    std::printf("%-12s %-8s %-6s %6s %-7s %9s %9s %10s %8s\n", "format", "proj", "case", "T", "band",
                "us", "GB/s", "bytes/exp", "active");
    for (int si = 0; si < 2; ++si) {
        const Proj& P = kTp2[si];
        std::printf("\n# two-rank production slice: %s N=%lld K=%lld\n", P.name, (long long)P.N, (long long)P.K);
        for (const Fmt* F : kFmts) bench_bank(pl, *F, P, reps, trials);
    }
    for (int si = 0; si < 2; ++si) {
        const Proj& P = kTp1[si];
        std::printf("\n# single-rank whole expert: %s N=%lld K=%lld\n", P.name, (long long)P.N, (long long)P.K);
        for (const Fmt* F : kFmts) bench_bank(pl, *F, P, reps, trials);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: halo_moe_unrot_bench <libhalo.so> layout|check|bench [reps [trials]]\n");
        return 2;
    }
    Plugin pl;
    if (!load_plugin(&pl, argv[1])) return 2;
    const std::string mode = argv[2];
    if (mode == "layout") {
        layout_tests(pl);
        std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "all passed", g_fail);
        return g_fail ? 1 : 0;
    }
    if (mode == "check") {
        const std::vector<int64_t> Ms = { 1, 4, 32 };
        for (const Fmt* F : kFmts) {
            check_case(pl, *F, "gate_up", 128, 256, false, Ms);
            check_case(pl, *F, "down", 256, 128, true, Ms);
        }
        std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "all passed", g_fail);
        return g_fail ? 1 : 0;
    }
    if (mode == "bench") {
        bench(pl, argc > 3 ? std::atoi(argv[3]) : 100, argc > 4 ? std::atoi(argv[4]) : 3);
        return 0;
    }
    return 2;
}
