/* halo_q6k_bench -- the ggml Q6_K dense kernels (libhalo/halo_gemm_q6ka16.hip): layout hooks,
 * numerics against host models and ggml's own dequantisation, the two launch forms against each
 * other, and speed against the FP8 dense kernels. Not part of the plugin and not built by CMake:
 *
 *   clang++ -x hip --offload-arch=gfx1151 -O2 -std=gnu++17 -ffp-contract=off -I abi -I libhalo \
 *       -I build/rad_include tools/q6k_bench/halo_q6k_bench.hip -o halo_q6k_bench -ldl -lamdhip64
 *   halo_q6k_bench <libhalo.so> layout                 host-only layout/relayout/unrelayout tests
 *   halo_q6k_bench <libhalo.so> check                  numerics on random planes
 *   halo_q6k_bench <libhalo.so> real <dir> <stem>...   numerics on GGUF Q6_K blocks dumped by
 *                                                      gguf_q6k_dump.py (every element: the matvec
 *                                                      bit for bit, the WMMA within 4 ulp)
 *   halo_q6k_bench <libhalo.so> bench [trials]         speed, Q6_K vs FP8, the three task shapes
 *   halo_q6k_bench <libhalo.so> tokeq [trials]         per-decoded-token dense time from the shapes
 */
#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "rad_encoding.h"
#include "halo_q6k.h"

#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

static int g_fail = 0, g_ok = 0;
static void ok(const std::string& what, bool c, const std::string& d = "") {
    if (!c) ++g_fail; else ++g_ok;
    std::printf("  %s  %s%s%s\n", c ? "ok  " : "FAIL", what.c_str(), d.empty() ? "" : " -- ", d.c_str());
}

/* ------------------------------------------------------------------ plugin */
struct Plugin {
    void* h = nullptr;
    int (*kernel_count)(void) = nullptr;
    const RadKernelInfo* (*kernel_at)(int) = nullptr;
    int (*mv)(long, long, long, long, long, int, int, int, int, int, int, int, int, long) = nullptr;
    int (*wmma)(long, long, long, long, long, int, int, int, int, int, int, int, long) = nullptr;
    int (*fp8_mt)(long, long, long, long, int, int, int, int, int, int, int, long) = nullptr;
    int (*fp8_m16)(long, long, long, long, long, int, int, int, int, int, int, int, long, long, long) = nullptr;
    long (*fp8_m16_scratch)(int, int, int) = nullptr;
    int (*fp8_tiled)(long, long, long, long, long, int, int, int, int, int, int, int, long) = nullptr;
    int (*quant)(long, long, long, int, int, int, int, int, long) = nullptr;
};
template <class F> static bool sym(void* h, const char* n, F* out) {
    void* p = dlsym(h, n);
    if (!p) { std::printf("missing symbol %s\n", n); return false; }
    std::memcpy(out, &p, sizeof p);
    return true;
}
static bool load_plugin(Plugin* pl, const char* path) {
    pl->h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!pl->h) { std::printf("dlopen(%s): %s\n", path, dlerror()); return false; }
    return sym(pl->h, "rad_kernel_count", &pl->kernel_count) && sym(pl->h, "rad_kernel_at", &pl->kernel_at) &&
           sym(pl->h, "halo_gemm_q6ka16_nt_mv", &pl->mv) && sym(pl->h, "halo_gemm_q6ka16_nt_wmma", &pl->wmma) &&
           sym(pl->h, "halo_gemm_fp8a16_nt_mt", &pl->fp8_mt) && sym(pl->h, "halo_gemm_fp8a8_nt_m16", &pl->fp8_m16) &&
           sym(pl->h, "halo_gemm_fp8a8_nt_m16_scratch", &pl->fp8_m16_scratch) &&
           sym(pl->h, "halo_gemm_fp8a8_tiled", &pl->fp8_tiled) && sym(pl->h, "halo_quant_act_fp8", &pl->quant);
}
static const RadKernelInfo* find(const Plugin& pl, const char* name) {
    for (int k = 0; k < pl.kernel_count(); ++k)
        if (!std::strcmp(pl.kernel_at(k)->name, name)) return pl.kernel_at(k);
    return nullptr;
}

/* ------------------------------------------------------------------ helpers */
static RadParam PI(const char* k, long long v) {
    RadParam p; std::memset(&p, 0, sizeof p); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p;
}
static RadParam PS(const char* k, const char* v) {
    RadParam p; std::memset(&p, 0, sizeof p); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p;
}
static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static uint16_t h_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float h_bf16(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f;
}
static uint16_t h_to_f16(float f) {   /* round to nearest even, normal range only (test data) */
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    int e = (int)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u & 0x7FFFFFu;
    if (e <= 0 || e >= 31) return (uint16_t)s;
    uint32_t r = (uint32_t)(e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t)(s | r);
}
static void* dalloc(size_t bytes) { void* p = nullptr; HCK(hipMalloc(&p, bytes ? bytes : 16)); return p; }
static void* dup_to_dev(const void* src, size_t bytes) {
    void* d = dalloc(bytes);
    HCK(hipMemcpy(d, src, bytes, hipMemcpyHostToDevice));
    return d;
}

/* ------------------------------------------------------------------ Q6_K planes */
static constexpr int64_t kCodeRowBytes(int64_t K) { return K * 6 / 8; }

struct QP {
    int64_t N = 0, K = 0;
    std::vector<uint8_t> codes;   /* [N][K] canonical i6, packed 6 bits per value, q - 32 */
    std::vector<int8_t> sc;       /* [N][K/16] */
    std::vector<uint16_t> d;      /* [N][K/256] f16 bits */
    int64_t KB() const { return K / 16; }
    int64_t KG() const { return K / 256; }
    int code(int64_t n, int64_t k) const {
        const uint8_t* row = codes.data() + n * kCodeRowBytes(K);
        const int64_t bit = 6 * k;
        uint32_t v = row[bit / 8] | ((uint32_t)row[bit / 8 + 1] << 8);
        v = (v >> (bit % 8)) & 63u;
        return (v & 32u) ? (int)v - 64 : (int)v;
    }
    float w(int64_t n, int64_t k) const {
        const float dq = halo_q6k::f16_to_f32(d[(size_t)(n * KG() + k / 256)]);
        const float s = (float)sc[(size_t)(n * KB() + k / 16)];
        return (dq * s) * (float)code(n, k);
    }
};

/* Pack 16 signed codes (q - 32, two's complement) into the 12 bytes of a sub-block row, as
 * split_q6_K's `c.put(q - 32u, 6)` does. */
static void pack6(uint8_t* out, const int cv[16]) {
    std::memset(out, 0, 12);
    for (int i = 0; i < 16; ++i) {
        const uint32_t q = (uint32_t)cv[i] & 63u;
        const int bit = 6 * i, by = bit / 8, sh = bit % 8;
        out[by] |= (uint8_t)(q << sh);
        if (sh > 2) out[by + 1] |= (uint8_t)(q >> (8 - sh));
    }
}

static QP rand_planes(int64_t N, int64_t K, uint32_t seed) {
    QP p; p.N = N; p.K = K;
    p.codes.resize((size_t)(N * kCodeRowBytes(K)));
    p.sc.resize((size_t)(N * (K / 16)));
    p.d.resize((size_t)(N * (K / 256)));
    for (size_t i = 0; i < p.codes.size(); ++i) p.codes[i] = (uint8_t)mix32((uint32_t)i ^ seed);
    for (size_t i = 0; i < p.sc.size(); ++i) p.sc[i] = (int8_t)mix32((uint32_t)i * 3u ^ (seed + 11u));
    for (size_t i = 0; i < p.d.size(); ++i) {
        const uint32_t r = mix32((uint32_t)i * 2654435761u ^ (seed + 77u));
        p.d[i] = (uint16_t)(((r & 1u) << 15) | ((6u + (r >> 1) % 15u) << 10) | ((r >> 8) & 0x3FFu));
    }
    return p;
}

/* Planes from `rows` x K of raw GGUF block_q6_K (ql u8[128] @0, qh u8[64] @128, scales i8[16]
 * @192, d f16 @208): split_q6_K of core/format/ggml_planes.h. Also fills the Python reference's
 * value, (d * sc) * (q - 32), straight from the block fields. */
static QP planes_from_blocks(const std::vector<uint8_t>& blob, int64_t rows, int64_t K,
                             std::vector<float>* ref) {
    QP p; p.N = rows; p.K = K;
    const int64_t KB = K / 16, KG = K / 256;
    p.codes.assign((size_t)(rows * kCodeRowBytes(K)), 0);
    p.sc.resize((size_t)(rows * KB));
    p.d.resize((size_t)(rows * KG));
    if (ref) ref->resize((size_t)(rows * K));
    for (int64_t n = 0; n < rows; ++n)
        for (int64_t kb = 0; kb < KB; ++kb) {
            const int64_t b = n * KG + kb / 16;
            const uint8_t* blk = blob.data() + (size_t)b * 210;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = (const int8_t*)(blk + 192);
            uint16_t dh; std::memcpy(&dh, blk + 208, 2);
            p.sc[(size_t)(n * KB + kb)] = scales[kb % 16];
            if (kb % 16 == 0) p.d[(size_t)(n * KG + kb / 16)] = dh;
            int cv[16];
            for (int i = 0; i < 16; ++i) {
                const int kk = (int)(kb % 16) * 16 + i;
                const int n128 = kk / 128, t = (kk % 128) / 32, l = kk % 32;
                const uint8_t lo = ql[64 * n128 + l + 32 * (t & 1)];
                const uint32_t q = (uint32_t)(t < 2 ? lo & 15 : lo >> 4) |
                                   (uint32_t)((qh[32 * n128 + l] >> (2 * t)) & 3) << 4;
                cv[i] = (int)q - 32;
                if (ref)
                    (*ref)[(size_t)(n * K + kb * 16 + i)] =
                        (halo_q6k::f16_to_f32(dh) * (float)scales[kb % 16]) * (float)cv[i];
            }
            pack6(p.codes.data() + n * kCodeRowBytes(K) + kb * 12, cv);
        }
    return p;
}

static RadEncoding q6k_enc() {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "codes", RAD_I6, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_I8, 1, 16);
    rad_enc_add_plane(&e, "scale.1", RAD_F16, 1, 256);
    return e;
}
static RadTensor planeT(void* data, uint32_t dt, int64_t r, int64_t c) {
    RadTensor t; std::memset(&t, 0, sizeof t);
    t.data = data; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c;
    rad_tensor_pack(&t);
    return t;
}
/* All three operands through the row's own hooks. */
static bool stored_q6k(const RadKernelInfo* r, const QP& p, std::vector<uint8_t>* wq,
                       std::vector<uint8_t>* ws, std::vector<uint8_t>* wd) {
    std::vector<RadParam> lp = { PI("N", p.N), PI("K", p.K), PI("group", 256), PS("dtype", "q6k") };
    const RadEncoding enc = q6k_enc();
    for (int operand : { 2, 3, 6 }) {
        const int sel[1] = { operand == 2 ? 0 : operand == 3 ? 1 : 2 };
        RadTensor pt = operand == 2 ? planeT((void*)p.codes.data(), RAD_I6, p.N, p.K)
                      : operand == 3 ? planeT((void*)p.sc.data(), RAD_I8, p.N, p.KB())
                                     : planeT((void*)p.d.data(), RAD_F16, p.N, p.KG());
        RadLayout L; std::memset(&L, 0, sizeof L);
        int rc = r->layout(lp.data(), (int)lp.size(), operand, &enc, sel, &pt, 1, &L);
        if (rc != RAD_OK) { std::printf("layout rc %d operand %d\n", rc, operand); return false; }
        std::vector<uint8_t>& out = operand == 2 ? *wq : operand == 3 ? *ws : *wd;
        out.assign((size_t)L.bytes, 0);
        rc = r->relayout(lp.data(), (int)lp.size(), operand, &enc, sel, &pt, 1, out.data(), L.bytes);
        if (rc != RAD_OK) { std::printf("relayout rc %d operand %d\n", rc, operand); return false; }
    }
    return true;
}

/* ================================================================== layout tests (host only) */
static void test_layout(const Plugin& pl) {
    std::printf("[layout]\n");
    const RadKernelInfo* rows[2] = { find(pl, "gemm_q6ka16_nt_m8"), find(pl, "gemm_q6ka16_nt_wmma") };
    for (const RadKernelInfo* r : rows)
        if (!r || !r->layout || !r->relayout || !r->unrelayout) { ok("rows and hooks present", false); return; }
    const RadKernelInfo* r = rows[0];
    ok("both rows share the layout hooks",
       rows[0]->layout == rows[1]->layout && rows[0]->relayout == rows[1]->relayout &&
           rows[0]->unrelayout == rows[1]->unrelayout);
    const RadEncoding enc = q6k_enc();
    const int64_t G = 64;

    const int64_t shapes[][2] = { {16, 256}, {16, 512}, {48, 768}, {64, 2048}, {128, 1024}, {32, 8192} };
    for (const auto& sh : shapes) {
        const int64_t N = sh[0], K = sh[1];
        QP p = rand_planes(N, K, 1234u + (uint32_t)N);
        bool good = true; std::string why;
        std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 256) };
        for (int operand : { 2, 3, 6 }) {
            const int sel[1] = { operand == 2 ? 0 : operand == 3 ? 1 : 2 };
            RadTensor pt = operand == 2 ? planeT(p.codes.data(), RAD_I6, N, K)
                          : operand == 3 ? planeT(p.sc.data(), RAD_I8, N, p.KB())
                                         : planeT(p.d.data(), RAD_F16, N, p.KG());
            RadLayout L; std::memset(&L, 0, sizeof L);
            if (r->layout(lp.data(), 3, operand, &enc, sel, &pt, 1, &L) != RAD_OK) {
                good = false; why = "layout refused"; break;
            }
            const int64_t want = operand == 2 ? N * kCodeRowBytes(K)
                               : operand == 3 ? N * (K / 16) : N * (K / 256) * 2;
            if (L.bytes != want) {
                good = false; why = "bytes " + std::to_string(L.bytes) + " want " + std::to_string(want); break;
            }
            std::vector<uint8_t> a((size_t)(L.bytes + 2 * G), 0x00), b((size_t)(L.bytes + 2 * G), 0xFF);
            const int ra = r->relayout(lp.data(), 3, operand, &enc, sel, &pt, 1, a.data() + G, L.bytes);
            const int rb = r->relayout(lp.data(), 3, operand, &enc, sel, &pt, 1, b.data() + G, L.bytes);
            if (ra != RAD_OK || rb != RAD_OK) { good = false; why = "relayout rc"; break; }
            for (int64_t i = 0; i < G; ++i)
                if (a[(size_t)i] != 0x00 || a[(size_t)(G + L.bytes + i)] != 0x00 ||
                    b[(size_t)i] != 0xFF || b[(size_t)(G + L.bytes + i)] != 0xFF) {
                    good = false; why = "guard bytes touched"; break;
                }
            if (!good) break;
            if (std::memcmp(a.data() + G, b.data() + G, (size_t)L.bytes)) {
                good = false; why = "a destination byte was not written"; break;
            }
            const uint8_t* st = a.data() + G;
            for (int64_t n = 0; n < N && good; ++n)
                for (int64_t kb = 0; kb < K / 16 && good; ++kb) {
                    const int64_t at = halo_q6k::code_off(n, kb * 16, K);
                    const int64_t from = n * kCodeRowBytes(K) + kb * 12;
                    if (operand == 2 && std::memcmp(st + at, p.codes.data() + from, 12)) {
                        good = false; why = "code placement";
                    } else if (operand == 3 &&
                               (int8_t)st[halo_q6k::scale_idx(n, kb, K)] != p.sc[(size_t)(n * (K / 16) + kb)]) {
                        good = false; why = "scale placement";
                    } else if (operand == 6 && kb % 16 == 0) {
                        uint16_t h;
                        std::memcpy(&h, st + 2 * halo_q6k::d_idx(n, kb / 16, K), 2);
                        if (h != p.d[(size_t)(n * (K / 256) + kb / 16)]) { good = false; why = "d placement"; }
                    }
                }
            if (!good) break;
            const int64_t pb = want;
            std::vector<uint8_t> back((size_t)(pb + 2 * G), 0x5C);
            RadTensor bt = operand == 2 ? planeT(back.data() + G, RAD_I6, N, K)
                          : operand == 3 ? planeT(back.data() + G, RAD_I8, N, K / 16)
                                         : planeT(back.data() + G, RAD_F16, N, K / 256);
            if (r->unrelayout(lp.data(), 3, operand, &enc, sel, st, L.bytes, &bt, 1) != RAD_OK) {
                good = false; why = "unrelayout rc"; break;
            }
            const uint8_t* orig = operand == 2 ? p.codes.data()
                                : operand == 3 ? (const uint8_t*)p.sc.data()
                                               : (const uint8_t*)p.d.data();
            if (std::memcmp(back.data() + G, orig, (size_t)pb)) { good = false; why = "roundtrip differs"; break; }
            for (int64_t i = 0; i < G; ++i)
                if (back[(size_t)i] != 0x5C || back[(size_t)(G + pb + i)] != 0x5C) {
                    good = false; why = "unrelayout guard bytes"; break;
                }
        }
        ok("N=" + std::to_string(N) + " K=" + std::to_string(K) +
               ": formula, whole destination written, guards intact, roundtrip", good, why);
    }

    struct Bad { int64_t N, K; int group; const char* why; };
    const Bad bads[] = { {16, 128, 256, "K half a block"}, {16, 384, 256, "K=384 (1.5 blocks)"},
                         {16, 255, 256, "K=255"}, {16, 0, 256, "K=0"}, {16, -256, 256, "K<0"},
                         {8, 256, 256, "N=8"}, {17, 256, 256, "N=17"}, {24, 256, 256, "N=24 (1.5 tiles)"},
                         {0, 256, 256, "N=0"}, {16, 256, 32, "group 32"}, {16, 256, 512, "group 512"} };
    for (const Bad& b : bads) {
        bool good = true; std::string why;
        const int64_t Np = 32, Kp = 512;
        QP p = rand_planes(Np, Kp, 5);
        std::vector<RadParam> lp = { PI("N", b.N), PI("K", b.K), PI("group", b.group) };
        for (int operand : { 2, 3, 6 }) {
            const int sel[1] = { operand == 2 ? 0 : operand == 3 ? 1 : 2 };
            RadTensor pt = operand == 2 ? planeT(p.codes.data(), RAD_I6, Np, Kp)
                          : operand == 3 ? planeT(p.sc.data(), RAD_I8, Np, Kp / 16)
                                         : planeT(p.d.data(), RAD_F16, Np, Kp / 256);
            RadLayout L; std::memset(&L, 0, sizeof L);
            const int rl = r->layout(lp.data(), 3, operand, &enc, sel, &pt, 1, &L);
            std::vector<uint8_t> dst((size_t)(1024 + 2 * G), 0xA5);
            const int rr = r->relayout(lp.data(), 3, operand, &enc, sel, &pt, 1, dst.data() + G, 1024);
            std::vector<uint8_t> plane((size_t)(1024 + 2 * G), 0x5C);
            RadTensor ot = planeT(plane.data() + G, operand == 2 ? RAD_I6 : operand == 3 ? RAD_I8 : RAD_F16, 1, 1);
            const int ru = r->unrelayout(lp.data(), 3, operand, &enc, sel, dst.data() + G, 1024, &ot, 1);
            bool clean = true;
            for (uint8_t v : dst) clean = clean && v == 0xA5;
            for (uint8_t v : plane) clean = clean && v == 0x5C;
            if (rl != RAD_E_SHAPE || rr != RAD_E_SHAPE || ru != RAD_E_SHAPE || !clean) {
                good = false;
                why = "operand " + std::to_string(operand) + " rc " + std::to_string(rl) + "/" +
                      std::to_string(rr) + "/" + std::to_string(ru) + (clean ? "" : " wrote memory");
            }
        }
        ok(std::string("refuses ") + b.why + " (layout, relayout, unrelayout; nothing written)", good, why);
    }

    {
        const int64_t N = 32, K = 512;
        QP p = rand_planes(N, K, 9);
        std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 256) };
        int sel[1] = { 0 };
        RadTensor pt = planeT(p.codes.data(), RAD_I6, N, K);
        RadLayout L; std::memset(&L, 0, sizeof L);
        r->layout(lp.data(), 3, 2, &enc, sel, &pt, 1, &L);
        std::vector<uint8_t> dst((size_t)L.bytes + 2 * G, 0xA5);
        const int r1 = r->relayout(lp.data(), 3, 2, &enc, sel, &pt, 1, dst.data() + G, L.bytes - 1);
        const int r2 = r->relayout(lp.data(), 3, 2, &enc, sel, &pt, 1, dst.data() + G, L.bytes + 1);
        bool clean = true; for (uint8_t v : dst) clean = clean && v == 0xA5;
        ok("destination one byte short or long is refused, nothing written",
           r1 == RAD_E_SHAPE && r2 == RAD_E_SHAPE && clean, "rc " + std::to_string(r1) + "/" + std::to_string(r2));
        RadTensor wrongK = planeT(p.codes.data(), RAD_I6, N, K - 16);
        RadTensor wrongN = planeT(p.codes.data(), RAD_I6, N - 16, K);
        const int r3 = r->layout(lp.data(), 3, 2, &enc, sel, &wrongK, 1, &L);
        const int r4 = r->layout(lp.data(), 3, 2, &enc, sel, &wrongN, 1, &L);
        ok("plane extents that disagree with N, K are refused", r3 == RAD_E_SHAPE && r4 == RAD_E_SHAPE,
           "rc " + std::to_string(r3) + "/" + std::to_string(r4));
        RadEncoding bad = enc; bad.plane[1].block[1] = 32;
        RadTensor st32 = planeT(p.sc.data(), RAD_I8, N, K / 32);
        const int r5 = r->layout(lp.data(), 3, 3, &bad, sel, &st32, 1, &L);
        RadEncoding bad2 = enc; bad2.n_planes = 2;
        const int r6 = r->layout(lp.data(), 3, 2, &bad2, sel, &pt, 1, &L);
        RadEncoding bad3 = enc; bad3.plane[0].dtype = RAD_U6;
        const int r7 = r->layout(lp.data(), 3, 2, &bad3, sel, &pt, 1, &L);
        ok("a wrong sub-block, a missing scale.1 or a u6 code plane is refused as a dtype or shape",
           (r5 == RAD_E_SHAPE || r5 == RAD_E_DTYPE) && r6 == RAD_E_DTYPE && r7 == RAD_E_DTYPE,
           "rc " + std::to_string(r5) + "/" + std::to_string(r6) + "/" + std::to_string(r7));
        const int r8 = r->layout(lp.data(), 3, 0, &enc, sel, &pt, 1, &L);
        ok("an operand that is neither codes, scale nor scale.1 is unsupported", r8 == RAD_E_UNSUPPORTED,
           "rc " + std::to_string(r8));
    }
}

/* ================================================================== device runs */
static float* d_to_host_f32(void* dptr, size_t n, std::vector<float>* out) {
    out->resize(n);
    HCK(hipMemcpy(out->data(), dptr, n * 4, hipMemcpyDeviceToHost));
    return out->data();
}

/* the matvec's ordered host model (halo_gemm_q6ka16.hip header) */
static void model_mv(const QP& p, const std::vector<float>& x, int M, int wk, int64_t K_ld_x,
                     std::vector<float>* y) {
    const int64_t N = p.N, K = p.K, P = K / 32;
    y->assign((size_t)(M * N), 0.0f);
    for (int m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            float tot = 0;
            for (int w = 0; w < wk; ++w) {
                float a[2] = { 0, 0 };
                for (int h = 0; h < 2; ++h)
                    for (int64_t pass = w; pass < P; pass += wk) {
                        const int64_t kb = 2 * pass + h;
                        float part = 0.0f;
                        for (int i = 0; i < 16; ++i)
                            part = std::fmaf((float)p.code(n, kb * 16 + i), x[(size_t)(m * K_ld_x + kb * 16 + i)], part);
                        const float dq = halo_q6k::f16_to_f32(p.d[(size_t)(n * p.KG() + (kb / 16))]) *
                                         (float)p.sc[(size_t)(n * p.KB() + kb)];
                        a[h] = std::fmaf(dq, part, a[h]);
                    }
                const float wt = a[0] + a[1];
                tot = w == 0 ? wt : tot + wt;
            }
            (*y)[(size_t)(m * N + n)] = tot;
        }
}
/* the double-precision value, and sum |w x| (the scale an f32 error is judged against) */
static void ref_double(const QP& p, const std::vector<float>& x, int M, std::vector<double>* y,
                       std::vector<double>* mag) {
    y->assign((size_t)(M * p.N), 0); mag->assign((size_t)(M * p.N), 0);
    for (int m = 0; m < M; ++m)
        for (int64_t n = 0; n < p.N; ++n) {
            double s = 0, a = 0;
            for (int64_t k = 0; k < p.K; ++k) {
                const double w = (double)p.w(n, k);
                s += w * (double)x[(size_t)(m * p.K + k)];
                a += std::fabs(w * (double)x[(size_t)(m * p.K + k)]);
            }
            (*y)[(size_t)(m * p.N + n)] = s; (*mag)[(size_t)(m * p.N + n)] = a;
        }
}

struct Run {
    Plugin* pl; const RadKernelInfo* row;
    QP p; std::vector<uint8_t> wq, ws, wd; void *dwq = nullptr, *dws = nullptr, *dwd = nullptr;
    void upload(Plugin* plug, const QP& planes) {
        pl = plug; row = find(*plug, "gemm_q6ka16_nt_m8"); p = planes;
        if (dwq) { hipFree(dwq); hipFree(dws); hipFree(dwd); }
        if (!stored_q6k(row, p, &wq, &ws, &wd)) std::exit(2);
        dwq = dup_to_dev(wq.data(), wq.size());
        dws = dup_to_dev(ws.data(), ws.size());
        dwd = dup_to_dev(wd.data(), wd.size());
    }
    int mv(const std::vector<uint16_t>& xb, int M, int wk, int act, std::vector<float>* y) {
        void* dx = dup_to_dev(xb.data(), xb.size() * 2);
        void* dy = dalloc((size_t)M * p.N * 4);
        const int rc = pl->mv((long)dx, (long)dwq, (long)dws, (long)dwd, (long)dy, M, (int)p.K, (int)p.N,
                              0, 0, wk, act, 1, 0);
        HCK(hipDeviceSynchronize());
        if (rc == 0) d_to_host_f32(dy, (size_t)M * p.N, y);
        hipFree(dx); hipFree(dy);
        return rc;
    }
    int wmma(const std::vector<uint16_t>& xb, int M, int tt, std::vector<float>* y) {
        void* dx = dup_to_dev(xb.data(), xb.size() * 2);
        void* dy = dalloc((size_t)M * p.N * 4);
        const int rc = pl->wmma((long)dx, (long)dwq, (long)dws, (long)dwd, (long)dy, M, (int)p.K, (int)p.N,
                                0, 0, tt, 1, 0);
        HCK(hipDeviceSynchronize());
        if (rc == 0) d_to_host_f32(dy, (size_t)M * p.N, y);
        hipFree(dx); hipFree(dy);
        return rc;
    }
    int mv_bf16(const std::vector<uint16_t>& xb, int M, int wk, std::vector<uint16_t>* y) {
        void* dx = dup_to_dev(xb.data(), xb.size() * 2);
        void* dy = dalloc((size_t)M * p.N * 2);
        const int rc = pl->mv((long)dx, (long)dwq, (long)dws, (long)dwd, (long)dy, M, (int)p.K, (int)p.N,
                              0, 0, wk, 0, 0, 0);
        HCK(hipDeviceSynchronize());
        y->resize((size_t)M * p.N);
        if (rc == 0) HCK(hipMemcpy(y->data(), dy, (size_t)M * p.N * 2, hipMemcpyDeviceToHost));
        hipFree(dx); hipFree(dy);
        return rc;
    }
    int wmma_bf16(const std::vector<uint16_t>& xb, int M, int tt, std::vector<uint16_t>* y) {
        void* dx = dup_to_dev(xb.data(), xb.size() * 2);
        void* dy = dalloc((size_t)M * p.N * 2);
        const int rc = pl->wmma((long)dx, (long)dwq, (long)dws, (long)dwd, (long)dy, M, (int)p.K, (int)p.N,
                                0, 0, tt, 0, 0);
        HCK(hipDeviceSynchronize());
        y->resize((size_t)M * p.N);
        if (rc == 0) HCK(hipMemcpy(y->data(), dy, (size_t)M * p.N * 2, hipMemcpyDeviceToHost));
        hipFree(dx); hipFree(dy);
        return rc;
    }
};

static std::vector<uint16_t> rand_acts_bf16(int M, int64_t K, uint32_t seed, std::vector<float>* xf,
                                            bool ints) {
    std::vector<uint16_t> xb((size_t)(M * K));
    xf->resize(xb.size());
    for (size_t i = 0; i < xb.size(); ++i) {
        const uint32_t r = mix32((uint32_t)i ^ seed);
        float v = ints ? (float)((int)(r % 17u) - 8)
                       : ((float)(r & 0xFFFFu) / 32768.0f - 1.0f) * (1.0f + (float)((r >> 16) % 7u));
        xb[i] = h_to_bf16(v);
        (*xf)[i] = h_bf16(xb[i]);
    }
    return xb;
}

static bool bits_eq(const std::vector<float>& a, const std::vector<float>& b, std::string* why) {
    if (a.size() != b.size()) { *why = "size"; return false; }
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y; std::memcpy(&x, &a[i], 4); std::memcpy(&y, &b[i], 4);
        if (x != y && !(a[i] == 0.0f && b[i] == 0.0f)) {
            *why = "first diff at " + std::to_string(i) + ": " + std::to_string(a[i]) + " vs " + std::to_string(b[i]);
            return false;
        }
    }
    return true;
}

static int64_t ulp_dist(float a, float b) {
    if (a == b) return 0;
    int32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4);
    if (x < 0) x = (int32_t)0x80000000 - x;
    if (y < 0) y = (int32_t)0x80000000 - y;
    return std::llabs((int64_t)x - (int64_t)y);
}
static int64_t max_ulp(const std::vector<float>& a, const std::vector<float>& b,
                       int64_t* n_inexact = nullptr) {
    int64_t w = 0, c = 0;
    if (a.size() != b.size()) return 1 << 30;
    for (size_t i = 0; i < a.size(); ++i) { const int64_t d = ulp_dist(a[i], b[i]); w = std::max(w, d); c += d != 0; }
    if (n_inexact) *n_inexact = c;
    return w;
}
static double wmma_units(const std::vector<float>& got, const std::vector<double>& exact,
                         const std::vector<double>& mag) {
    double w = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (mag[i] > 0) w = std::max(w, std::fabs((double)got[i] - exact[i]) / (mag[i] * 5.96e-8));
    return w;
}

static void test_check(Plugin& pl) {
    std::printf("[check] random planes\n");
    const struct { int64_t N, K; } shp[] = { {64, 256}, {512, 2048}, {2048, 512}, {160, 768}, {1024, 1024} };
    for (const auto& s : shp) {
        QP p = rand_planes(s.N, s.K, 4242u + (uint32_t)s.N);
        Run run; run.upload(&pl, p);
        std::string tag = "N=" + std::to_string(s.N) + " K=" + std::to_string(s.K);

        for (int M : { 1, 2, 3, 5, 8 }) {
            std::vector<float> xf; auto xb = rand_acts_bf16(M, s.K, 17u + (uint32_t)M, &xf, false);
            for (int wk : { 1, 2, 4 }) {
                std::vector<float> got, want; std::string why;
                const int rc = run.mv(xb, M, wk, 0, &got);
                model_mv(p, xf, M, wk, s.K, &want);
                ok(tag + " matvec M=" + std::to_string(M) + " WK=" + std::to_string(wk) +
                       " bf16 act == ordered host model, bitwise",
                   rc == 0 && bits_eq(got, want, &why), rc ? "rc " + std::to_string(rc) : why);
            }
        }
        {
            const int M = 3;
            std::vector<uint16_t> xh((size_t)(M * s.K)); std::vector<float> xf(xh.size());
            for (size_t i = 0; i < xh.size(); ++i) {
                const float v = ((float)(mix32((uint32_t)i ^ 99u) & 0xFFFFu) / 32768.0f - 1.0f) * 3.0f;
                xh[i] = h_to_f16(v); xf[i] = halo_q6k::f16_to_f32(xh[i]);
            }
            std::vector<float> got, want; std::string why;
            const int rc = run.mv(xh, M, 0, 1, &got);
            const int nt = (int)(s.N / 16); const int wk = nt >= 512 ? 1 : (nt >= 256 ? 2 : 4);
            model_mv(p, xf, M, wk, s.K, &want);
            ok(tag + " matvec M=3 f16 act == ordered host model, bitwise",
               rc == 0 && bits_eq(got, want, &why), rc ? "rc " + std::to_string(rc) : why);
        }
        /* Integer activations, d powers of two and sc = +-1: every partial sum is an integer and
         * exact, so ANY summation order is bitwise equal. The two forms must then agree byte for
         * byte, and both must equal the exact value. */
        {
            const int M = 8;
            std::vector<float> xf; auto xb = rand_acts_bf16(M, s.K, 31u, &xf, true);
            QP pi = p;
            for (size_t i = 0; i < pi.d.size(); ++i)
                pi.d[i] = (uint16_t)(((mix32((uint32_t)i) & 1u) << 15) | ((11u + mix32((uint32_t)i + 3u) % 5u) << 10));
            for (size_t i = 0; i < pi.sc.size(); ++i) pi.sc[i] = (int8_t)((mix32((uint32_t)i + 7u) & 1u) ? 1 : -1);
            Run ri; ri.upload(&pl, pi);
            std::vector<double> yd0, mg0; ref_double(pi, xf, M, &yd0, &mg0);
            std::vector<float> exact(yd0.size());
            for (size_t i = 0; i < exact.size(); ++i) exact[i] = (float)yd0[i];
            std::vector<float> got; std::string why;
            const int rc = ri.mv(xb, M, 0, 0, &got);
            ok(tag + " exact case: matvec == exact value, bitwise", rc == 0 && bits_eq(got, exact, &why), why);
            /* The matrix unit's own 16-term sum is not bit-faithful on gfx1151 (as the Q8_0 kernel's
             * report found), so its f32 accumulator is held to 4 units of 2^-24 x sum|wx|. The two
             * forms share the stored weight bytes (one layout, one set of plane pointers); their
             * values must agree to that tolerance, not bit for bit. */
            for (int tt : { 1, 2, 4, 8 }) {
                std::vector<float> g2;
                const int r2 = ri.wmma(xb, M, tt, &g2);
                const double u = r2 == 0 ? wmma_units(g2, yd0, mg0) : 1e9;
                char b[200];
                std::snprintf(b, sizeof b, "worst %.2f units of 2^-24 x sum|wx| (the unit is not bit-exact, see the report); %lld ulp from the matvec f32",
                              u, (long long)max_ulp(g2, got));
                ok(tag + " exact case: WMMA TT=" + std::to_string(tt) + " within 4 units of the exact sum",
                   r2 == 0 && u <= 4.0, r2 ? "rc " + std::to_string(r2) : b);
            }
            if (s.N * s.K <= 1 << 22) {
                for (int Mx : { 1, 7, 17, 33, 130 }) {
                    std::vector<float> xf2; auto xb2 = rand_acts_bf16(Mx, s.K, 77u + (uint32_t)Mx, &xf2, true);
                    std::vector<double> yd, mg; ref_double(pi, xf2, Mx, &yd, &mg);
                    std::vector<float> g3;
                    const int r3 = ri.wmma(xb2, Mx, 0, &g3);
                    const double u = r3 == 0 ? wmma_units(g3, yd, mg) : 1e9;
                    char b[160]; std::snprintf(b, sizeof b, "worst %.2f units of 2^-24 x sum|wx|", u);
                    ok(tag + " exact case: WMMA M=" + std::to_string(Mx) + " within 4 units of the exact value",
                       r3 == 0 && u <= 4.0, r3 ? "rc " + std::to_string(r3) : b);
                }
            }
        }
        /* random floats: error against the double value, in units of the f32 ulp of the magnitude
         * sum(|w x|) -- the bound any f32 summation order meets up to a small factor */
        {
            const int M = 8;
            std::vector<float> xf; auto xb = rand_acts_bf16(M, s.K, 5u, &xf, false);
            std::vector<double> yd, mg; ref_double(p, xf, M, &yd, &mg);
            auto worst = [&](const std::vector<float>& g) {
                double w = 0;
                for (size_t i = 0; i < g.size(); ++i) w = std::max(w, std::fabs((double)g[i] - yd[i]) / (mg[i] * 5.96e-8));
                return w;
            };
            std::vector<float> g1, g2;
            run.mv(xb, M, 0, 0, &g1); run.wmma(xb, M, 0, &g2);
            const double e1 = worst(g1), e2 = worst(g2);
            char b[160]; std::snprintf(b, sizeof b, "matvec %.2f, WMMA %.2f (units: 2^-24 x sum|wx|)", e1, e2);
            ok(tag + " random acts: worst error vs double within the f32 summation bound (<= 8)",
               e1 <= 8.0 && e2 <= 8.0, b);
            std::vector<uint16_t> yb, yw;
            run.mv_bf16(xb, M, 0, &yb); run.wmma_bf16(xb, M, 0, &yw);
            bool b1 = true, b2 = true;
            for (size_t i = 0; i < yb.size(); ++i) {
                b1 = b1 && yb[i] == h_to_bf16(g1[i]);
                b2 = b2 && yw[i] == h_to_bf16(g2[i]);
            }
            ok(tag + " bf16 output == RNE(f32 accumulator), matvec and WMMA", b1 && b2,
               std::string(b1 ? "" : "matvec differs ") + (b2 ? "" : "wmma differs"));
        }
    }
}

/* ================================================================== the row shims */
/* The rows' launch shims, not just the raw entries: operands 2 (b), 3 (b_scale) and 6 (b_ref =
 * scale.1), the dtype/group checks, and the bf16 store. A random plane, M=5, both rows; each is
 * compared to the raw entry's f32 accumulator rounded to bf16. */
static void test_rows(Plugin& pl) {
    std::printf("[rows] launch shims\n");
    const RadKernelInfo* rmv = find(pl, "gemm_q6ka16_nt_m8");
    const RadKernelInfo* rwm = find(pl, "gemm_q6ka16_nt_wmma");
    if (!rmv || !rwm) { ok("both rows present", false); return; }
    const int64_t N = 512, K = 2048;
    const int M = 5;
    QP p = rand_planes(N, K, 606u);
    Run run; run.upload(&pl, p);
    std::vector<float> xf;
    auto xb = rand_acts_bf16(M, K, 61u, &xf, false);
    std::vector<float> want;
    run.mv(xb, M, 0, 0, &want);
    std::vector<float> want_w;
    run.wmma(xb, M, 0, &want_w);

    void* dx = dup_to_dev(xb.data(), xb.size() * 2);
    void* dy = dalloc((size_t)M * N * 2);
    RadTensor X = planeT(dx, RAD_BF16, M, K);
    RadTensor W = planeT(run.dwq, RAD_I6, N, K);
    RadTensor S = planeT(run.dws, RAD_I8, N, K / 16);
    RadTensor D = planeT(run.dwd, RAD_F16, N, K / 256);
    RadTensor Y = planeT(dy, RAD_BF16, M, N);
    const RadTensor operands[7] = { X, RadTensor{}, W, S, Y, RadTensor{}, D };
    const RadParam params[] = { PI("M", M), PI("N", N), PI("K", K), PI("group", 256), PS("dtype", "q6k") };
    RadArgs a; std::memset(&a, 0, sizeof a);
    a.t = operands; a.n_t = 7; a.p = params; a.n_p = 5;
    for (int which = 0; which < 2; ++which) {
        const RadKernelInfo* row = which == 0 ? rmv : rwm;
        std::vector<uint16_t> junk((size_t)(M * N), 0x7f7fu);
        HCK(hipMemcpy(dy, junk.data(), junk.size() * 2, hipMemcpyHostToDevice));
        const int rc = row->launch(&a, nullptr);
        HCK(hipDeviceSynchronize());
        std::vector<uint16_t> got((size_t)(M * N));
        HCK(hipMemcpy(got.data(), dy, got.size() * 2, hipMemcpyDeviceToHost));
        const std::vector<float>& w = which == 0 ? want : want_w;
        int64_t diff = 0; std::string first;
        for (int64_t i = 0; i < M * N; ++i) {
            const uint16_t r = h_to_bf16(w[(size_t)i]);
            if (got[(size_t)i] != r) { if (!diff) first = "at " + std::to_string(i); ++diff; }
        }
        ok("the " + std::string(which == 0 ? "matvec" : "WMMA") +
               " row's shim reads operands 2/3/6 and stores bf16 RNE of the raw entry",
           rc == RAD_OK && diff == 0,
           rc != RAD_OK ? "launch rc " + std::to_string(rc) : std::to_string(diff) + " differ " + first);
    }
    hipFree(dx); hipFree(dy);
}

/* ================================================================== real tensors */
static bool read_file(const std::string& pth, std::vector<uint8_t>* out) {
    std::ifstream f(pth, std::ios::binary);
    if (!f) return false;
    out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}
static void test_real(Plugin& pl, const std::string& dir, const std::vector<std::string>& stems) {
    for (const std::string& stem : stems) {
        std::vector<uint8_t> blob;
        std::ifstream sf(dir + "/" + stem + ".shape");
        long long rows = 0, K = 0, Nfull = 0;
        if (!(sf >> rows >> K >> Nfull) || !read_file(dir + "/" + stem + ".q6k", &blob) ||
            blob.size() != (size_t)(rows * (K / 256) * 210)) {
            ok("read " + stem, false, "missing or wrong size");
            continue;
        }
        std::vector<uint8_t> refraw;
        std::vector<float> pyref;
        if (!read_file(dir + "/" + stem + ".ref", &refraw) || refraw.size() != (size_t)(rows * K * 4)) {
            ok("read " + stem + ".ref", false, "missing or wrong size");
            continue;
        }
        pyref.resize((size_t)(rows * K));
        std::memcpy(pyref.data(), refraw.data(), refraw.size());
        std::printf("[real] %s: %lld of %lld rows, K=%lld (%zu bytes of GGUF blocks)\n", stem.c_str(),
                    rows, Nfull, K, blob.size());
        std::vector<float> ours;
        QP p = planes_from_blocks(blob, rows, K, &ours);
        /* the C++ split of the block bytes must equal the Python reference for every element */
        {
            std::string why;
            ok(stem + ": the block split equals tools/ud_repack decode_q6_K, bitwise",
               bits_eq(ours, pyref, &why), why);
        }
        Run run; run.upload(&pl, p);
        /* (1) every element through both kernels: one-hot bf16 activations read the weight back as
         *     fma(d*sc, code, 0), which is the reference's (d*sc)*(q-32). */
        for (int which = 0; which < 2; ++which) {
            int64_t bad = 0, cmp = 0, inexact = 0, maxud = 0; std::string first;
            for (int64_t j0 = 0; j0 < K; j0 += 8) {
                const int M = 8;
                std::vector<uint16_t> xb((size_t)(M * K), 0);
                for (int m = 0; m < M; ++m) if (j0 + m < K) xb[(size_t)(m * K + j0 + m)] = 0x3F80;
                std::vector<float> y;
                const int rc = which == 0 ? run.mv(xb, M, 0, 0, &y) : run.wmma(xb, M, 0, &y);
                if (rc) { bad = -1; first = "rc " + std::to_string(rc); break; }
                for (int m = 0; m < M && j0 + m < K; ++m)
                    for (int64_t n = 0; n < rows; ++n) {
                        const float g = y[(size_t)(m * rows + n)], w = pyref[(size_t)(n * K + j0 + m)];
                        ++cmp;
                        const int64_t ud = ulp_dist(g, w);
                        if (ud) ++inexact; maxud = std::max(maxud, ud);
                        if (ud > (which == 0 ? 0 : 4)) {
                            if (!bad) { char dbg[200]; std::snprintf(dbg, sizeof dbg, "n=%lld k=%lld got %.9g (0x%08x) want %.9g (0x%08x) p.w %.9g", (long long)n, (long long)(j0 + m), g, *(uint32_t*)&g, w, *(uint32_t*)&w, p.w(n, j0 + m)); first = dbg; }
                            ++bad;
                        }
                    }
            }
            ok(stem + ": " + (which == 0 ? "matvec one-hot readback of all " : "WMMA one-hot readback (within 4 ulp) of all ") +
                   std::to_string(rows * K) +
                   (which == 0 ? " weights == the Python reference, bitwise" : " weights vs the Python reference"),
               bad == 0, bad ? std::to_string(bad) + " beyond tolerance, " + first
                             : std::to_string(cmp) + " compared, " + std::to_string(inexact) +
                                   " not bit-equal, max " + std::to_string(maxud) + " ulp");
        }
        /* (2) a real-weight matvec/GEMM: normal activations, error vs double, matvec vs model */
        {
            const int M = 8;
            std::vector<float> xf(M * K);
            for (size_t i = 0; i < xf.size(); ++i) {
                const uint32_t r = mix32((uint32_t)i ^ 0xABCDu);
                xf[i] = ((float)(r & 0xFFFFu) / 32768.0f - 1.0f) * (0.5f + (float)((r >> 16) % 5u));
            }
            std::vector<uint16_t> xb(xf.size());
            for (size_t i = 0; i < xf.size(); ++i) { xb[i] = h_to_bf16(xf[i]); xf[i] = h_bf16(xb[i]); }
            std::vector<double> yd, mg; ref_double(p, xf, M, &yd, &mg);
            std::vector<float> g1, g2, want;
            run.mv(xb, M, 0, 0, &g1); run.wmma(xb, M, 0, &g2);
            double e1 = 0, e2 = 0;
            for (size_t i = 0; i < g1.size(); ++i) {
                e1 = std::max(e1, std::fabs((double)g1[i] - yd[i]) / (mg[i] * 5.96e-8));
                e2 = std::max(e2, std::fabs((double)g2[i] - yd[i]) / (mg[i] * 5.96e-8));
            }
            const int nt = (int)(rows / 16); const int wk = nt >= 512 ? 1 : (nt >= 256 ? 2 : 4);
            model_mv(p, xf, M, wk, K, &want);
            std::string why;
            char b[200]; std::snprintf(b, sizeof b, "matvec %.2f, WMMA %.2f (2^-24 x sum|wx|)", e1, e2);
            ok(stem + ": real weights x random bf16, error vs double (<= 8 units)", e1 <= 8.0 && e2 <= 8.0, b);
            ok(stem + ": real weights matvec == ordered host model, bitwise", bits_eq(g1, want, &why), why);
        }
    }
}

/* ================================================================== speed */
struct Shape { const char* name; int64_t N, K; };
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <class F>
static double time_us(F&& launch, int copies, int trials) {
    hipEvent_t e0, e1; HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
    const int reps = std::max(copies * 2, 24);
    for (int i = 0; i < copies; ++i) launch(i);
    HCK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int tr = 0; tr < trials; ++tr) {
        HCK(hipEventRecord(e0, 0));
        for (int i = 0; i < reps; ++i) launch(i % copies);
        HCK(hipEventRecord(e1, 0));
        HCK(hipEventSynchronize(e1));
        float ms = 0; HCK(hipEventElapsedTime(&ms, e0, e1));
        t.push_back(1000.0 * ms / reps);
    }
    hipEventDestroy(e0); hipEventDestroy(e1);
    return median(t);
}

struct Bench {
    Plugin& pl;
    int trials;
    std::vector<void*> q6w, q6s, q6d, fp8w, fp8s;
    void *x = nullptr, *y = nullptr, *xq = nullptr, *xs = nullptr;
    int copies = 1;
    int64_t N = 0, K = 0;
    size_t q_wbytes = 0, f_wbytes = 0;
    void setup(int64_t N_, int64_t K_, int64_t Mmax) {
        N = N_; K = K_;
        q_wbytes = (size_t)(N * (K * 6 / 8) + N * (K / 16) + N * (K / 256) * 2);
        f_wbytes = (size_t)(N * K + ((N + 127) / 128) * (K / 128) * 2);
        const size_t target = (size_t)320 << 20;
        copies = (int)std::max<size_t>(1, std::min<size_t>(32, (target + q_wbytes - 1) / q_wbytes));
        if (q_wbytes > ((size_t)256 << 20)) copies = 1;
        for (auto* v : { &q6w, &q6s, &q6d, &fp8w, &fp8s }) { for (void* p : *v) hipFree(p); v->clear(); }
        std::vector<uint8_t> junk(std::max(q_wbytes, f_wbytes));
        for (size_t i = 0; i < junk.size(); ++i) junk[i] = (uint8_t)mix32((uint32_t)i);
        for (int c = 0; c < copies; ++c) {
            q6w.push_back(dup_to_dev(junk.data(), (size_t)(N * (K * 6 / 8))));
            std::vector<int8_t> sc((size_t)(N * (K / 16)), 1);
            q6s.push_back(dup_to_dev(sc.data(), sc.size()));
            std::vector<uint16_t> dd((size_t)(N * (K / 256)), 0x3C00);
            q6d.push_back(dup_to_dev(dd.data(), dd.size() * 2));
            std::vector<uint8_t> fb((size_t)(N * K));
            for (size_t i = 0; i < fb.size(); ++i) fb[i] = junk[i] & 0x77;
            fp8w.push_back(dup_to_dev(fb.data(), fb.size()));
            std::vector<uint16_t> fs((size_t)(((N + 127) / 128) * (K / 128)), 0x3F80);
            fp8s.push_back(dup_to_dev(fs.data(), fs.size() * 2));
        }
        if (x) { hipFree(x); hipFree(y); hipFree(xq); hipFree(xs); }
        std::vector<uint16_t> xb((size_t)(Mmax * K));
        for (size_t i = 0; i < xb.size(); ++i)
            xb[i] = h_to_bf16(((float)(mix32((uint32_t)i) & 0xFFFF) / 32768.0f - 1.0f));
        x = dup_to_dev(xb.data(), xb.size() * 2);
        y = dalloc((size_t)Mmax * N * 2);
        xq = dalloc((size_t)Mmax * K);
        xs = dalloc((size_t)Mmax * (K / 128) * 4);
    }
    double q_mv(int M, int wk) {
        if (pl.mv((long)x, (long)q6w[0], (long)q6s[0], (long)q6d[0], (long)y, M, (int)K, (int)N, 0, 0, wk, 0, 0, 0))
            return -1;
        return time_us([&](int i) { pl.mv((long)x, (long)q6w[i], (long)q6s[i], (long)q6d[i], (long)y, M,
                                         (int)K, (int)N, 0, 0, wk, 0, 0, 0); }, copies, trials);
    }
    double q_wmma(int M, int tt) {
        if (pl.wmma((long)x, (long)q6w[0], (long)q6s[0], (long)q6d[0], (long)y, M, (int)K, (int)N, 0, 0, tt, 0, 0))
            return -1;
        return time_us([&](int i) { pl.wmma((long)x, (long)q6w[i], (long)q6s[i], (long)q6d[i], (long)y, M,
                                            (int)K, (int)N, 0, 0, tt, 0, 0); }, copies, trials);
    }
    double f_mt(int M) {
        if (pl.fp8_mt((long)x, (long)fp8w[0], (long)fp8s[0], (long)y, M, (int)K, (int)N, 0, 0, 0, 0, 0)) return -1;
        return time_us([&](int i) { pl.fp8_mt((long)x, (long)fp8w[i], (long)fp8s[i], (long)y, M, (int)K,
                                              (int)N, 0, 0, 0, 0, 0); }, copies, trials);
    }
    double f_quant(int M) {
        return time_us([&](int) { pl.quant((long)x, (long)xq, (long)xs, M, (int)K, 0, 0, 0, 0); }, 1, trials);
    }
    double f_m16(int M) {
        const long sb = pl.fp8_m16_scratch(M, (int)K, (int)N);
        void* s = sb > 0 ? dalloc((size_t)sb) : nullptr;
        pl.quant((long)x, (long)xq, (long)xs, M, (int)K, 0, 0, 0, 0);
        double r = -1;
        if (!pl.fp8_m16((long)xq, (long)xs, (long)fp8w[0], (long)fp8s[0], (long)y, M, (int)K, (int)N, 0, 0,
                        0, 0, (long)s, sb > 0 ? sb : 0, 0))
            r = time_us([&](int i) { pl.fp8_m16((long)xq, (long)xs, (long)fp8w[i], (long)fp8s[i], (long)y, M,
                                                (int)K, (int)N, 0, 0, 0, 0, (long)s, sb > 0 ? sb : 0, 0); },
                        copies, trials);
        if (s) hipFree(s);
        return r;
    }
    double f_tiled(int M) {
        pl.quant((long)x, (long)xq, (long)xs, M, (int)K, 0, 0, 0, 0);
        if (pl.fp8_tiled((long)xq, (long)xs, (long)fp8w[0], (long)fp8s[0], (long)y, M, (int)K, (int)N, 0, 0,
                         0, 0, 0))
            return -1;
        return time_us([&](int i) { pl.fp8_tiled((long)xq, (long)xs, (long)fp8w[i], (long)fp8s[i], (long)y, M,
                                                 (int)K, (int)N, 0, 0, 0, 0, 0); }, copies, trials);
    }
};

static void bench_main(Plugin& pl, int trials, bool tokeq) {
    /* the task's three shapes, plus the Qwen3.6 dense set for context */
    const Shape shapes[] = {
        { "attn_qkv 8192x2048", 8192, 2048 },
        { "attn_output 2048x4096", 2048, 4096 },
        { "lm_head 248320x2048", 248320, 2048 },
        { "attn_gate 4096x2048", 4096, 2048 },
        { "shexp gate,up 512x2048", 512, 2048 },
        { "shexp down 2048x512", 2048, 512 },
    };
    const int Ms[] = { 1, 4, 8, 32, 512, 4096 };
    Bench B{ pl, trials };
    std::map<std::string, std::map<int, double>> q_best, f_best;
    std::printf("shape,M,kernel,us,GB/s(weights),notes\n");
    for (const Shape& s : shapes) {
        const bool big = s.N > 100000;
        B.setup(s.N, s.K, 4096);
        for (int M : Ms) {
            if (big && M > 32) continue;
            struct R { std::string k; double us; size_t bytes; std::string note; };
            std::vector<R> rs;
            if (M <= 8) {
                double best = 1e30;
                for (int wk : { 0, 1, 2, 4 }) {
                    const double t = B.q_mv(M, wk);
                    if (t > 0) { rs.push_back({ "q6k_mv wk" + std::to_string(wk), t, B.q_wbytes, "" }); best = std::min(best, t); }
                }
                q_best[s.name][M] = best;
                const double f = B.f_mt(M);
                if (f > 0) { rs.push_back({ "fp8a16_mt", f, B.f_wbytes, "bf16 act, no quant launch" }); f_best[s.name][M] = f; }
            }
            double bw = 1e30;
            for (int tt : { 0, 1, 2, 4, 8 }) {
                if (tt && (M > 16 * tt && tt != 8)) continue;
                const double t = B.q_wmma(M, tt);
                if (t > 0) { rs.push_back({ "q6k_wmma tt" + std::to_string(tt), t, B.q_wbytes, "" }); bw = std::min(bw, t); }
            }
            if (M > 8) q_best[s.name][M] = bw;
            if (M > 8 && M <= 64) {
                const double f = B.f_m16(M), qz = B.f_quant(M);
                if (f > 0) {
                    rs.push_back({ "fp8a8_m16 (gemm only)", f, B.f_wbytes, "" });
                    rs.push_back({ "fp8a8_m16 + quant_act", f + qz, B.f_wbytes, "" });
                    f_best[s.name][M] = f + qz;
                }
            }
            if (M > 8) {
                const double f = B.f_tiled(M), qz = B.f_quant(M);
                if (f > 0) {
                    rs.push_back({ "fp8a8_tiled (gemm only)", f, B.f_wbytes, "" });
                    rs.push_back({ "fp8a8_tiled + quant_act", f + qz, B.f_wbytes, "" });
                    if (!f_best[s.name].count(M) || f + qz < f_best[s.name][M]) f_best[s.name][M] = f + qz;
                }
            }
            for (const R& r : rs) {
                const double gbs = (double)r.bytes / (r.us * 1e3);
                const double tf = M > 8 ? 2.0 * M * s.N * s.K / (r.us * 1e6) : 0;
                std::printf("%s,%d,%s,%.1f,%.1f,%s%s\n", s.name, M, r.k.c_str(), r.us, gbs,
                            M > 8 ? ("TFLOP/s " + std::to_string(tf).substr(0, 5) + " ").c_str() : "",
                            r.note.c_str());
            }
        }
    }
    if (tokeq) {
        std::printf("\nper-token dense time (sum over the model's Q6_K matrices), us\n");
        const struct { const char* name; int count; } cnt[] = {
            { "attn_qkv 8192x2048", 40 }, { "attn_gate 4096x2048", 30 }, { "attn_output 2048x4096", 40 },
            { "shexp gate,up 512x2048", 100 }, { "shexp down 2048x512", 40 }, { "lm_head 248320x2048", 1 } };
        for (int M : { 1, 4, 8 }) {
            double q = 0, f = 0;
            for (const auto& c : cnt) { q += c.count * q_best[c.name][M]; f += c.count * f_best[c.name][M]; }
            std::printf("M=%d: Q6_K %.0f us, FP8 %.0f us  -> dense-only ceiling %.0f vs %.0f steps/s\n",
                        M, q, f, 1e6 / q, 1e6 / f);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s <libhalo.so> layout|check|real|bench|tokeq ...\n", argv[0]); return 2; }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    Plugin pl;
    if (!load_plugin(&pl, argv[1])) return 2;
    const std::string mode = argv[2];
    if (mode == "layout") test_layout(pl);
    else if (mode == "check") { test_check(pl); test_rows(pl); }
    else if (mode == "real") {
        std::vector<std::string> st; for (int i = 4; i < argc; ++i) st.push_back(argv[i]);
        test_real(pl, argv[3], st);
    } else if (mode == "bench" || mode == "tokeq")
        bench_main(pl, argc > 3 ? std::atoi(argv[3]) : 5, mode == "tokeq");
    else { std::printf("unknown mode\n"); return 2; }
    std::printf("\n%d ok, %d FAIL\n", g_ok, g_fail);
    return g_fail ? 1 : 0;
}
