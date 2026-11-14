/* halo_q80_bench -- the ggml Q8_0 dense kernels (libhalo/halo_gemm_q80a16.hip): layout hooks,
 * numerics against host models and ggml's own dequantisation, and speed against the FP8 dense
 * kernels. Not part of the plugin and not built by CMake:
 *
 *   clang++ -x hip --offload-arch=gfx1151 -O2 -std=gnu++17 -ffp-contract=off -I abi -I libhalo \
 *       -I build/rad_include tools/q80_bench/halo_q80_bench.hip -o halo_q80_bench -ldl -lamdhip64
 *   halo_q80_bench <libhalo.so> layout                 host-only layout/relayout/unrelayout tests
 *   halo_q80_bench <libhalo.so> check                  numerics on random planes
 *   halo_q80_bench <libhalo.so> real <dir> <stem>...   numerics on GGUF Q8_0 blocks dumped by
 *                                                      gguf_q80_dump.py (every element, bit for bit)
 *   halo_q80_bench <libhalo.so> bench [trials]         speed, Q8_0 vs FP8, Qwen3.6 dense shapes
 *   halo_q80_bench <libhalo.so> tokeq [trials]         per-decoded-token dense time from the shapes
 */
#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"
#include "rad_encoding.h"
#include "halo_q80.h"

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
    int (*mv)(long, long, long, long, int, int, int, int, int, int, int, int, long) = nullptr;
    int (*wmma)(long, long, long, long, int, int, int, int, int, int, int, long) = nullptr;
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
           sym(pl->h, "halo_gemm_q80a16_nt_mv", &pl->mv) && sym(pl->h, "halo_gemm_q80a16_nt_wmma", &pl->wmma) &&
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

/* ------------------------------------------------------------------ Q8_0 planes */
struct QP {
    int64_t N = 0, K = 0;
    std::vector<int8_t> q;       /* [N][K] canonical codes plane */
    std::vector<uint16_t> d;     /* [N][K/32] canonical scale plane, f16 bits */
    int64_t KB() const { return K / 32; }
};
static QP rand_planes(int64_t N, int64_t K, uint32_t seed) {
    QP p; p.N = N; p.K = K;
    p.q.resize((size_t)(N * K)); p.d.resize((size_t)(N * K / 32));
    for (int64_t i = 0; i < N * K; ++i) p.q[(size_t)i] = (int8_t)(mix32((uint32_t)i ^ seed) & 0xFF);
    for (int64_t i = 0; i < N * K / 32; ++i) {
        const uint32_t r = mix32((uint32_t)i * 2654435761u ^ (seed + 77u));
        /* f16 with exponent 6..20 (2^-9 .. 2^5), random mantissa and sign: a normal range */
        p.d[(size_t)i] = (uint16_t)(((r & 1u) << 15) | ((6u + (r >> 1) % 15u) << 10) | ((r >> 8) & 0x3FFu));
    }
    return p;
}
/* Planes from `rows` x K of raw GGUF block_q8_0 (d f16 @0, qs i8[32] @2): split_q8_0 of
 * core/format/ggml_planes.h. Also fills the ggml dequantisation straight from the blocks. */
static QP planes_from_blocks(const std::vector<uint8_t>& blob, int64_t rows, int64_t K,
                             std::vector<float>* ggml_deq) {
    QP p; p.N = rows; p.K = K;
    p.q.resize((size_t)(rows * K)); p.d.resize((size_t)(rows * K / 32));
    if (ggml_deq) ggml_deq->resize((size_t)(rows * K));
    for (int64_t b = 0; b < rows * K / 32; ++b) {
        const uint8_t* blk = blob.data() + (size_t)b * 34;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        p.d[(size_t)b] = dh;
        std::memcpy(&p.q[(size_t)b * 32], blk + 2, 32);
        if (ggml_deq)   /* dequantize_row_q8_0: y[j] = x[i].qs[j] * GGML_FP16_TO_FP32(x[i].d) */
            for (int j = 0; j < 32; ++j)
                (*ggml_deq)[(size_t)b * 32 + j] = (float)(int8_t)blk[2 + j] * halo_q80::f16_to_f32(dh);
    }
    return p;
}

static RadEncoding q80_enc() {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, "affine");
    rad_enc_add_plane(&e, "codes", RAD_I8, 1, 1);
    rad_enc_add_plane(&e, "scale", RAD_F16, 1, 32);
    return e;
}
static RadTensor planeT(void* data, uint32_t dt, int64_t r, int64_t c) {
    RadTensor t; std::memset(&t, 0, sizeof t);
    t.data = data; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c;
    rad_tensor_pack(&t);
    return t;
}
/* Both operands through the row's own hooks. */
static bool stored_q80(const RadKernelInfo* r, const QP& p, std::vector<uint8_t>* wq, std::vector<uint8_t>* ws) {
    std::vector<RadParam> lp = { PI("N", p.N), PI("K", p.K), PI("group", 32), PS("dtype", "q80a16") };
    const RadEncoding enc = q80_enc();
    for (int operand = 2; operand <= 3; ++operand) {
        int sel[1] = { operand == 2 ? 0 : 1 };
        RadTensor pt = operand == 2 ? planeT((void*)p.q.data(), RAD_I8, p.N, p.K)
                                    : planeT((void*)p.d.data(), RAD_F16, p.N, p.KB());
        RadLayout L; std::memset(&L, 0, sizeof L);
        int rc = r->layout(lp.data(), (int)lp.size(), operand, &enc, sel, &pt, 1, &L);
        if (rc != RAD_OK) { std::printf("layout rc %d operand %d\n", rc, operand); return false; }
        std::vector<uint8_t>& out = operand == 2 ? *wq : *ws;
        out.assign((size_t)L.bytes, 0);
        rc = r->relayout(lp.data(), (int)lp.size(), operand, &enc, sel, &pt, 1, out.data(), L.bytes);
        if (rc != RAD_OK) { std::printf("relayout rc %d operand %d\n", rc, operand); return false; }
    }
    return true;
}

/* ================================================================== layout tests (host only) */
static void test_layout(const Plugin& pl) {
    std::printf("[layout]\n");
    const RadKernelInfo* rows[2] = { find(pl, "gemm_q80a16_nt_m8"), find(pl, "gemm_q80a16_nt_wmma") };
    for (const RadKernelInfo* r : rows) {
        if (!r || !r->layout || !r->relayout || !r->unrelayout) { ok("rows and hooks present", false); return; }
    }
    const RadKernelInfo* r = rows[0];
    ok("both rows share the layout hooks", rows[0]->layout == rows[1]->layout && rows[0]->relayout == rows[1]->relayout &&
                                           rows[0]->unrelayout == rows[1]->unrelayout);
    const RadEncoding enc = q80_enc();
    const int64_t G = 64;

    /* (1) accepted shapes: relayout equals the tile formula, unrelayout inverts, nothing outside the
     *     destination is touched, and every destination byte is written */
    const int64_t shapes[][2] = { {16, 32}, {16, 64}, {48, 96}, {64, 2048}, {128, 480}, {32, 8192} };
    for (const auto& sh : shapes) {
        const int64_t N = sh[0], K = sh[1];
        QP p = rand_planes(N, K, 1234u + (uint32_t)N);
        std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 32) };
        bool good = true; std::string why;
        for (int operand = 2; operand <= 3 && good; ++operand) {
            int sel[1] = { operand == 2 ? 0 : 1 };
            RadTensor pt = operand == 2 ? planeT(p.q.data(), RAD_I8, N, K) : planeT(p.d.data(), RAD_F16, N, K / 32);
            RadLayout L; std::memset(&L, 0, sizeof L);
            if (r->layout(lp.data(), 3, operand, &enc, sel, &pt, 1, &L) != RAD_OK) { good = false; why = "layout refused"; break; }
            const int64_t want = operand == 2 ? N * K : N * (K / 32) * 2;
            if (L.bytes != want) { good = false; why = "bytes " + std::to_string(L.bytes) + " want " + std::to_string(want); break; }
            std::vector<uint8_t> a((size_t)(L.bytes + 2 * G), 0x00), b((size_t)(L.bytes + 2 * G), 0xFF);
            const int ra = r->relayout(lp.data(), 3, operand, &enc, sel, &pt, 1, a.data() + G, L.bytes);
            const int rb = r->relayout(lp.data(), 3, operand, &enc, sel, &pt, 1, b.data() + G, L.bytes);
            if (ra != RAD_OK || rb != RAD_OK) { good = false; why = "relayout rc"; break; }
            for (int64_t i = 0; i < G; ++i)
                if (a[(size_t)i] != 0x00 || a[(size_t)(G + L.bytes + i)] != 0x00 || b[(size_t)i] != 0xFF ||
                    b[(size_t)(G + L.bytes + i)] != 0xFF) { good = false; why = "guard bytes touched"; break; }
            if (!good) break;
            if (std::memcmp(a.data() + G, b.data() + G, (size_t)L.bytes)) { good = false; why = "a destination byte was not written"; break; }
            /* the formula */
            const uint8_t* st = a.data() + G;
            for (int64_t n = 0; n < N && good; ++n)
                for (int64_t k = 0; k < K; ++k) {
                    if (operand == 2) {
                        if ((int8_t)st[halo_q80::code_off(n, k, K)] != p.q[(size_t)(n * K + k)]) { good = false; why = "code placement"; break; }
                    } else if (k % 32 == 0) {
                        uint16_t h; std::memcpy(&h, st + 2 * halo_q80::scale_idx(n, k / 32, K), 2);
                        if (h != p.d[(size_t)(n * (K / 32) + k / 32)]) { good = false; why = "scale placement"; break; }
                    }
                }
            if (!good) break;
            /* inverse, into a guard-bracketed plane */
            const size_t pb = (size_t)(operand == 2 ? N * K : N * (K / 32) * 2);
            std::vector<uint8_t> back(pb + 2 * G, 0x5C);
            RadTensor bt = operand == 2 ? planeT(back.data() + G, RAD_I8, N, K) : planeT(back.data() + G, RAD_F16, N, K / 32);
            if (r->unrelayout(lp.data(), 3, operand, &enc, sel, st, L.bytes, &bt, 1) != RAD_OK) { good = false; why = "unrelayout rc"; break; }
            const uint8_t* orig = operand == 2 ? (const uint8_t*)p.q.data() : (const uint8_t*)p.d.data();
            if (std::memcmp(back.data() + G, orig, pb)) { good = false; why = "roundtrip differs"; break; }
            for (int64_t i = 0; i < G; ++i)
                if (back[(size_t)i] != 0x5C || back[(size_t)(G + pb + i)] != 0x5C) { good = false; why = "unrelayout guard bytes"; break; }
        }
        ok("N=" + std::to_string(N) + " K=" + std::to_string(K) + ": formula, whole destination written, guards intact, roundtrip", good, why);
    }

    /* (2) refused shapes, per hook, with a guard-bracketed destination and plane: nothing written */
    struct Bad { int64_t N, K; int group; const char* why; };
    const Bad bads[] = { {16, 16, 32, "K half a block"}, {16, 31, 32, "K=31"}, {16, 48, 32, "K=48 (1.5 blocks)"},
                         {16, 33, 32, "K=33"}, {16, 0, 32, "K=0"}, {16, -32, 32, "K<0"}, {8, 64, 32, "N=8"},
                         {17, 64, 32, "N=17"}, {24, 64, 32, "N=24 (1.5 tiles)"}, {0, 64, 32, "N=0"}, {16, 64, 64, "group 64"},
                         {16, 64, 128, "group 128"}, {int64_t(1) << 30, int64_t(1) << 30, 32, "N*K past the bound"} };
    for (const Bad& b : bads) {
        bool good = true; std::string why;
        /* a plane sized for the (rounded-up) shape so only the hook's shape check can refuse */
        const int64_t Np = std::max<int64_t>(16, ((b.N > 0 && b.N < 4096 ? b.N : 16) + 15) / 16 * 16);
        const int64_t Kp = std::max<int64_t>(32, ((b.K > 0 && b.K < 4096 ? b.K : 32) + 31) / 32 * 32);
        QP p = rand_planes(Np, Kp, 5);
        std::vector<RadParam> lp = { PI("N", b.N), PI("K", b.K), PI("group", b.group) };
        for (int operand = 2; operand <= 3; ++operand) {
            int sel[1] = { operand == 2 ? 0 : 1 };
            RadTensor pt = operand == 2 ? planeT(p.q.data(), RAD_I8, Np, Kp) : planeT(p.d.data(), RAD_F16, Np, Kp / 32);
            if (b.N > 0 && b.K > 0) {   /* plane extents that match the declared (bad) shape where small */
                if (b.N < 4096 && b.K < 4096) {
                    p = rand_planes(b.N, b.K, 5);   /* ragged planes of the declared shape */
                    pt = operand == 2 ? planeT(p.q.data(), RAD_I8, b.N, b.K) : planeT(p.d.data(), RAD_F16, b.N, std::max<int64_t>(1, b.K / 32));
                }
            }
            RadLayout L; std::memset(&L, 0, sizeof L);
            const int rl = r->layout(lp.data(), 4 - 1, operand, &enc, sel, &pt, 1, &L);
            std::vector<uint8_t> dst(1024 + 2 * G, 0xA5);
            const int rr = r->relayout(lp.data(), 3, operand, &enc, sel, &pt, 1, dst.data() + G, 1024);
            std::vector<uint8_t> src(1024, 0x11), plane(1024 + 2 * G, 0x5C);
            RadTensor ot = planeT(plane.data() + G, operand == 2 ? RAD_I8 : RAD_F16, 1, 1);
            const int ru = r->unrelayout(lp.data(), 3, operand, &enc, sel, src.data(), 1024, &ot, 1);
            bool clean = true;
            for (size_t i = 0; i < dst.size(); ++i) clean = clean && dst[i] == 0xA5;
            for (size_t i = 0; i < plane.size(); ++i) clean = clean && plane[i] == 0x5C;
            if (rl != RAD_E_SHAPE || rr != RAD_E_SHAPE || ru != RAD_E_SHAPE || !clean) {
                good = false;
                why = "operand " + std::to_string(operand) + " rc " + std::to_string(rl) + "/" + std::to_string(rr) + "/" + std::to_string(ru) + (clean ? "" : " wrote memory");
            }
        }
        ok(std::string("refuses ") + b.why + " (layout, relayout, unrelayout; nothing written)", good, why);
    }

    /* (3) a good shape, wrong sizes and wrong encodings */
    {
        const int64_t N = 32, K = 128;
        QP p = rand_planes(N, K, 9);
        std::vector<RadParam> lp = { PI("N", N), PI("K", K), PI("group", 32) };
        int sel[1] = { 0 };
        RadTensor pt = planeT(p.q.data(), RAD_I8, N, K);
        RadLayout L; std::memset(&L, 0, sizeof L);
        r->layout(lp.data(), 3, 2, &enc, sel, &pt, 1, &L);
        std::vector<uint8_t> dst((size_t)L.bytes + 2 * G, 0xA5);
        const int r1 = r->relayout(lp.data(), 3, 2, &enc, sel, &pt, 1, dst.data() + G, L.bytes - 1);
        const int r2 = r->relayout(lp.data(), 3, 2, &enc, sel, &pt, 1, dst.data() + G, L.bytes + 1);
        bool clean = true; for (uint8_t v : dst) clean = clean && v == 0xA5;
        ok("destination one byte short or long is refused, nothing written", r1 == RAD_E_SHAPE && r2 == RAD_E_SHAPE && clean,
           "rc " + std::to_string(r1) + "/" + std::to_string(r2));
        RadTensor wrongK = planeT(p.q.data(), RAD_I8, N, K - 32);
        const int r3 = r->layout(lp.data(), 3, 2, &enc, sel, &wrongK, 1, &L);
        RadTensor wrongN = planeT(p.q.data(), RAD_I8, N - 16, K);
        const int r4 = r->layout(lp.data(), 3, 2, &enc, sel, &wrongN, 1, &L);
        ok("plane extents that disagree with N, K are refused", r3 == RAD_E_SHAPE && r4 == RAD_E_SHAPE, "rc " + std::to_string(r3) + "/" + std::to_string(r4));
        RadEncoding bad = enc;
        rad_enc_clear(&bad); rad_enc_copy_str(bad.scheme, "affine");
        rad_enc_add_plane(&bad, "codes", RAD_I8, 1, 1);
        rad_enc_add_plane(&bad, "scale", RAD_BF16, 1, 32);
        int sel1[1] = { 1 };
        RadTensor st = planeT(p.d.data(), RAD_BF16, N, K / 32);
        const int r5 = r->layout(lp.data(), 3, 3, &bad, sel1, &st, 1, &L);
        RadEncoding bad2; rad_enc_clear(&bad2); rad_enc_copy_str(bad2.scheme, "affine");
        rad_enc_add_plane(&bad2, "codes", RAD_I8, 1, 1);
        rad_enc_add_plane(&bad2, "scale", RAD_F16, 1, 128);   /* the 128-group int8 (w8a16) encoding */
        RadTensor st2 = planeT(p.d.data(), RAD_F16, N, K / 128);
        const int r6 = r->layout(lp.data(), 3, 3, &bad2, sel1, &st2, 1, &L);
        ok("scale plane that is bf16, or per 128 rather than per 32, is refused as a dtype", r5 == RAD_E_DTYPE && r6 == RAD_E_DTYPE,
           "rc " + std::to_string(r5) + "/" + std::to_string(r6));
        const int r7 = r->layout(lp.data(), 3, 0, &enc, sel, &pt, 1, &L);
        ok("operand that is neither codes nor scale is unsupported", r7 == RAD_E_UNSUPPORTED, "rc " + std::to_string(r7));
    }
}

/* ================================================================== device runs */
struct Dev {
    void *x = nullptr, *wq = nullptr, *ws = nullptr, *y = nullptr;
};
static float* d_to_host_f32(void* dptr, size_t n, std::vector<float>* out) {
    out->resize(n);
    HCK(hipMemcpy(out->data(), dptr, n * 4, hipMemcpyDeviceToHost));
    return out->data();
}

/* the matvec's ordered host model (halo_gemm_q80a16.hip header) */
static void model_mv(const QP& p, const std::vector<float>& x, int M, int wk, int64_t K_ld_x, std::vector<float>* y) {
    const int64_t N = p.N, K = p.K, KB = K / 32;
    y->assign((size_t)(M * N), 0.0f);
    for (int m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            float tot = 0;
            for (int w = 0; w < wk; ++w) {
                float a[2] = { 0, 0 };
                for (int h = 0; h < 2; ++h)
                    for (int64_t kb = w; kb < KB; kb += wk) {
                        float part = 0.0f;
                        for (int i = 0; i < 16; ++i) {
                            const int64_t k = kb * 32 + h * 16 + i;
                            part = std::fmaf((float)p.q[(size_t)(n * K + k)], x[(size_t)(m * K_ld_x + k)], part);
                        }
                        a[h] = std::fmaf(halo_q80::f16_to_f32(p.d[(size_t)(n * KB + kb)]), part, a[h]);
                    }
                const float wt = a[0] + a[1];
                tot = w == 0 ? wt : tot + wt;
            }
            (*y)[(size_t)(m * N + n)] = tot;
        }
}
/* the double-precision value, and sum |w x| (the scale an f32 error is judged against) */
static void ref_double(const QP& p, const std::vector<float>& x, int M, std::vector<double>* y, std::vector<double>* mag) {
    const int64_t N = p.N, K = p.K, KB = K / 32;
    y->assign((size_t)(M * N), 0); mag->assign((size_t)(M * N), 0);
    for (int m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double s = 0, a = 0;
            for (int64_t k = 0; k < K; ++k) {
                const double w = (double)halo_q80::f16_to_f32(p.d[(size_t)(n * KB + k / 32)]) * (double)p.q[(size_t)(n * K + k)];
                s += w * (double)x[(size_t)(m * K + k)];
                a += std::fabs(w * (double)x[(size_t)(m * K + k)]);
            }
            (*y)[(size_t)(m * N + n)] = s; (*mag)[(size_t)(m * N + n)] = a;
        }
}

struct Run {
    Plugin* pl; const RadKernelInfo* row;
    QP p; std::vector<uint8_t> wq, ws; void *dwq = nullptr, *dws = nullptr;
    void upload(Plugin* plug, const QP& planes) {
        pl = plug; row = find(*plug, "gemm_q80a16_nt_m8"); p = planes;
        if (dwq) { hipFree(dwq); hipFree(dws); }
        if (!stored_q80(row, p, &wq, &ws)) std::exit(2);
        dwq = dup_to_dev(wq.data(), wq.size()); dws = dup_to_dev(ws.data(), ws.size());
    }
    /* f32 outputs (the exact accumulator) */
    int mv(const std::vector<uint16_t>& xb, int M, int wk, int act, std::vector<float>* y) {
        void* dx = dup_to_dev(xb.data(), xb.size() * 2);
        void* dy = dalloc((size_t)M * p.N * 4);
        const int rc = pl->mv((long)dx, (long)dwq, (long)dws, (long)dy, M, (int)p.K, (int)p.N, 0, 0, wk, act, 1, 0);
        HCK(hipDeviceSynchronize());
        if (rc == 0) d_to_host_f32(dy, (size_t)M * p.N, y);
        hipFree(dx); hipFree(dy);
        return rc;
    }
    int wmma(const std::vector<uint16_t>& xb, int M, int tt, std::vector<float>* y) {
        void* dx = dup_to_dev(xb.data(), xb.size() * 2);
        void* dy = dalloc((size_t)M * p.N * 4);
        const int rc = pl->wmma((long)dx, (long)dwq, (long)dws, (long)dy, M, (int)p.K, (int)p.N, 0, 0, tt, 1, 0);
        HCK(hipDeviceSynchronize());
        if (rc == 0) d_to_host_f32(dy, (size_t)M * p.N, y);
        hipFree(dx); hipFree(dy);
        return rc;
    }
    /* the bf16-output path, as the row runs it */
    int mv_bf16(const std::vector<uint16_t>& xb, int M, int wk, std::vector<uint16_t>* y) {
        void* dx = dup_to_dev(xb.data(), xb.size() * 2);
        void* dy = dalloc((size_t)M * p.N * 2);
        const int rc = pl->mv((long)dx, (long)dwq, (long)dws, (long)dy, M, (int)p.K, (int)p.N, 0, 0, wk, 0, 0, 0);
        HCK(hipDeviceSynchronize());
        y->resize((size_t)M * p.N);
        if (rc == 0) HCK(hipMemcpy(y->data(), dy, (size_t)M * p.N * 2, hipMemcpyDeviceToHost));
        hipFree(dx); hipFree(dy);
        return rc;
    }
    int wmma_bf16(const std::vector<uint16_t>& xb, int M, int tt, std::vector<uint16_t>* y) {
        void* dx = dup_to_dev(xb.data(), xb.size() * 2);
        void* dy = dalloc((size_t)M * p.N * 2);
        const int rc = pl->wmma((long)dx, (long)dwq, (long)dws, (long)dy, M, (int)p.K, (int)p.N, 0, 0, tt, 0, 0);
        HCK(hipDeviceSynchronize());
        y->resize((size_t)M * p.N);
        if (rc == 0) HCK(hipMemcpy(y->data(), dy, (size_t)M * p.N * 2, hipMemcpyDeviceToHost));
        hipFree(dx); hipFree(dy);
        return rc;
    }
};

static std::vector<uint16_t> rand_acts_bf16(int M, int64_t K, uint32_t seed, std::vector<float>* xf, bool ints) {
    std::vector<uint16_t> xb((size_t)(M * K));
    xf->resize(xb.size());
    for (size_t i = 0; i < xb.size(); ++i) {
        const uint32_t r = mix32((uint32_t)i ^ seed);
        float v = ints ? (float)((int)(r % 17u) - 8) : ((float)(r & 0xFFFFu) / 32768.0f - 1.0f) * (1.0f + (float)((r >> 16) % 7u));
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

/* distance in f32 ulps (sign-magnitude ordered); 0 for equal values including +-0 */
static int64_t ulp_dist(float a, float b) {
    if (a == b) return 0;
    int32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4);
    if (x < 0) x = (int32_t)0x80000000 - x;
    if (y < 0) y = (int32_t)0x80000000 - y;
    return std::llabs((int64_t)x - (int64_t)y);
}
static int64_t max_ulp(const std::vector<float>& a, const std::vector<float>& b, int64_t* n_inexact = nullptr) {
    int64_t w = 0, c = 0;
    if (a.size() != b.size()) return 1 << 30;
    for (size_t i = 0; i < a.size(); ++i) { const int64_t d = ulp_dist(a[i], b[i]); w = std::max(w, d); c += d != 0; }
    if (n_inexact) *n_inexact = c;
    return w;
}
/* WMMA error against the exact value in the unit that matters: 2^-24 * sum|w x| (an f32 summation
 * error of one rounding is 1 unit; any order of f32 adds stays within K/32 units, the matrix
 * unit's measured worst is reported) */
static double wmma_units(const std::vector<float>& got, const std::vector<double>& exact, const std::vector<double>& mag) {
    double w = 0;
    for (size_t i = 0; i < got.size(); ++i) if (mag[i] > 0) w = std::max(w, std::fabs((double)got[i] - exact[i]) / (mag[i] * 5.96e-8));
    return w;
}

static void test_check(Plugin& pl) {
    std::printf("[check] random planes\n");
    const struct { int64_t N, K; } shp[] = { {64, 256}, {512, 2048}, {2048, 512}, {160, 96}, {1024, 1024} };
    for (const auto& s : shp) {
        QP p = rand_planes(s.N, s.K, 4242u + (uint32_t)s.N);
        Run run; run.upload(&pl, p);
        std::string tag = "N=" + std::to_string(s.N) + " K=" + std::to_string(s.K);

        /* (a) the matvec against its ordered host model, bit for bit, every WK, M = 1..8 */
        for (int M : { 1, 2, 3, 5, 8 }) {
            std::vector<float> xf; auto xb = rand_acts_bf16(M, s.K, 17u + (uint32_t)M, &xf, false);
            for (int wk : { 1, 2, 4 }) {
                std::vector<float> got, want; std::string why;
                const int rc = run.mv(xb, M, wk, 0, &got);
                model_mv(p, xf, M, wk, s.K, &want);
                ok(tag + " matvec M=" + std::to_string(M) + " WK=" + std::to_string(wk) + " bf16 act == ordered host model, bitwise",
                   rc == 0 && bits_eq(got, want, &why), rc ? "rc " + std::to_string(rc) : why);
            }
        }
        /* (b) f16 activations, same model */
        {
            const int M = 3;
            std::vector<uint16_t> xh((size_t)(M * s.K)); std::vector<float> xf(xh.size());
            for (size_t i = 0; i < xh.size(); ++i) {
                const float v = ((float)(mix32((uint32_t)i ^ 99u) & 0xFFFFu) / 32768.0f - 1.0f) * 3.0f;
                xh[i] = h_to_f16(v); xf[i] = halo_q80::f16_to_f32(xh[i]);
            }
            std::vector<float> got, want; std::string why;
            const int rc = run.mv(xh, M, 0, 1, &got);
            const int nt = (int)(s.N / 16); const int wk = nt >= 512 ? 1 : (nt >= 256 ? 2 : 4);
            model_mv(p, xf, M, wk, s.K, &want);
            ok(tag + " matvec M=3 f16 act == ordered host model, bitwise", rc == 0 && bits_eq(got, want, &why), rc ? "rc " + std::to_string(rc) : why);
        }
        /* (c) integer activations: every partial sum exact, so ANY summation order is bitwise equal.
         *     Matvec (rule WK) and every WMMA form against one host value. */
        {
            const int M = 8;
            std::vector<float> xf; auto xb = rand_acts_bf16(M, s.K, 31u, &xf, true);
            QP pi = p;   /* scales that are powers of two, so d*sum is exact too */
            for (size_t i = 0; i < pi.d.size(); ++i) pi.d[i] = (uint16_t)(((mix32((uint32_t)i) & 1u) << 15) | ((10u + mix32((uint32_t)i + 3u) % 6u) << 10));
            Run ri; ri.upload(&pl, pi);
            std::vector<float> want; model_mv(pi, xf, M, 1, s.K, &want);   /* exact => order independent */
            std::vector<float> got; std::string why;
            int rc = ri.mv(xb, M, 0, 0, &got);
            ok(tag + " integer acts: matvec == host, bitwise", rc == 0 && bits_eq(got, want, &why), why);
            std::vector<double> yd0, mg0; ref_double(pi, xf, M, &yd0, &mg0);
            for (int tt : { 1, 2, 4, 8 }) {
                std::vector<float> g2;
                rc = ri.wmma(xb, M, tt, &g2);
                const double u = rc == 0 ? wmma_units(g2, yd0, mg0) : 1e9;
                char b[100]; std::snprintf(b, sizeof b, "worst %.2f units of 2^-24 x sum|wx|; %lld ulp", u, (long long)max_ulp(g2, want));
                ok(tag + " integer acts: WMMA TT=" + std::to_string(tt) + " within 4 units of the exact sum (the unit is not bit-exact, see report)", rc == 0 && u <= 4.0, b);
            }
            if (s.N * s.K <= 1 << 22) {   /* ragged M through the WMMA kernel (M not a tile multiple), bigger M */
                for (int Mx : { 1, 7, 17, 33, 130 }) {
                    std::vector<float> xf2; auto xb2 = rand_acts_bf16(Mx, s.K, 77u + (uint32_t)Mx, &xf2, true);
                    std::vector<float> w2, g3;
                    std::vector<double> yd, mg; ref_double(pi, xf2, Mx, &yd, &mg);
                    w2.resize(yd.size()); for (size_t i = 0; i < yd.size(); ++i) w2[i] = (float)yd[i];
                    rc = ri.wmma(xb2, Mx, 0, &g3);
                    const double u = rc == 0 ? wmma_units(g3, yd, mg) : 1e9;
                    char b[100]; std::snprintf(b, sizeof b, "worst %.2f units; %lld ulp", u, (long long)max_ulp(g3, w2));
                    ok(tag + " integer acts: WMMA M=" + std::to_string(Mx) + " (rule tile) within 4 units of the exact value", rc == 0 && u <= 4.0, b);
                }
            }
        }
        /* (c2) probes for the matrix unit: d = 1 with integer activations, and one-hot readback */
        if (s.N <= 1024) {
            const int M = 8;
            QP p1 = p; for (auto& v : p1.d) v = 0x3C00;   /* d = 1.0 */
            Run r1; r1.upload(&pl, p1);
            std::vector<float> xf; auto xb = rand_acts_bf16(M, s.K, 31u, &xf, true);
            std::vector<float> want, got; std::string why; model_mv(p1, xf, M, 1, s.K, &want);
            int rc = r1.wmma(xb, M, 1, &got);
            { std::vector<double> yd1, mg1; ref_double(p1, xf, M, &yd1, &mg1);
              int64_t ni = 0; max_ulp(got, want, &ni);
              const double u = rc == 0 ? wmma_units(got, yd1, mg1) : 1e9;
              char b[120]; std::snprintf(b, sizeof b, "%lld of %zu not bit-equal, worst %.2f units of 2^-24 x sum|wx|", (long long)ni, got.size(), u);
              ok(tag + " probe: d=1, integer acts, WMMA within 4 units of the exact integer sum", rc == 0 && u <= 4.0, b); }
            std::vector<uint16_t> oh((size_t)(M * s.K), 0);
            for (int m = 0; m < M; ++m) oh[(size_t)(m * s.K + m)] = 0x3F80;
            std::vector<float> y; rc = run.wmma(oh, M, 1, &y);
            bool good = rc == 0; std::string first;
            for (int m = 0; m < M && good; ++m)
                for (int64_t n = 0; n < s.N && good; ++n) {
                    const float w = halo_q80::f16_to_f32(p.d[(size_t)(n * s.K / 32)]) * (float)p.q[(size_t)(n * s.K + m)];
                    if (ulp_dist(y[(size_t)(m * s.N + n)], w) > 4) { good = false; uint32_t gb, wb; float gg = y[(size_t)(m * s.N + n)]; std::memcpy(&gb, &gg, 4); std::memcpy(&wb, &w, 4);
                        char bb[200]; std::snprintf(bb, sizeof bb, "n=%lld m=%d q=%d d=0x%04x got 0x%08x want 0x%08x ratio %.9g", (long long)n, m, (int)p.q[(size_t)(n * s.K + m)], p.d[(size_t)(n * s.K / 32)], gb, wb, (double)gg / (double)w); first = bb; }
                }
            ok(tag + " probe: one-hot WMMA readback within 4 ulp of d*q", good, first);
        }
        /* (d) random floats: error against the double value, in units of the f32 ulp of the
         *     magnitude sum(|w x|) -- the bound any f32 summation order meets up to a small factor */
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
            ok(tag + " random acts: worst error vs double within the f32 summation bound (<= 8)", e1 <= 8.0 && e2 <= 8.0, b);
            /* the bf16 store: both rows' bf16 output is the correctly rounded f32 accumulator */
            std::vector<uint16_t> yb, yw;
            run.mv_bf16(xb, M, 0, &yb); run.wmma_bf16(xb, M, 0, &yw);
            bool b1 = true, b2 = true;
            for (size_t i = 0; i < yb.size(); ++i) {
                b1 = b1 && yb[i] == h_to_bf16(g1[i]);
                b2 = b2 && yw[i] == h_to_bf16(g2[i]);
            }
            ok(tag + " bf16 output == RNE(f32 accumulator), matvec and WMMA", b1 && b2, std::string(b1 ? "" : "matvec differs ") + (b2 ? "" : "wmma differs (the two runs are separate launches; a tie can differ only if the accumulator does)"));
        }
    }
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
        std::vector<uint8_t> blob; std::ifstream sf(dir + "/" + stem + ".shape");
        long long rows = 0, K = 0, Nfull = 0;
        if (!(sf >> rows >> K >> Nfull) || !read_file(dir + "/" + stem + ".q80", &blob) ||
            blob.size() != (size_t)(rows * (K / 32) * 34)) { ok("read " + stem, false, "missing or wrong size"); continue; }
        std::printf("[real] %s: %lld of %lld rows, K=%lld (%zu bytes of GGUF blocks)\n", stem.c_str(), rows, Nfull, K, blob.size());
        std::vector<float> ggml;
        QP p = planes_from_blocks(blob, rows, K, &ggml);
        Run run; run.upload(&pl, p);
        /* planes -> tiles -> planes equals the GGUF blocks' fields */
        /* (1) every element, both kernels: one-hot bf16 activations (M = 8 columns a launch) read the
         *     weight back through the kernel's own decode: y[m][n] = fma(d, q*1, 0) = d*q, which is
         *     ggml's dequantize_row_q8_0 value. f32 outputs, compared bitwise (+0 = -0). */
        for (int which = 0; which < 2; ++which) {
            int64_t bad = 0, cmp = 0, inexact = 0, maxud = 0; std::string first;
            const int64_t step = (K + 7) / 8 * 8;
            for (int64_t j0 = 0; j0 < K; j0 += 8) {
                const int M = 8;
                std::vector<uint16_t> xb((size_t)(M * K), 0);
                for (int m = 0; m < M; ++m) if (j0 + m < K) xb[(size_t)(m * K + j0 + m)] = 0x3F80;   /* bf16 1.0 */
                std::vector<float> y;
                const int rc = which == 0 ? run.mv(xb, M, 0, 0, &y) : run.wmma(xb, M, 0, &y);
                if (rc) { bad = -1; first = "rc " + std::to_string(rc); break; }
                for (int m = 0; m < M && j0 + m < K; ++m)
                    for (int64_t n = 0; n < rows; ++n) {
                        const float g = y[(size_t)(m * rows + n)], w = ggml[(size_t)(n * K + j0 + m)];
                        uint32_t a, b; std::memcpy(&a, &g, 4); std::memcpy(&b, &w, 4);
                        ++cmp;
                        const int64_t ud = ulp_dist(g, w);
                        if (ud) ++inexact; maxud = std::max(maxud, ud);
                        if (ud > (which == 0 ? 0 : 4)) { if (!bad) first = "n=" + std::to_string(n) + " k=" + std::to_string(j0 + m); ++bad; }
                    }
            }
            (void)step;
            ok(stem + ": " + (which == 0 ? "matvec one-hot readback of all " : "WMMA one-hot readback (within 4 ulp) of all ") + std::to_string(rows * K) +
               (which == 0 ? " weights == ggml dequantize_row_q8_0, bitwise" : " weights vs ggml dequantize_row_q8_0"), bad == 0,
               bad ? std::to_string(bad) + " beyond tolerance, " + first : std::to_string(cmp) + " compared, " + std::to_string(inexact) + " not bit-equal, max " + std::to_string(maxud) + " ulp");
        }
        /* (2) a real-weight matvec/GEMM: normal activations, error vs double */
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
    for (int i = 0; i < copies; ++i) launch(i);      /* warm */
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
    /* device buffers for one shape: `copies` independent weight sets so no launch hits a warm cache */
    std::vector<void*> q80w, q80s, fp8w, fp8s;
    void *x = nullptr, *y = nullptr, *xq = nullptr, *xs = nullptr, *scr = nullptr;
    int copies = 1;
    int64_t N = 0, K = 0;
    size_t q_wbytes = 0, f_wbytes = 0;
    void setup(int64_t N_, int64_t K_, int64_t Mmax) {
        N = N_; K = K_;
        q_wbytes = (size_t)(N * K + N * K / 16);
        f_wbytes = (size_t)(N * K + ((N + 127) / 128) * (K / 128) * 2);
        const size_t target = (size_t)320 << 20;
        copies = (int)std::max<size_t>(1, std::min<size_t>(32, (target + q_wbytes - 1) / q_wbytes));
        if (q_wbytes > ((size_t)256 << 20)) copies = 1;
        for (auto* v : { &q80w, &q80s, &fp8w, &fp8s }) { for (void* p : *v) hipFree(p); v->clear(); }
        std::vector<uint8_t> junk(std::max(q_wbytes, f_wbytes));
        for (size_t i = 0; i < junk.size(); ++i) junk[i] = (uint8_t)mix32((uint32_t)i);
        for (int c = 0; c < copies; ++c) {
            q80w.push_back(dup_to_dev(junk.data(), (size_t)(N * K)));
            /* scale plane: normal f16 values so no denormal/NaN path is timed */
            std::vector<uint16_t> sc((size_t)(N * K / 32), 0x2400);
            q80s.push_back(dup_to_dev(sc.data(), sc.size() * 2));
            /* e4m3 bytes in a finite range (no 0x7F/0xFF NaN) and bf16 scales of 1.0 */
            std::vector<uint8_t> fb((size_t)(N * K));
            for (size_t i = 0; i < fb.size(); ++i) fb[i] = junk[i] & 0x77;
            fp8w.push_back(dup_to_dev(fb.data(), fb.size()));
            std::vector<uint16_t> fs((size_t)(((N + 127) / 128) * (K / 128)), 0x3F80);
            fp8s.push_back(dup_to_dev(fs.data(), fs.size() * 2));
        }
        if (x) { hipFree(x); hipFree(y); hipFree(xq); hipFree(xs); if (scr) hipFree(scr); }
        std::vector<uint16_t> xb((size_t)(Mmax * K));
        for (size_t i = 0; i < xb.size(); ++i) xb[i] = h_to_bf16(((float)(mix32((uint32_t)i) & 0xFFFF) / 32768.0f - 1.0f));
        x = dup_to_dev(xb.data(), xb.size() * 2);
        y = dalloc((size_t)Mmax * N * 2);
        xq = dalloc((size_t)Mmax * K);
        xs = dalloc((size_t)Mmax * (K / 128) * 4);
        scr = nullptr;
    }
    /* returns us, or <0 if the kernel refused */
    double q_mv(int M, int wk) {
        if (pl.mv((long)x, (long)q80w[0], (long)q80s[0], (long)y, M, (int)K, (int)N, 0, 0, wk, 0, 0, 0)) return -1;
        return time_us([&](int i) { pl.mv((long)x, (long)q80w[i], (long)q80s[i], (long)y, M, (int)K, (int)N, 0, 0, wk, 0, 0, 0); }, copies, trials);
    }
    double q_wmma(int M, int tt) {
        if (pl.wmma((long)x, (long)q80w[0], (long)q80s[0], (long)y, M, (int)K, (int)N, 0, 0, tt, 0, 0)) return -1;
        return time_us([&](int i) { pl.wmma((long)x, (long)q80w[i], (long)q80s[i], (long)y, M, (int)K, (int)N, 0, 0, tt, 0, 0); }, copies, trials);
    }
    double f_mt(int M) {
        if (pl.fp8_mt((long)x, (long)fp8w[0], (long)fp8s[0], (long)y, M, (int)K, (int)N, 0, 0, 0, 0, 0)) return -1;
        return time_us([&](int i) { pl.fp8_mt((long)x, (long)fp8w[i], (long)fp8s[i], (long)y, M, (int)K, (int)N, 0, 0, 0, 0, 0); }, copies, trials);
    }
    double f_quant(int M) {
        return time_us([&](int) { pl.quant((long)x, (long)xq, (long)xs, M, (int)K, 0, 0, 0, 0); }, 1, trials);
    }
    double f_m16(int M) {
        const long sb = pl.fp8_m16_scratch(M, (int)K, (int)N);
        void* s = sb > 0 ? dalloc((size_t)sb) : nullptr;
        pl.quant((long)x, (long)xq, (long)xs, M, (int)K, 0, 0, 0, 0);
        double r = -1;
        if (!pl.fp8_m16((long)xq, (long)xs, (long)fp8w[0], (long)fp8s[0], (long)y, M, (int)K, (int)N, 0, 0, 0, 0, (long)s, sb > 0 ? sb : 0, 0))
            r = time_us([&](int i) { pl.fp8_m16((long)xq, (long)xs, (long)fp8w[i], (long)fp8s[i], (long)y, M, (int)K, (int)N, 0, 0, 0, 0, (long)s, sb > 0 ? sb : 0, 0); }, copies, trials);
        if (s) hipFree(s);
        return r;
    }
    double f_tiled(int M) {
        pl.quant((long)x, (long)xq, (long)xs, M, (int)K, 0, 0, 0, 0);
        if (pl.fp8_tiled((long)xq, (long)xs, (long)fp8w[0], (long)fp8s[0], (long)y, M, (int)K, (int)N, 0, 0, 0, 0, 0)) return -1;
        return time_us([&](int i) { pl.fp8_tiled((long)xq, (long)xs, (long)fp8w[i], (long)fp8s[i], (long)y, M, (int)K, (int)N, 0, 0, 0, 0, 0); }, copies, trials);
    }
};

static void bench_main(Plugin& pl, int trials, bool tokeq) {
    /* The Q8_0 dense tensors of Qwen3.6-35B-A3B UD (shapes from the GGUF's tensor list, N rows of K) */
    const Shape shapes[] = {
        { "attn_qkv / attn_q+gate 8192x2048", 8192, 2048 },
        { "attn_gate 4096x2048", 4096, 2048 },
        { "ssm_out / attn_output 2048x4096", 2048, 4096 },
        { "attn_k,v / shexp gate,up 512x2048", 512, 2048 },
        { "shexp down 2048x512", 2048, 512 },
        { "lm_head 248320x2048", 248320, 2048 },
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
                double best = 1e30; int bwk = 0;
                for (int wk : { 0, 1, 2, 4 }) {
                    const double t = B.q_mv(M, wk);
                    if (t > 0) { rs.push_back({ "q80_mv wk" + std::to_string(wk), t, B.q_wbytes, "" }); if (t < best) { best = t; bwk = wk; } }
                }
                (void)bwk;
                q_best[s.name][M] = best;
                const double f = B.f_mt(M);
                if (f > 0) { rs.push_back({ "fp8a16_mt", f, B.f_wbytes, "bf16 act, no quant launch" }); f_best[s.name][M] = f; }
            }
            double bw = 1e30;
            for (int tt : { 0, 1, 2, 4, 8 }) {
                if (tt && (M > 16 * tt && tt != 8)) continue;
                const double t = B.q_wmma(M, tt);
                if (t > 0) { rs.push_back({ "q80_wmma tt" + std::to_string(tt), t, B.q_wbytes, "" }); if (t < bw) bw = t; }
            }
            if (M > 8) q_best[s.name][M] = bw;
            if (M > 8 && M <= 64) {
                const double f = B.f_m16(M), qz = B.f_quant(M);
                if (f > 0) { rs.push_back({ "fp8a8_m16 (gemm only)", f, B.f_wbytes, "" }); rs.push_back({ "fp8a8_m16 + quant_act", f + qz, B.f_wbytes, "" }); f_best[s.name][M] = f + qz; }
            }
            if (M > 8) {
                const double f = B.f_tiled(M), qz = B.f_quant(M);
                if (f > 0) { rs.push_back({ "fp8a8_tiled (gemm only)", f, B.f_wbytes, "" }); rs.push_back({ "fp8a8_tiled + quant_act", f + qz, B.f_wbytes, "" });
                    if (!f_best[s.name].count(M) || f + qz < f_best[s.name][M]) f_best[s.name][M] = f + qz; }
            }
            for (const R& r : rs) {
                const double gbs = (double)r.bytes / (r.us * 1e3);
                const double tf = M > 8 ? 2.0 * M * s.N * s.K / (r.us * 1e6) : 0;
                std::printf("%s,%d,%s,%.1f,%.1f,%s%s\n", s.name, M, r.k.c_str(), r.us, gbs, M > 8 ? ("TFLOP/s " + std::to_string(tf).substr(0, 5) + " ").c_str() : "", r.note.c_str());
            }
        }
    }
    if (tokeq) {
        /* per decoded token (M = 1) and per 4-token verify (M = 4): the sum of the dense GEMVs of one
         * forward, counts from the GGUF (see the report) */
        std::printf("\nper-token dense time (sum over the model's Q8_0 matrices), us\n");
        const struct { const char* name; int count; } cnt[] = {
            { "attn_qkv / attn_q+gate 8192x2048", 40 }, { "attn_gate 4096x2048", 30 }, { "ssm_out / attn_output 2048x4096", 40 },
            { "attn_k,v / shexp gate,up 512x2048", 100 }, { "shexp down 2048x512", 40 }, { "lm_head 248320x2048", 1 } };
        for (int M : { 1, 4, 8 }) {
            double q = 0, f = 0, qb = 0, fb = 0;
            for (const auto& c : cnt) {
                q += c.count * q_best[c.name][M]; f += c.count * f_best[c.name][M];
            }
            (void)qb; (void)fb;
            std::printf("M=%d: Q8_0 %.0f us, FP8 %.0f us  -> dense-only ceiling %.0f vs %.0f steps/s\n", M, q, f, 1e6 / q, 1e6 / f);
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
    else if (mode == "check") test_check(pl);
    else if (mode == "real") {
        std::vector<std::string> st; for (int i = 4; i < argc; ++i) st.push_back(argv[i]);
        test_real(pl, argv[3], st);
    } else if (mode == "bench" || mode == "tokeq") bench_main(pl, argc > 3 ? std::atoi(argv[3]) : 5, mode == "tokeq");
    else { std::printf("unknown mode\n"); return 2; }
    std::printf("\n%d ok, %d FAIL\n", g_ok, g_fail);
    return g_fail ? 1 : 0;
}
