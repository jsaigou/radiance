/* halo_w3_bench -- stage 2 of the 3-bit routed-expert format (w3nl64a8h): the grouped MoE GEMM's
 * 3-bit form against a host reference, and its speed against the 4- and 5-bit forms.
 *
 * Not part of the plugin and not built by CMake; build it beside a built libhalo:
 *
 *   hipcc -O2 -std=c++17 --offload-arch=gfx1151 -ffp-contract=off -I abi -I libhalo \
 *       libhalo/halo_w3_bench.hip -o halo_w3_bench -ldl
 *   HALO_W3_IL=0|1 halo_w3_bench <libhalo.so> check [real <stem>...]
 *   HALO_W3_IL=0|1 halo_w3_bench <libhalo.so> bench [reps]
 *   HALO_W3_IL=0|1 halo_w3_bench <libhalo.so> decode [trials [reps]]
 *   halo_w3_bench <libhalo.so> readpat [reps]
 *
 * check   Builds experts from canonical planes (random codes, or the real quantised planes
 *         `<stem>.codes`/`<stem>.scale` that tests/w3_emulate_test.cpp writes under $W3_DUMP), takes
 *         them through the row's own layout and relayout hooks, runs the kernel in every form the
 *         launcher has (the rule, the aligned and uniform prefill forms, the (tile, run) decode form,
 *         the sorted operand's decode kernel, one or two K blocks a pass) and compares the bf16
 *         output bit for bit with a host model that does the kernel's arithmetic in the kernel's
 *         order (the K rotation, f32 raw sums per 64-wide scale group, the fold, the bf16 rounding).
 *         The activations are drawn so every f32 partial sum is exact, which is what makes the
 *         comparison bitwise: the matrix unit's own summation order cannot differ from the host's.
 * bench   Times the gate/up and down projections of Qwen3.6-35B-A3B (N 1024 K 2048, N 2048 K 512, 256
 *         experts, top 8) for each format at several token counts, on weights cycled over layers and
 *         routings so the experts are not resident in the caches.
 * decode  The three formats at T = 8, 16, 32 with the timing trials interleaved: median, min, max.
 * readpat Reads the same bytes in each format's per-thread access pattern with no arithmetic: the
 *         bandwidth the layout alone allows.
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
#include <fstream>
#include <string>
#include <vector>

#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::printf("HIP error %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e_)); std::exit(2); } } while (0)

static int g_fail = 0;
/* How the host model contracts the kernel's fold: bit 0 the second group's multiply-add, bit 1 the
 * block's accumulate (the device compiler may or may not fuse them; $W3_FMA picks the model that
 * matches, see the report). */
static int g_fma = 0;
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
/* Round to nearest even, as halo::f32_to_bf16's source does. (On exact ties the device's result is
 * not consistently even: about one tie in ten goes the other way, which is the matrix unit's sums
 * not being bit-exact, not the store; see the report.) */
static uint16_t h_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float h_bf16(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f;
}
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

/* ------------------------------------------------------------------ the three formats */
struct Fmt {
    const char* name;      /* the op's dtype */
    uint32_t    codes;     /* the container's code dtype */
    uint32_t    served;    /* the kernel's weight dtype */
    int         nt;        /* table entries */
    const float* table;
    int         bits;      /* bits a weight in the served plane */
};
static const Fmt kW3 = { "w3nl64a8h", RAD_U3, HALO_DT_MOEW3, 8, kW3nlTable, 3 };
static const Fmt kW4 = { "w4nl64a8h", RAD_U4, HALO_DT_MOEW4, 16, kW4nlTable, 4 };
static const Fmt kW5 = { "w5nl64a8h", RAD_U5, HALO_DT_MOEW5, 32, kW5nlTable, 5 };

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

/* The canonical planes of a [N, K] expert: random codes, E4M3 scales in a normal range (bytes
 * 0x30..0x4F), the constant second level and the table. With `real_*` the codes and scales are read
 * from the files tests/w3_emulate_test.cpp wrote. */
static bool read_file(const std::string& p, std::vector<uint8_t>* out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}
static Planes make_planes(const Fmt& F, int64_t N, int64_t K, uint32_t seed,
                          const std::string& real = "") {
    Planes p;
    rad_enc_clear(&p.enc);
    rad_enc_copy_str(p.enc.scheme, "affine");
    rad_enc_copy_str(p.enc.transform, "fwht128");
    rad_enc_add_plane(&p.enc, "codes", F.codes, 1, 1);
    rad_enc_add_plane(&p.enc, "scale", RAD_F8E4M3, 1, 64);
    rad_enc_add_plane(&p.enc, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&p.enc, "table", RAD_F32, 1, F.nt);
    p.size(N, K);
    std::memcpy(p.data[2].data(), &kW4G64Scale, sizeof kW4G64Scale);
    std::memcpy(p.data[3].data(), F.table, sizeof(float) * (size_t)F.nt);
    if (!real.empty()) {
        std::vector<uint8_t> c, s;
        if (!read_file(real + ".codes", &c) || !read_file(real + ".scale", &s) ||
            c.size() != p.data[0].size() || s.size() != p.data[1].size()) {
            std::printf("real planes %s: missing or the wrong size (%zu/%zu bytes, want %zu/%zu)\n",
                        real.c_str(), c.size(), s.size(), p.data[0].size(), p.data[1].size());
            std::exit(2);
        }
        p.data[0] = c;
        p.data[1] = s;
        return p;
    }
    const int64_t crow = p.row_bytes(0), srow = p.row_bytes(1);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k)
            rad_store_code(p.data[0].data() + n * crow, F.codes, k,
                           std::getenv("W3_CODE") ? std::atoi(std::getenv("W3_CODE"))
                                                  : (int)(mix32((uint32_t)(n * K + k) ^ seed) % (uint32_t)F.nt));
        for (int64_t g = 0; g < srow; ++g)
            p.data[1][(size_t)(n * srow + g)] =
                (uint8_t)(std::getenv("W3_SC") && std::atoi(std::getenv("W3_SC")) == 2 ? 0x38 :
                          std::getenv("W3_SC") ? 0x30 + 8 * (mix32((uint32_t)(n * srow + g) ^ (seed * 7u + 1u)) % 4u)
                                               : 0x30 + mix32((uint32_t)(n * srow + g) ^ (seed * 7u + 1u)) % 32u);
    }
    return p;
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
                e = (x & 1u) && drops ? (int32_t)((((x >> 1) % 32u) * 13u) % (uint32_t)ne)
                                       : (int32_t)((x >> 1) % (uint32_t)ne);
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

/* ------------------------------------------------------------------ one projection's device state */
struct Proj {
    const Fmt* F = nullptr;
    int64_t N = 0, K = 0, ne = 0, top_k = 8;
    bool sorted_op = false;               /* the down projection reads sorted activations */
    int64_t nec[4] = {0, 0, 0, 0};
    void* d_w[4] = {}; void* d_s[4] = {};
    size_t wbytes_expert = 0, sbytes_expert = 0;
    const RadKernelInfo* row = nullptr;
};

/* Experts e = 4x + q; class q holds ceil((ne - q) / 4). `base` supplies the distinct stored experts
 * (cycled over x), so a 256-expert layer costs a few host relayouts. */
static bool build_proj(const Plugin& pl, Proj& P, const Fmt& F, int64_t N, int64_t K, int64_t ne,
                       bool sorted_op, const std::vector<Planes>& base,
                       std::vector<std::vector<uint8_t>>* hostw = nullptr,
                       std::vector<std::vector<uint8_t>>* hosts = nullptr) {
    P.F = &F; P.N = N; P.K = K; P.ne = ne; P.sorted_op = sorted_op;
    P.row = find(pl, "moe_gemm_w4a8_mod4");
    if (!P.row) return false;
    const std::vector<RadParam> lp = {
        PI("N", N), PI("K", K), PI("N_1", N), PI("K_1", K), PI("N_2", N), PI("K_2", K),
        PI("N_3", N), PI("K_3", K), PI("group", 128), PS("dtype", F.name),
    };
    std::vector<std::vector<uint8_t>> w, s;
    for (size_t b = 0; b < base.size(); ++b) {
        std::vector<uint8_t> wb, sb;
        const int rw = stored(P.row, lp, 2, base[b], { "codes", "table" }, &wb);
        const int rs = stored(P.row, lp, 3, base[b], { "scale", "scale.1" }, &sb);
        if (rw != RAD_OK || rs != RAD_OK) {
            std::printf("  %s relayout rc %d/%d\n", F.name, rw, rs);
            return false;
        }
        w.push_back(wb);
        s.push_back(sb);
    }
    P.wbytes_expert = w[0].size();
    P.sbytes_expert = s[0].size();
    if (hostw) *hostw = w;
    if (hosts) *hosts = s;
    for (int q = 0; q < 4; ++q) {
        P.nec[q] = (ne + 3 - q) / 4;
        std::vector<uint8_t> wq, sq;
        wq.reserve((size_t)P.nec[q] * P.wbytes_expert);
        sq.reserve((size_t)P.nec[q] * P.sbytes_expert);
        for (int64_t x = 0; x < P.nec[q]; ++x) {
            const size_t b = (size_t)((4 * x + q) % (int64_t)base.size());
            wq.insert(wq.end(), w[b].begin(), w[b].end());
            sq.insert(sq.end(), s[b].begin(), s[b].end());
        }
        P.d_w[q] = dup_to_dev(wq.data(), wq.size());
        P.d_s[q] = dup_to_dev(sq.data(), sq.size());
    }
    return true;
}
static void free_proj(Proj& P) {
    for (int q = 0; q < 4; ++q) { if (P.d_w[q]) hipFree(P.d_w[q]); if (P.d_s[q]) hipFree(P.d_s[q]); P.d_w[q] = P.d_s[q] = nullptr; }
}

struct Call {
    std::vector<RadTensor> ts;
    std::vector<RadParam> ps;
    RadArgs a;
    void* d_a = nullptr; void* d_as = nullptr; void* d_sorted = nullptr; void* d_off = nullptr;
    void* d_y = nullptr; void* d_map = nullptr;
    int64_t Tr = 0, M = 0, active = 0;
};
static void make_call(const Proj& P, int64_t M, const std::vector<int32_t>& off,
                      const std::vector<int32_t>& sorted, const std::vector<uint8_t>& aq,
                      const std::vector<float>& as, Call& c, bool sorted_op) {
    const int64_t Tr = M * P.top_k, kb = P.K / 128, rows = sorted_op ? Tr : M;
    c.Tr = Tr; c.M = M;
    c.d_a = dup_to_dev(aq.data(), aq.size());
    c.d_as = dup_to_dev(as.data(), as.size() * 4);
    c.d_sorted = dup_to_dev(sorted.data(), sorted.size() * 4);
    c.d_off = dup_to_dev(off.data(), off.size() * 4);
    c.d_y = dalloc((size_t)(Tr * P.N * 2));
    const int64_t cap = P.ne + Tr / 16 + 4;
    c.d_map = dalloc((size_t)((1 + 2 * cap) * 4 + 4096));
    c.active = 0;
    for (int64_t e = 0; e < P.ne; ++e) c.active += off[(size_t)(e + 1)] > off[(size_t)e];
    auto codes_t = [&](int q) { return T(P.d_w[q], P.F->served, { P.nec[q], P.N, P.K }); };
    auto scale_t = [&](int q) { return T(P.d_s[q], RAD_F8E4M3, { P.nec[q], P.N, 2 * kb }); };
    c.ts = { T(c.d_a, RAD_F8E4M3, { rows, P.K }), T(c.d_as, RAD_F32, { rows, kb }), codes_t(0), scale_t(0),
             T(c.d_sorted, RAD_I32, { Tr }), T(c.d_off, RAD_I32, { P.ne + 1 }), codes_t(1), scale_t(1),
             codes_t(2), scale_t(2), codes_t(3), scale_t(3), T(c.d_y, RAD_BF16, { Tr, P.N }) };
    c.ps = { PI("M", M), PI("N", P.N), PI("K", P.K), PI("N_1", P.N), PI("K_1", P.K), PI("N_2", P.N),
             PI("K_2", P.K), PI("N_3", P.N), PI("K_3", P.K), PI("parts", 1), PI("n_expert", P.ne),
             PI("top_k", P.top_k), PI("group", 128), PS("a_order", sorted_op ? "sorted" : "token"),
             PS("dtype", P.F->name) };
    std::memset(&c.a, 0, sizeof c.a);
    c.a.t = c.ts.data(); c.a.n_t = (int)c.ts.size();
    c.a.p = c.ps.data(); c.a.n_p = (int)c.ps.size();
    c.a.scratch = c.d_map; c.a.scratch_bytes = (1 + 2 * cap) * 4 + 4096;
}
static void free_call(Call& c) {
    for (void* p : { c.d_a, c.d_as, c.d_sorted, c.d_off, c.d_y, c.d_map }) if (p) hipFree(p);
    c = Call{};
}

/* ------------------------------------------------------------------ the host model */
/* Activations whose products with every table level, summed over 64, are exact in f32: E4M3 bytes
 * with exponent field 6..9 (0.5 .. 7.5), any sign and mantissa. */
static void draw_act(int64_t rows, int64_t K, uint32_t seed, std::vector<uint8_t>* aq,
                     std::vector<float>* as) {
    const int64_t kb = K / 128;
    aq->resize((size_t)(rows * K));
    as->resize((size_t)(rows * kb));
    for (int64_t m = 0; m < rows; ++m) {
        for (int64_t g = 0; g < kb; ++g)
            (*as)[(size_t)(m * kb + g)] = std::getenv("W3_AS") ? 0.0625f
                                          : 0.011f + 0.003f * (float)g + 0.0007f * (float)(m % 11);
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t x = mix32((uint32_t)(m * K + k) ^ seed);
            uint32_t ex = 6 + (x >> 8) % 4, mn = (x >> 4) & 7u;
            static const int act = std::getenv("W3_ACT") ? std::atoi(std::getenv("W3_ACT")) : 0;
            if (act >= 1) ex = 7;
            if (act >= 2) mn = 0;
            (*aq)[(size_t)(m * K + k)] = (uint8_t)(((x & 1u) ? 0x80u : 0u) | (ex << 3) | mn);
        }
    }
}

/* The model of one output element the way moe_gemm_q_kernel forms it. `pl` is the expert's
 * canonical planes (levels by code, scale bytes by (row, 64-group)). */
static float model_elem(const Fmt& F, const Planes& pl, int64_t e, int64_t n, int64_t K,
                        const uint8_t* arow, const float* asrow, float* sumabs = nullptr) {
    const int64_t nkb = K / 128, cb = n / 128;
    const int64_t rot = nkb % 2 == 0 ? 2 * ((e + cb) % (nkb / 2)) : (e + cb) % nkb;
    const int64_t crow = pl.row_bytes(0), srow = pl.row_bytes(1);
    float out = 0.0f, sa = 0.0f;
    for (int64_t ps = 0; ps < nkb; ++ps) {
        const int64_t kb = (ps + rot) % nkb;
        float v = 0.0f;
        for (int q = 0; q < 2; ++q) {
            double raw = 0.0;
            for (int64_t j = 0; j < 64; ++j) {
                const int64_t k = kb * 128 + q * 64 + j;
                const int code = (int)rad_load_bits(pl.data[0].data() + n * crow, F.bits, k);
                raw += (double)F.table[code] * (double)h_e4m3(arow[k]);
            }
            const float rawf = (float)raw * (1.0f / 65536.0f);
            const float wq = h_e4m3(pl.data[1][(size_t)(n * srow + kb * 2 + q)]) * (kW4G64Scale * 65536.0f);
            if (q == 0) v = rawf * wq;
            else v = (g_fma & 1) ? std::fma(rawf, wq, v) : v + rawf * wq;
        }
        out = (g_fma & 2) ? std::fma(v, asrow[kb], out) : out + v * asrow[kb];
        sa += std::fabs(v * asrow[kb]);
    }
    if (sumabs) *sumabs = sa;
    return out;
}

static double model_exact(const Fmt& F, const Planes& pl, int64_t n, int64_t K, const uint8_t* arow,
                          const float* asrow) {
    const int64_t crow = pl.row_bytes(0), srow = pl.row_bytes(1);
    double out = 0.0;
    for (int64_t kb = 0; kb < K / 128; ++kb) {
        double v = 0.0;
        for (int q = 0; q < 2; ++q) {
            double raw = 0.0;
            for (int64_t j = 0; j < 64; ++j) {
                const int64_t k = kb * 128 + q * 64 + j;
                raw += (double)F.table[(int)rad_load_bits(pl.data[0].data() + n * crow, F.bits, k)] *
                       (double)h_e4m3(arow[k]);
            }
            v += raw * (double)h_e4m3(pl.data[1][(size_t)(n * srow + kb * 2 + q)]) * (double)kW4G64Scale;
        }
        out += v * (double)asrow[kb];
    }
    return out;
}

static std::vector<uint16_t> run_launch(const Plugin& pl, Call& c, size_t elems, int form, int fk,
                                        int ddec, int dtile, const RadKernelInfo* row, int* rc) {
    std::vector<uint16_t> junk(elems, 0x7f7fu);
    HCK(hipMemcpy(c.d_y, junk.data(), elems * 2, hipMemcpyHostToDevice));
    if (pl.pin_moe) pl.pin_moe(form);
    if (pl.pin_fk) pl.pin_fk(fk);
    if (pl.pin_ddec) pl.pin_ddec(ddec);
    if (pl.pin_dtile) pl.pin_dtile(dtile);
    *rc = row->launch(&c.a, nullptr);
    HCK(hipStreamSynchronize(nullptr));
    if (pl.pin_moe) pl.pin_moe(0);
    if (pl.pin_fk) pl.pin_fk(0);
    if (pl.pin_ddec) pl.pin_ddec(0);
    if (pl.pin_dtile) pl.pin_dtile(0);
    std::vector<uint16_t> got(elems);
    HCK(hipMemcpy(got.data(), c.d_y, elems * 2, hipMemcpyDeviceToHost));
    return got;
}

static void check_case(const Plugin& pl, const Fmt& F, const char* label, int64_t N, int64_t K,
                       bool sorted_op, const std::vector<int64_t>& Ms, const std::vector<std::string>& real) {
    const int64_t ne = 16, top_k = 8;
    std::vector<Planes> base;
    for (int b = 0; b < 8; ++b)
        base.push_back(make_planes(F, N, K, 0x1000u + (uint32_t)b,
                                   real.empty() ? "" : real[(size_t)b % real.size()]));
    Proj P;
    if (!build_proj(pl, P, F, N, K, ne, sorted_op, base)) { ok(std::string(F.name) + " " + label, false, "build"); return; }
    const RadKernelInfo* r = P.row;
    for (int64_t M : Ms) {
        const int64_t Tr = M * top_k, rows = sorted_op ? Tr : M;
        std::vector<int32_t> off, sorted;
        route(M, top_k, ne, 0x9E3779B97F4A7C15ull ^ (uint64_t)Tr, true, off, sorted);
        const int64_t live = off[(size_t)ne];
        std::vector<uint8_t> aq;
        std::vector<float> as;
        draw_act(rows, K, 0xABCDu + (uint32_t)M, &aq, &as);
        /* the model */
        std::vector<float> ref((size_t)(Tr * N), 0.0f), sab((size_t)(Tr * N), 0.0f);
        const int64_t kb = K / 128;
        for (int64_t i = 0; i < live; ++i) {
            if (sorted[(size_t)i] < 0) continue;
            int64_t e = 0;
            while (e + 1 < ne && off[(size_t)(e + 1)] <= i) ++e;
            const int64_t arow = sorted_op ? i : sorted[(size_t)i] / top_k;
            const Planes& pq = base[(size_t)(e % (int64_t)base.size())];
            for (int64_t n = 0; n < N; ++n)
                ref[(size_t)(i * N + n)] = model_elem(F, pq, e, n, K, aq.data() + arow * K,
                                                      as.data() + arow * kb, &sab[(size_t)(i * N + n)]);
        }
        Call c;
        make_call(P, M, off, sorted, aq, as, c, sorted_op);
        struct V { const char* nm; int form, fk, ddec, dtile; };
        std::vector<V> vs = { { "rule", 0, 0, 0, 0 }, { "aligned prefill", 1, 1, -1, 1 },
                              { "aligned prefill, wide down tile", 1, 1, -1, 2 },
                              { "decode (tile,run) 1 block/pass", 2, 1, -1, 0 },
                              { "decode (tile,run) 2 blocks/pass", 2, 2, -1, 0 },
                              { "uniform prefill", 3, 1, -1, 0 }, { "decode kernel G=1", 2, 1, 1, 0 },
                              { "decode kernel G=4", 2, 1, 4, 0 } };
        std::vector<uint16_t> form0;
        for (const V& v : vs) {
            if (v.ddec > 0 && !sorted_op) continue;
            if (v.dtile == 2 && !sorted_op) continue;
            if (v.fk == 2 && K % 256) continue;
            int rc = 0;
            const std::vector<uint16_t> got = run_launch(pl, c, (size_t)(Tr * N), v.form, v.fk, v.ddec, v.dtile, r, &rc);
            const std::string what = std::string(F.name) + " " + label + " T=" + std::to_string(Tr) + " " + v.nm;
            if (rc != RAD_OK) {
                /* a pin the shape cannot take is refused by the launcher, not wrong */
                std::printf("  skip  %s -- launch rc=%d (the shape cannot take this form)\n", what.c_str(), rc);
                continue;
            }
            if (std::getenv("W3_TIES")) {
                int64_t cnt[2][2][2] = {};   /* [sign][parity of the truncated value][chose up] */
                for (int64_t i = 0; i < Tr * N; ++i) {
                    uint32_t u; std::memcpy(&u, &ref[(size_t)i], 4);
                    if ((u & 0xffffu) != 0x8000u) continue;
                    const uint16_t lo = (uint16_t)(u >> 16);
                    const int up = got[(size_t)i] == (uint16_t)(lo + 1) ? 1 : got[(size_t)i] == lo ? 0 : -1;
                    if (up >= 0) ++cnt[u >> 31][lo & 1][up];
                }
                std::printf("    ties [sign][trunc parity] down/up: +even %lld/%lld +odd %lld/%lld -even %lld/%lld -odd %lld/%lld\n",
                            (long long)cnt[0][0][0], (long long)cnt[0][0][1], (long long)cnt[0][1][0], (long long)cnt[0][1][1],
                            (long long)cnt[1][0][0], (long long)cnt[1][0][1], (long long)cnt[1][1][0], (long long)cnt[1][1][1]);
            }
            /* The forms against each other, bit for bit; the first launch is the reference form. */
            if (form0.empty()) form0 = got;
            int64_t fdiff = 0;
            for (size_t x = 0; x < got.size(); ++x) fdiff += got[x] != form0[x];
            /* Against the host model: exact where the arithmetic is, else within a bf16 ulp of the
             * value plus f32 noise on the block terms' magnitude (the matrix unit's sums are not
             * bit-exact even on exactly representable data; see the report). */
            int64_t exact = 0, tolok = 0, out = 0;
            double worst = 0.0;
            for (int64_t x = 0; x < Tr * N; ++x) {
                const float want = ref[(size_t)x], g = h_bf16(got[(size_t)x]);
                if (got[(size_t)x] == h_to_bf16(want)) { ++exact; ++tolok; continue; }
                const double err = std::fabs((double)g - (double)want);
                const double tol = (1.0 / 128.0) * std::fabs((double)want) + 8.0 * 5.96e-8 * sab[(size_t)x];
                worst = std::max(worst, tol > 0 ? err / tol : 1e9);
                if (err <= tol) ++tolok; else ++out;
            }
            ok(what + " (forms agree bitwise)", fdiff == 0, std::to_string(fdiff) + " of " + std::to_string(Tr * N) + " differ from the first form");
            ok(what, out == 0,
               std::to_string(exact) + "/" + std::to_string(Tr * N) + " bit-exact vs model, " +
                   std::to_string(tolok - exact) + " within 1 bf16 ulp + f32 noise, " +
                   std::to_string(out) + " outside (worst err/tol " + std::to_string(worst) + ")");
        }
        free_call(c);
    }
    free_proj(P);
}

/* ------------------------------------------------------------------ timing */
static double time_calls(const Plugin& pl, std::vector<Call>& cs, const RadKernelInfo* row, int reps,
                         int form = 0) {
    if (pl.pin_moe) pl.pin_moe(form);
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
    if (pl.pin_moe) pl.pin_moe(0);
    return (double)ms / reps;
}

struct Shape { const char* name; int64_t N, K; bool sorted_op; };
static const Shape kGU = { "gate_up", 1024, 2048, false };
static const Shape kDN = { "down", 2048, 512, true };

static void bench(const Plugin& pl, int reps) {
    const int64_t ne = 256, top_k = 8;
    const int64_t Ms[] = { 1, 2, 4, 8, 32, 128, 512 };
    const Fmt* fmts[] = { &kW3, &kW4, &kW5 };
    const int L = 3, R = 8;
    std::printf("bench: %d layers x %d routings cycled, %d launches a point\n", L, R, reps);
    std::printf("%-10s %-8s %5s %9s %9s %10s %9s\n", "format", "proj", "T", "us", "GB/s", "bytes/exp", "active");
    double dec_us[3][2] = {};
    for (int fi = 0; fi < 3; ++fi) {
        const Fmt& F = *fmts[fi];
        for (int si = 0; si < 2; ++si) {
            const Shape& S = si ? kDN : kGU;
            std::vector<Planes> base;
            for (int b = 0; b < 8; ++b) base.push_back(make_planes(F, S.N, S.K, 0x77u + (uint32_t)b));
            std::vector<Proj> layers(L);
            for (int l = 0; l < L; ++l)
                if (!build_proj(pl, layers[(size_t)l], F, S.N, S.K, ne, S.sorted_op, base)) return;
            for (int64_t M : Ms) {
                const int64_t Tr = M * top_k, rows = S.sorted_op ? Tr : M;
                std::vector<uint8_t> aq;
                std::vector<float> as;
                draw_act(rows, S.K, 0x55u, &aq, &as);
                std::vector<Call> cs((size_t)(L * R));
                double act = 0;
                for (int l = 0; l < L; ++l)
                    for (int r = 0; r < R; ++r) {
                        std::vector<int32_t> off, sorted;
                        route(M, top_k, ne, 0x1234567ull * (uint64_t)(r + 1) + (uint64_t)Tr, false, off, sorted);
                        make_call(layers[(size_t)l], M, off, sorted, aq, as, cs[(size_t)(l * R + r)], S.sorted_op);
                        act += (double)cs[(size_t)(l * R + r)].active;
                    }
                act /= (double)(L * R);
                const double ms = time_calls(pl, cs, layers[0].row, reps);
                const double bytes_e = (double)(layers[0].wbytes_expert + layers[0].sbytes_expert);
                std::printf("%-10s %-8s %5lld %9.1f %9.1f %10.0f %9.1f\n", F.name, S.name, (long long)Tr,
                            ms * 1e3, act * bytes_e / (ms * 1e-3) / 1e9, bytes_e, act);
                if (M == 1) dec_us[fi][si] = ms * 1e3;
                for (Call& c : cs) free_call(c);
            }
            for (Proj& p : layers) free_proj(p);
        }
    }
    std::printf("\nexpert GEMMs only, one token (T=8), 40 layers: tokens/s if nothing else cost anything\n");
    for (int fi = 0; fi < 3; ++fi)
        std::printf("  %-10s gate_up %.1f us + down %.1f us per layer -> %.1f tok/s\n", fmts[fi]->name,
                    dec_us[fi][0], dec_us[fi][1], 1e6 / (40.0 * (dec_us[fi][0] + dec_us[fi][1])));
}

/* Decode, repeated: the three formats' weights all resident, and the timing trials interleaved
 * (format A, B, C, A, B, C ...) so a clock or a neighbour's burst moves all three together. Reports
 * the median, minimum and spread of each format's per-launch time over `trials` trials. */
static void decode(const Plugin& pl, int trials, int reps) {
    const int64_t ne = 256, top_k = 8;
    const Fmt* fmts[] = { &kW3, &kW4, &kW5 };
    const int L = 3, R = 8;
    const int64_t Ms[] = { 1, 2, 4 };
    struct Cell { std::vector<Proj> layers; std::vector<Call> cs; };
    for (int si = 0; si < 2; ++si) {
        const Shape& S = si ? kDN : kGU;
        for (int64_t M : Ms) {
            const int64_t Tr = M * top_k, rows = S.sorted_op ? Tr : M;
            std::vector<uint8_t> aq;
            std::vector<float> as;
            draw_act(rows, S.K, 0x55u, &aq, &as);
            Cell cell[3];
            for (int fi = 0; fi < 3; ++fi) {
                std::vector<Planes> base;
                for (int b = 0; b < 8; ++b) base.push_back(make_planes(*fmts[fi], S.N, S.K, 0x77u + (uint32_t)b));
                cell[fi].layers.resize(L);
                for (int l = 0; l < L; ++l) build_proj(pl, cell[fi].layers[(size_t)l], *fmts[fi], S.N, S.K, ne, S.sorted_op, base);
                cell[fi].cs.resize((size_t)(L * R));
                for (int l = 0; l < L; ++l)
                    for (int r = 0; r < R; ++r) {
                        std::vector<int32_t> off, sorted;
                        route(M, top_k, ne, 0x1234567ull * (uint64_t)(r + 1) + (uint64_t)Tr, false, off, sorted);
                        make_call(cell[fi].layers[(size_t)l], M, off, sorted, aq, as, cell[fi].cs[(size_t)(l * R + r)], S.sorted_op);
                    }
            }
            std::vector<double> t[3];
            for (int tr = 0; tr < trials; ++tr)
                for (int fi = 0; fi < 3; ++fi)
                    t[fi].push_back(time_calls(pl, cell[fi].cs, cell[fi].layers[0].row, reps) * 1e3);
            std::printf("%-8s T=%-3lld", S.name, (long long)Tr);
            for (int fi = 0; fi < 3; ++fi) {
                std::sort(t[fi].begin(), t[fi].end());
                std::printf("  %s median %6.1f min %6.1f max %6.1f us |", fmts[fi]->name, t[fi][t[fi].size() / 2], t[fi].front(), t[fi].back());
            }
            std::printf("  w3/w4 median %.3f\n", t[0][t[0].size() / 2] / t[1][t[1].size() / 2]);
            std::fflush(stdout);
            for (int fi = 0; fi < 3; ++fi) {
                for (Call& c : cell[fi].cs) free_call(c);
                for (Proj& p : cell[fi].layers) free_proj(p);
            }
        }
    }
}

/* ------------------------------------------------------------------ the layout alone */
/* The weight read the way the kernel's threads read it, nothing else: a block of 128 threads takes 64
 * rows, four threads a row (one 32-element segment each), the K loop in the kernel's order (start
 * rotated by block), every pass's loads issued before the xor that keeps them alive. */
template <int MODE>   /* 0 w3 planar, 1 w3 interleaved, 2 w4, 3 w5 */
__global__ __launch_bounds__(128) void readpat_kernel(const uint8_t* __restrict__ W, uint64_t expert_bytes,
                                                      uint32_t N, uint32_t K, uint32_t* sink) {
    const uint32_t tid = threadIdx.x, sn = tid / 4, sk = (tid % 4) * 32u;
    const uint32_t tiles = N / 64, e = blockIdx.x / tiles, row0 = (blockIdx.x % tiles) * 64u;
    const uint32_t nkb = K / 128, rowb = MODE <= 1 ? K / 8 * 3 : MODE == 2 ? K / 2 : K / 8 * 5;
    const uint8_t* base = W + (uint64_t)e * expert_bytes;
    uint32_t acc = 0;
    for (uint32_t ps = 0; ps < nkb; ++ps) {
        const uint32_t cc = ((ps + blockIdx.x) % nkb) * 128u;
#pragma unroll
        for (uint32_t p = 0; p < 2; ++p) {
            const uint8_t* wr = base + (uint64_t)(row0 + sn + 32u * p) * rowb;
            if (MODE == 0) {
                const uint2 a = *(const uint2*)(wr + ((cc + sk) >> 2));
                const uint32_t b = *(const uint32_t*)(wr + (K >> 2) + ((cc + sk) >> 3));
                acc ^= a.x ^ a.y ^ b;
            } else if (MODE == 1) {
                const uint32_t blk = (cc >> 7) * 48u, s = sk >> 5;
                const uint2 a = *(const uint2*)(wr + blk + 8u * s);
                const uint32_t b = *(const uint32_t*)(wr + blk + 32u + 4u * s);
                acc ^= a.x ^ a.y ^ b;
            } else if (MODE == 2) {
                const uint4 a = *(const uint4*)(wr + ((cc + sk) >> 1));
                acc ^= a.x ^ a.y ^ a.z ^ a.w;
            } else {
                const uint4 a = *(const uint4*)(wr + ((cc + sk) >> 1));
                const uint32_t b = *(const uint32_t*)(wr + (K >> 1) + ((cc + sk) >> 3));
                acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b;
            }
        }
    }
    if (acc == 0x9E3779B9u) sink[0] = acc;
}
template <int MODE>
static double readpat_run(const uint8_t* W, uint64_t expert_bytes, int ne, uint32_t N, uint32_t K,
                          uint32_t* sink, int reps) {
    const dim3 grid((N / 64) * (uint32_t)ne);
    readpat_kernel<MODE><<<grid, 128>>>(W, expert_bytes, N, K, sink);
    HCK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
    HCK(hipEventRecord(e0, nullptr));
    for (int i = 0; i < reps; ++i) readpat_kernel<MODE><<<grid, 128>>>(W, expert_bytes, N, K, sink);
    HCK(hipEventRecord(e1, nullptr));
    HCK(hipEventSynchronize(e1));
    float ms = 0;
    HCK(hipEventElapsedTime(&ms, e0, e1));
    return (double)ms / reps;
}
static void readpat(int reps) {
    std::printf("readpat: the weight planes read in the kernel's per-thread pattern, no arithmetic\n");
    std::printf("%-26s %6s %6s %9s %9s %9s\n", "pattern", "N", "K", "MB read", "us", "GB/s");
    uint32_t* sink = (uint32_t*)dalloc(64);
    struct S { uint32_t N, K; int ne; } shapes[] = { { 1024, 2048, 64 }, { 2048, 512, 128 } };
    for (const S& s : shapes) {
        struct M { const char* name; int mode; } modes[] = { { "w4nl64 (16 B)", 2 }, { "w5nl64 (16 B + 4 B)", 3 },
                                                        { "w3 planar (8 B + 4 B)", 0 }, { "w3 interleaved 48 B", 1 } };
        for (const M& m : modes) {
            const uint64_t rowb = m.mode <= 1 ? s.K / 8 * 3 : m.mode == 2 ? s.K / 2 : s.K / 8 * 5;
            const uint64_t eb = rowb * s.N;
            uint8_t* W = (uint8_t*)dalloc(eb * (uint64_t)s.ne);
            HCK(hipMemset(W, 0x5A, eb * (uint64_t)s.ne));
            double ms = 0;
            switch (m.mode) {
                case 0: ms = readpat_run<0>(W, eb, s.ne, s.N, s.K, sink, reps); break;
                case 1: ms = readpat_run<1>(W, eb, s.ne, s.N, s.K, sink, reps); break;
                case 2: ms = readpat_run<2>(W, eb, s.ne, s.N, s.K, sink, reps); break;
                default: ms = readpat_run<3>(W, eb, s.ne, s.N, s.K, sink, reps); break;
            }
            const double mb = (double)(eb * (uint64_t)s.ne) / 1e6;
            std::printf("%-26s %6u %6u %9.1f %9.1f %9.1f\n", m.name, s.N, s.K, mb, ms * 1e3, mb / ms);
            hipFree(W);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: halo_w3_bench <libhalo.so> check|bench|readpat ...\n"); return 2; }
    Plugin pl;
    if (!load_plugin(&pl, argv[1])) return 2;
    const std::string mode = argv[2];
    if (const char* f = std::getenv("W3_FMA")) g_fma = std::atoi(f);
    std::printf("HALO_W3_IL=%s\n", std::getenv("HALO_W3_IL") ? std::getenv("HALO_W3_IL") : "(unset)");
    if (mode == "check") {
        std::vector<std::string> gu, dn;
        if (argc > 4 && std::string(argv[3]) == "real") {
            for (int i = 4; i < argc; ++i) {
                const std::string s = argv[i];
                (s.find("gate_up") != std::string::npos ? gu : dn).push_back(s);
            }
        }
        const std::vector<int64_t> Ms = { 1, 5, 40, 80 };
        for (const Fmt* F : { &kW3, &kW4, &kW5 }) {
            if (!gu.empty() && F != &kW3) continue;     /* real planes are w3's */
            check_case(pl, *F, gu.empty() ? "gate_up random" : "gate_up real", kGU.N, kGU.K, false, Ms, gu);
            check_case(pl, *F, dn.empty() ? "down random" : "down real", kDN.N, kDN.K, true, Ms, dn);
        }
        std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "all passed", g_fail);
        return g_fail ? 1 : 0;
    }
    if (mode == "decode") { decode(pl, argc > 3 ? std::atoi(argv[3]) : 9, argc > 4 ? std::atoi(argv[4]) : 200); return 0; }
    if (mode == "bench") { bench(pl, argc > 3 ? std::atoi(argv[3]) : 96); return 0; }
    if (mode == "readpat") { readpat(argc > 3 ? std::atoi(argv[3]) : 20); return 0; }
    return 2;
}
