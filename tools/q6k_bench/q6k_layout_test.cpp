/* q6k_layout_test -- the Q6_K stored-form hooks (libhalo/halo_q6k_layout.cpp) on the host, with no
 * ROCm: guard bytes, the tile formula, the exact inverse, and every shape or encoding the format
 * cannot address. Pure host C++; the implementation is included so the test needs neither a device
 * nor a built plugin.
 *
 *   clang++ -std=c++17 -O2 -I abi -I libhalo tools/q6k_bench/q6k_layout_test.cpp -o q6k_layout_test
 */
#include "halo_q6k_layout.cpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_ok = 0, g_fail = 0;
static void ok(const std::string& what, bool c, const std::string& detail = "") {
    if (!c) ++g_fail; else ++g_ok;
    std::printf("  %s  %s%s%s\n", c ? "ok  " : "FAIL", what.c_str(),
                detail.empty() ? "" : " -- ", detail.c_str());
}

static RadParam PI(const char* k, long long v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k; p.kind = RAD_P_INT; p.ival = v;
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
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = data; t.dtype = dt; t.rank = 2; t.shape[0] = r; t.shape[1] = c;
    return t;
}
static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

struct Planes {
    int64_t N, K, KB, KG;
    std::vector<uint8_t> codes;    /* canonical i6 row-major */
    std::vector<int8_t> scale;
    std::vector<uint16_t> d;
    Planes(int64_t n, int64_t k, uint32_t seed) : N(n), K(k), KB(k / 16), KG(k / 256) {
        codes.resize((size_t)(n * (k * 6 / 8)));
        scale.resize((size_t)(n * KB));
        d.resize((size_t)(n * KG));
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = (uint8_t)mix32((uint32_t)i ^ seed);
        for (size_t i = 0; i < scale.size(); ++i) scale[i] = (int8_t)mix32((uint32_t)i * 3u ^ (seed + 11u));
        for (size_t i = 0; i < d.size(); ++i) {
            const uint32_t r = mix32((uint32_t)i * 2654435761u ^ (seed + 77u));
            d[i] = (uint16_t)(((r & 1u) << 15) | ((6u + (r >> 1) % 15u) << 10) | ((r >> 8) & 0x3FFu));
        }
    }
};

static const RadKernelInfo* no_row = nullptr;   /* the hooks are called directly */

static int call_layout(const int64_t N, const int64_t K, int operand, const RadEncoding& enc,
                       int sel, const RadTensor& t, RadLayout* L, const char* group = "256") {
    const RadParam lp[] = { PI("N", N), PI("K", K), PI(group, atoll(group)) };
    const int s[1] = { sel };
    return halo_rad_layout_q6k(lp, 3, operand, &enc, s, &t, 1, L);
}
static int call_relayout(const int64_t N, const int64_t K, int operand, const RadEncoding& enc,
                         int sel, const RadTensor& t, void* dst, int64_t bytes) {
    const RadParam lp[] = { PI("N", N), PI("K", K), PI("group", 256) };
    const int s[1] = { sel };
    return halo_rad_relayout_q6k(lp, 3, operand, &enc, s, &t, 1, dst, bytes);
}
static int call_unrelayout(const int64_t N, const int64_t K, int operand, const RadEncoding& enc,
                           int sel, const void* src, int64_t bytes, const RadTensor& t) {
    const RadParam lp[] = { PI("N", N), PI("K", K), PI("group", 256) };
    const int s[1] = { sel };
    return halo_rad_unrelayout_q6k(lp, 3, operand, &enc, s, src, bytes, &t, 1);
}

/* The three weight operands: codes=2 (plane 0), scale=3 (plane 1), scale.1=6 (plane 2). */
struct Op { int operand, sel; const char* tag; uint32_t dt; };

static void test_shapes() {
    std::printf("[shapes]\n");
    const RadEncoding enc = q6k_enc();
    const Op ops[] = {
        { 2, 0, "halo.q6k.w.tile16x64", RAD_I6 },
        { 3, 1, "halo.q6k.s.tile16x16", RAD_I8 },
        { 6, 2, "halo.q6k.d.tile16x256", RAD_F16 },
    };
    const int64_t shapes[][2] = { {16, 256}, {16, 512}, {32, 768}, {48, 1024}, {64, 2048}, {128, 256} };
    for (const auto& sh : shapes) {
        const int64_t N = sh[0], K = sh[1];
        Planes p(N, K, 1234u + (uint32_t)N);
        bool good = true;
        std::string why;
        for (const Op& o : ops) {
            RadTensor t = o.operand == 2 ? planeT(p.codes.data(), RAD_I6, N, K)
                        : o.operand == 3 ? planeT(p.scale.data(), RAD_I8, N, K / 16)
                                         : planeT(p.d.data(), RAD_F16, N, K / 256);
            RadLayout L;
            std::memset(&L, 0, sizeof L);
            const int rl = call_layout(N, K, o.operand, enc, o.sel, t, &L);
            const int64_t want = o.operand == 2 ? N * (K * 6 / 8)
                               : o.operand == 3 ? N * (K / 16)
                                                : N * (K / 256) * 2;
            if (rl != RAD_OK || L.bytes != want || L.dtype != o.dt || std::strcmp(L.tag, o.tag)) {
                good = false;
                why = std::string("layout rc=") + std::to_string(rl) + " bytes=" +
                      std::to_string(L.bytes) + " want=" + std::to_string(want);
                break;
            }
            const int64_t G = 32;
            std::vector<uint8_t> a((size_t)(L.bytes + 2 * G), 0x00);
            std::vector<uint8_t> b((size_t)(L.bytes + 2 * G), 0xFF);
            if (call_relayout(N, K, o.operand, enc, o.sel, t, a.data() + G, L.bytes) != RAD_OK ||
                call_relayout(N, K, o.operand, enc, o.sel, t, b.data() + G, L.bytes) != RAD_OK) {
                good = false; why = "relayout rc"; break;
            }
            for (int64_t i = 0; i < G; ++i)
                if (a[(size_t)i] != 0x00 || a[(size_t)(G + L.bytes + i)] != 0x00 ||
                    b[(size_t)i] != 0xFF || b[(size_t)(G + L.bytes + i)] != 0xFF) {
                    good = false; why = "guard bytes touched"; break;
                }
            if (!good) break;
            if (std::memcmp(a.data() + G, b.data() + G, (size_t)L.bytes)) {
                good = false; why = "a destination byte was not written"; break;
            }
            /* the tile formula, element by element */
            const uint8_t* st = a.data() + G;
            if (o.operand == 2) {
                for (int64_t n = 0; n < N && good; ++n)
                    for (int64_t kb = 0; kb < K / 16 && good; ++kb) {
                        const int64_t at = halo_q6k::code_off(n, kb * 16, K);
                        if (std::memcmp(st + at, p.codes.data() + n * (K * 6 / 8) + kb * 12, 12)) {
                            good = false; why = "code placement";
                        }
                    }
            } else if (o.operand == 3) {
                for (int64_t n = 0; n < N && good; ++n)
                    for (int64_t kb = 0; kb < K / 16 && good; ++kb)
                        if ((int8_t)st[halo_q6k::scale_idx(n, kb, K)] != p.scale[(size_t)(n * (K / 16) + kb)]) {
                            good = false; why = "scale placement";
                        }
            } else {
                for (int64_t n = 0; n < N && good; ++n)
                    for (int64_t g = 0; g < K / 256 && good; ++g) {
                        uint16_t h;
                        std::memcpy(&h, st + 2 * halo_q6k::d_idx(n, g, K), 2);
                        if (h != p.d[(size_t)(n * (K / 256) + g)]) { good = false; why = "d placement"; }
                    }
            }
            if (!good) break;
            /* the exact inverse, into a guard-bracketed canonical plane */
            const int64_t pb = want;
            std::vector<uint8_t> back((size_t)(pb + 2 * G), 0x5C);
            RadTensor bt = o.operand == 2 ? planeT(back.data() + G, RAD_I6, N, K)
                         : o.operand == 3 ? planeT(back.data() + G, RAD_I8, N, K / 16)
                                          : planeT(back.data() + G, RAD_F16, N, K / 256);
            if (call_unrelayout(N, K, o.operand, enc, o.sel, st, L.bytes, bt) != RAD_OK) {
                good = false; why = "unrelayout rc"; break;
            }
            const uint8_t* orig = o.operand == 2 ? p.codes.data()
                                : o.operand == 3 ? (const uint8_t*)p.scale.data()
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
    (void)no_row;
}

static void test_refusals() {
    std::printf("[refusals]\n");
    const RadEncoding enc = q6k_enc();
    struct Bad { int64_t N, K; long long group; const char* why; };
    const Bad bads[] = {
        {16, 128, 256, "K=128 (half a block)"}, {16, 384, 256, "K=384 (1.5 blocks)"},
        {16, 255, 256, "K=255"}, {16, 0, 256, "K=0"}, {16, -256, 256, "K<0"},
        {8, 256, 256, "N=8"}, {17, 256, 256, "N=17"}, {24, 256, 256, "N=24 (1.5 tiles)"},
        {0, 256, 256, "N=0"}, {16, 256, 32, "group 32"}, {16, 256, 512, "group 512"},
    };
    for (const Bad& b : bads) {
        bool good = true;
        std::string why;
        /* a plane sized for a shape the format can address, so only the hook's shape check refuses */
        const int64_t Np = 32, Kp = 512;
        Planes p(Np, Kp, 7);
        for (const int operand : {2, 3, 6}) {
            const int sel = operand == 2 ? 0 : operand == 3 ? 1 : 2;
            RadTensor t = operand == 2 ? planeT(p.codes.data(), RAD_I6, Np, Kp)
                        : operand == 3 ? planeT(p.scale.data(), RAD_I8, Np, Kp / 16)
                                       : planeT(p.d.data(), RAD_F16, Np, Kp / 256);
            RadLayout L;
            std::memset(&L, 0, sizeof L);
            const int rl = call_layout(b.N, b.K, operand, enc, sel, t, &L, std::to_string(b.group).c_str());
            const int64_t G = 32;
            std::vector<uint8_t> dst((size_t)(1024 + 2 * G), 0xA5);
            const int rr = call_relayout(b.N, b.K, operand, enc, sel, t, dst.data() + G, 1024);
            std::vector<uint8_t> plane((size_t)(1024 + 2 * G), 0x5C);
            RadTensor ot = planeT(plane.data() + G, operand == 2 ? RAD_I6 : operand == 3 ? RAD_I8 : RAD_F16, 1, 1);
            const int ru = call_unrelayout(b.N, b.K, operand, enc, sel, dst.data() + G, 1024, ot);
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

    /* a good shape: wrong destination size, wrong plane extents, wrong encodings */
    {
        const int64_t N = 32, K = 512, G = 32;
        Planes p(N, K, 9);
        RadTensor t = planeT(p.codes.data(), RAD_I6, N, K);
        RadLayout L;
        std::memset(&L, 0, sizeof L);
        call_layout(N, K, 2, enc, 0, t, &L);
        std::vector<uint8_t> dst((size_t)(L.bytes + 2 * G), 0xA5);
        const int r1 = call_relayout(N, K, 2, enc, 0, t, dst.data() + G, L.bytes - 1);
        const int r2 = call_relayout(N, K, 2, enc, 0, t, dst.data() + G, L.bytes + 1);
        bool clean = true;
        for (uint8_t v : dst) clean = clean && v == 0xA5;
        ok("destination one byte short or long is refused, nothing written",
           r1 == RAD_E_SHAPE && r2 == RAD_E_SHAPE && clean,
           "rc " + std::to_string(r1) + "/" + std::to_string(r2));
        RadTensor wrongK = planeT(p.codes.data(), RAD_I6, N, K - 16);
        RadTensor wrongN = planeT(p.codes.data(), RAD_I6, N - 16, K);
        const int r3 = call_layout(N, K, 2, enc, 0, wrongK, &L);
        const int r4 = call_layout(N, K, 2, enc, 0, wrongN, &L);
        ok("plane extents that disagree with N, K are refused", r3 == RAD_E_SHAPE && r4 == RAD_E_SHAPE,
           "rc " + std::to_string(r3) + "/" + std::to_string(r4));

        /* a scale plane of the wrong sub-block width, and a missing/other second level */
        RadEncoding e2 = enc;
        e2.plane[1].block[1] = 32;
        RadTensor st32 = planeT(p.scale.data(), RAD_I8, N, K / 32);
        RadLayout L2;
        std::memset(&L2, 0, sizeof L2);
        const int r5 = call_layout(N, K, 3, e2, 1, st32, &L2);
        RadEncoding e3 = enc;
        e3.n_planes = 2;
        const int r6 = call_layout(N, K, 2, e3, 0, t, &L2);
        RadEncoding e4 = enc;
        e4.plane[2].block[1] = 128;
        RadTensor dt = planeT(p.d.data(), RAD_F16, N, K / 128);
        const int r7 = call_layout(N, K, 6, e4, 2, dt, &L2);
        RadEncoding e5 = enc;
        e5.plane[0].dtype = RAD_U6;
        const int r8 = call_layout(N, K, 2, e5, 0, t, &L2);
        ok("a wrong sub-block, a missing scale.1, a wrong second level or a u6 code plane is refused as a dtype or shape",
           (r5 == RAD_E_SHAPE || r5 == RAD_E_DTYPE) && r6 == RAD_E_DTYPE && r7 == RAD_E_DTYPE &&
               r8 == RAD_E_DTYPE,
           "rc " + std::to_string(r5) + "/" + std::to_string(r6) + "/" + std::to_string(r7) + "/" + std::to_string(r8));
        RadLayout L3;
        std::memset(&L3, 0, sizeof L3);
        const int r9 = call_layout(N, K, 0, enc, 0, t, &L3);
        ok("an operand that is neither codes, scale nor scale.1 is unsupported", r9 == RAD_E_UNSUPPORTED,
           "rc " + std::to_string(r9));
    }
}

int main() {
    std::printf("q6k_layout_test\n");
    test_shapes();
    test_refusals();
    std::printf("\n%d ok, %d FAIL\n", g_ok, g_fail);
    return g_fail ? 1 : 0;
}
