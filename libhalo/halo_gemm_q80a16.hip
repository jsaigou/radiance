// halo_gemm_q80a16.hip - gemm_q80a16_nt_m8 and gemm_q80a16_nt_wmma: dense GEMMs against ggml Q8_0
// weights (int8 codes, one f16 scale per 32 K) with a 16-bit activation read directly.
//
//   C[M, N] = X[M, K] @ dequant(W)[N, K]^T,   dequant(W)[n, k] = f32(d[n, k/32]) * f32(q[n, k])
//
// The weight is ggml's own Q8_0 numbers, bit for bit (halo_q80.h): d*q is exact in f32, and the
// kernels never round it. The activation is NOT quantised (llama.cpp quantises it to Q8_0 for its
// dot; libhalo does not), so the arithmetic is:
//   products   q (exact in f32, bf16 and f16) times the activation (bf16 or f16): exact in f32
//   per block  S = sum over 32 k of q*x, accumulated in f32 -- in a fixed order in the matvec
//              (below), in the matrix unit's own order in the WMMA kernel
//   per row    acc = fma(d, S, acc) over the blocks, f32; one bf16 (or f32) rounding at the store
// Only that summation order can differ between the host model and the device.
//
// Stored form: halo_q80.h's 16x32 tiles (512 contiguous code bytes + 16 f16 scales per tile).
//
// ---- the matvec (M <= 8): one wave reads one tile per pass at 16 bytes a lane, lane = (row r =
// lane/2, half h = lane%2); the lane's partial for its row/half is S_h = fma chain over its 16 k
// (element order), acc = fma(d, S_h, acc). The two half-lanes close with one xor shuffle, and WK
// waves that split a row tile's K blocks (wave w takes blocks w, w+WK, ...) close in LDS in wave
// order. The activation is read from global (L1/L2, a broadcast across the 16 lanes sharing a
// half), M rows against one weight read.
//
// ---- the WMMA kernel (any M): the weight is the B operand (so one scale d per lane per block
// scales an accumulator column), the activation the A operand from an LDS tile. A block is NW waves,
// each one 16-row weight tile, sharing a 16*TT-token activation tile staged through LDS 64 k at a
// time. The weight goes from global straight to registers: a lane's 16 bytes of a tile row become a
// bf16 B fragment (int8 -> f32 -> upper half is exactly bf16 for |q| <= 128), no LDS for the weight.
// Per 32-k block: two WMMAs into a zeroed accumulator, then acc = fma(d, tmp, acc).
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_wmma.h"
#include "halo_q80.h"

#include <cstdint>
#include <cstring>

namespace {

constexpr int kU = 4;   /* matvec: tiles in flight a wave */

template <bool BF>
__device__ __forceinline__ float act_lo(uint32_t w) {
    if constexpr (BF) return __builtin_bit_cast(float, w << 16);
    else return (float)__builtin_bit_cast(__fp16, (uint16_t)(w & 0xffffu));
}
template <bool BF>
__device__ __forceinline__ float act_hi(uint32_t w) {
    if constexpr (BF) return __builtin_bit_cast(float, w & 0xffff0000u);
    else return (float)__builtin_bit_cast(__fp16, (uint16_t)(w >> 16));
}
__device__ __forceinline__ float i8_of(uint32_t w, int b) {
    return (float)((int)(w << (24 - 8 * b)) >> 24);
}
__device__ __forceinline__ float h16_f32(uint16_t h) {
    return (float)__builtin_bit_cast(__fp16, h);
}

template <bool BF, bool F32O, int WK>
__global__ void __launch_bounds__(128) q80_mv_kernel(const uint16_t* __restrict__ x,
                                                     const uint8_t* __restrict__ wq,
                                                     const uint16_t* __restrict__ ws,
                                                     void* __restrict__ y, int M, int ntiles,
                                                     long K, long a_ld, long c_ld) {
    constexpr int TB = 4 / WK;                      /* row tiles a block */
    __shared__ float red[WK > 1 ? 4 : 1][16][8];
    const int lane = (int)threadIdx.x & 31;
    const int wave = (int)threadIdx.x >> 5;
    const int ts = wave / WK, ks = wave % WK;
    const long rt = (long)blockIdx.x * TB + ts;
    const long KB = K >> 5;
    const int r = lane >> 1, h = lane & 1;
    float acc[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) acc[m] = 0.0f;
    if (rt < ntiles) {
        const uint8_t* wt = wq + rt * KB * 512 + lane * 16;
        const uint16_t* st = ws + rt * KB * 16 + r;
        const uint16_t* xh = x + h * 16;
        for (long kb = ks; kb < KB; kb += (long)WK * kU) {
            uint4 w[kU];
            uint16_t d[kU];
#pragma unroll
            for (int u = 0; u < kU; ++u) {
                const long kbu = kb + (long)u * WK;
                if (kbu < KB) {
                    w[u] = *(const uint4*)(wt + kbu * 512);
                    d[u] = st[kbu * 16];
                } else {
                    w[u] = make_uint4(0, 0, 0, 0);
                    d[u] = 0;
                }
            }
#pragma unroll
            for (int u = 0; u < kU; ++u) {
                const long kbu = kb + (long)u * WK;
                if (kbu >= KB) break;
                const float dq = h16_f32(d[u]);
                const uint32_t qw[4] = { w[u].x, w[u].y, w[u].z, w[u].w };
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    if (m < M) {
                        const uint16_t* xr = xh + (long)m * a_ld + kbu * 32;
                        const uint4 xb0 = *(const uint4*)xr;
                        const uint4 xb1 = *(const uint4*)(xr + 8);
                        const uint32_t xp[8] = { xb0.x, xb0.y, xb0.z, xb0.w,
                                                 xb1.x, xb1.y, xb1.z, xb1.w };
                        float part = 0.0f;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            part = fmaf(i8_of(qw[i], 0), act_lo<BF>(xp[2 * i]), part);
                            part = fmaf(i8_of(qw[i], 1), act_hi<BF>(xp[2 * i]), part);
                            part = fmaf(i8_of(qw[i], 2), act_lo<BF>(xp[2 * i + 1]), part);
                            part = fmaf(i8_of(qw[i], 3), act_hi<BF>(xp[2 * i + 1]), part);
                        }
                        acc[m] = fmaf(dq, part, acc[m]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) acc[m] += __shfl_xor(acc[m], 1);
    if constexpr (WK > 1) {
        if (!(lane & 1)) {
#pragma unroll
            for (int m = 0; m < 8; ++m) red[wave][r][m] = acc[m];
        }
        __syncthreads();
        if (ks != 0) return;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            float v = red[ts * WK][r][m];
            for (int k = 1; k < WK; ++k) v += red[ts * WK + k][r][m];
            acc[m] = v;
        }
    }
    if (!(lane & 1) && rt < ntiles) {
        const long n = rt * 16 + r;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m < M) {
                if constexpr (F32O) ((float*)y)[(long)m * c_ld + n] = acc[m];
                else ((uint16_t*)y)[(long)m * c_ld + n] = f32_to_bf16(acc[m]);
            }
        }
    }
}

/* ---------------------------------------------------------------- WMMA kernel */

constexpr int kLdsW = 72;   /* LDS activation row: 64 k + 8 pad (144 bytes; ds_read_b128 conflict-free) */

/* A lane's 16 int8 codes (one uint4) as a bf16 B fragment: element e is q[e]. */
__device__ __forceinline__ v16bf q_frag(uint4 w) {
    uint32_t o[8];
    const uint32_t qw[4] = { w.x, w.y, w.z, w.w };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t a = __float_as_uint(i8_of(qw[i], 0));
        const uint32_t b = __float_as_uint(i8_of(qw[i], 1));
        const uint32_t c = __float_as_uint(i8_of(qw[i], 2));
        const uint32_t dd = __float_as_uint(i8_of(qw[i], 3));
        o[2 * i] = bf16_pack_hi(a, b);
        o[2 * i + 1] = bf16_pack_hi(c, dd);
    }
    return halo::mk<v16bf>(o);
}

template <int NW, int TT, bool F32O>
__global__ void __launch_bounds__(NW * 32) q80_wmma_kernel(const uint16_t* __restrict__ x,
                                                          const uint8_t* __restrict__ wq,
                                                          const uint16_t* __restrict__ ws,
                                                          void* __restrict__ y, int M, int ntiles,
                                                          long K, long a_ld, long c_ld) {
    constexpr int BM = 16 * TT;
    constexpr int NT = NW * 32;
    constexpr int CH = (BM * 8) / NT;      /* 16-byte activation chunks a thread stages a pass */
    static_assert(CH >= 1 && (BM * 8) % NT == 0, "tile/threads mismatch");
    __shared__ __attribute__((aligned(16))) uint16_t sx[BM * kLdsW];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const long rt = (long)blockIdx.x * NW + wave;
    const long m0 = (long)blockIdx.y * BM;
    const long KB = K >> 5;
    const long npass = (KB + 1) >> 1;
    const int lr = lane & 15;

    v8f acc[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[t][e] = 0.0f;

    const bool live = rt < ntiles;
    const uint8_t* wtile = wq + (live ? rt : 0) * KB * 512 + lr * 32;
    const uint16_t* stile = ws + (live ? rt : 0) * KB * 16 + lr;

    uint4 xr[CH], xn[CH];
    uint4 wr[4], wn[4];
    uint16_t dr[2], dn[2];

    auto load_pass = [&](long p, uint4 (&xo)[CH], uint4 (&wo)[4], uint16_t (&dd)[2]) {
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            const int ch = tid + c * NT;
            const int tok = ch >> 3, kc = ch & 7;
            const long m = m0 + tok;
            const long k = p * 64 + kc * 8;
            xo[c] = make_uint4(0, 0, 0, 0);
            if (m < M && k < K) xo[c] = *(const uint4*)(x + m * a_ld + k);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const long kb = p * 2 + j;
            wo[2 * j] = make_uint4(0, 0, 0, 0);
            wo[2 * j + 1] = make_uint4(0, 0, 0, 0);
            dd[j] = 0;
            if (live && kb < KB) {
                wo[2 * j] = *(const uint4*)(wtile + kb * 512);
                wo[2 * j + 1] = *(const uint4*)(wtile + kb * 512 + 16);
                dd[j] = stile[kb * 16];
            }
        }
    };

    load_pass(0, xr, wr, dr);
    for (long p = 0; p < npass; ++p) {
        __syncthreads();                                   /* the previous pass is done with sx */
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            const int ch = tid + c * NT;
            *(uint4*)(sx + (ch >> 3) * kLdsW + (ch & 7) * 8) = xr[c];
        }
        __syncthreads();
        if (p + 1 < npass) load_pass(p + 1, xn, wn, dn);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (p * 2 + j < KB) {
                const float dq = h16_f32(dr[j]);
                const v16bf w0 = q_frag(wr[2 * j]);
                const v16bf w1 = q_frag(wr[2 * j + 1]);
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    const uint16_t* ar = sx + (t * 16 + lr) * kLdsW + j * 32;
                    v8f tmp;
#pragma unroll
                    for (int e = 0; e < 8; ++e) tmp[e] = 0.0f;
                    tmp = halo::mma(halo::mk_ldg<v16bf>(ar), w0, tmp);
                    tmp = halo::mma(halo::mk_ldg<v16bf>(ar + 16), w1, tmp);
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[t][e] = fmaf(dq, tmp[e], acc[t][e]);
                }
            }
        }
        if (p + 1 < npass) {
#pragma unroll
            for (int c = 0; c < CH; ++c) xr[c] = xn[c];
#pragma unroll
            for (int c = 0; c < 4; ++c) wr[c] = wn[c];
            dr[0] = dn[0];
            dr[1] = dn[1];
        }
    }
    if (!live) return;
    const long n = rt * 16 + lr;
#pragma unroll
    for (int t = 0; t < TT; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const long m = m0 + t * 16 + 2 * e + (lane >> 4);
            if (m < M) {
                if constexpr (F32O) ((float*)y)[m * c_ld + n] = acc[t][e];
                else ((uint16_t*)y)[m * c_ld + n] = f32_to_bf16(acc[t][e]);
            }
        }
}

/* ---------------------------------------------------------------- launchers */

/* WK: 0 = the rule (enough waves to fill the part: row tiles x WK >= 512). */
int pick_wk(int ntiles, int wk) {
    if (wk == 1 || wk == 2 || wk == 4) return wk;
    return ntiles >= 512 ? 1 : (ntiles >= 256 ? 2 : 4);
}

template <bool BF, bool F32O, int WK>
void launch_mv(const void* x, const void* wq, const void* ws, void* y, int M, int ntiles, long K,
               long a_ld, long c_ld, hipStream_t s) {
    constexpr int TB = 4 / WK;
    q80_mv_kernel<BF, F32O, WK><<<dim3((ntiles + TB - 1) / TB), 128, 0, s>>>(
        (const uint16_t*)x, (const uint8_t*)wq, (const uint16_t*)ws, y, M, ntiles, K, a_ld, c_ld);
}

template <bool BF, bool F32O>
void dispatch_mv(int wk, const void* x, const void* wq, const void* ws, void* y, int M, int ntiles,
                 long K, long a_ld, long c_ld, hipStream_t s) {
    switch (wk) {
        case 1: launch_mv<BF, F32O, 1>(x, wq, ws, y, M, ntiles, K, a_ld, c_ld, s); break;
        case 2: launch_mv<BF, F32O, 2>(x, wq, ws, y, M, ntiles, K, a_ld, c_ld, s); break;
        default: launch_mv<BF, F32O, 4>(x, wq, ws, y, M, ntiles, K, a_ld, c_ld, s); break;
    }
}

/* tt: token tiles a wave (1, 2, 4 with NW = 4; 8 with NW = 8). 0 = by M. */
int pick_tt(int M, int tt) {
    if (tt == 1 || tt == 2 || tt == 4 || tt == 8) return tt;
    if (M <= 16) return 1;
    if (M <= 32) return 2;
    if (M <= 64) return 4;
    return 8;
}

template <int NW, int TT, bool F32O>
void launch_wmma(const void* x, const void* wq, const void* ws, void* y, int M, int ntiles, long K,
                 long a_ld, long c_ld, hipStream_t s) {
    q80_wmma_kernel<NW, TT, F32O><<<dim3((ntiles + NW - 1) / NW, (M + 16 * TT - 1) / (16 * TT)),
                                    NW * 32, 0, s>>>(
        (const uint16_t*)x, (const uint8_t*)wq, (const uint16_t*)ws, y, M, ntiles, K, a_ld, c_ld);
}

template <bool F32O>
void dispatch_wmma(int tt, const void* x, const void* wq, const void* ws, void* y, int M,
                   int ntiles, long K, long a_ld, long c_ld, hipStream_t s) {
    switch (tt) {
        case 1: launch_wmma<4, 1, F32O>(x, wq, ws, y, M, ntiles, K, a_ld, c_ld, s); break;
        case 2: launch_wmma<4, 2, F32O>(x, wq, ws, y, M, ntiles, K, a_ld, c_ld, s); break;
        case 4: launch_wmma<4, 4, F32O>(x, wq, ws, y, M, ntiles, K, a_ld, c_ld, s); break;
        default: launch_wmma<8, 8, F32O>(x, wq, ws, y, M, ntiles, K, a_ld, c_ld, s); break;
    }
}

}  /* namespace */

/* Raw-pointer entries (halo.h): negative codes, -1 a null pointer, -2 a non-positive extent, -3 M
 * past the band, -4 K not a whole 32-block, -5 N not a whole 16-row tile, -6 the launch failed,
 * -7 a bad option. `a_ld`/`c_ld` are elements (0 = tight). `act` 0 = bf16, 1 = f16 (matvec only);
 * `f32out` 1 writes f32 instead of bf16 (a test hook: the exact accumulator). `wk` / `tt` pin the
 * launch form, 0 = the rule. */
extern "C" int halo_gemm_q80a16_nt_mv(long a, long wq, long ws, long c, int M, int K, int N,
                                      int a_ld, int c_ld, int wk, int act, int f32out,
                                      long stream) {
    if (!a || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (M > 8) return -3;
    if (K % 32) return -4;
    if (N % 16) return -5;
    if ((wk != 0 && wk != 1 && wk != 2 && wk != 4) || (act != 0 && act != 1)) return -7;
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    const int nt = N / 16;
    const int w = pick_wk(nt, wk);
    const void* X = (const void*)(uintptr_t)a;
    const void* W = (const void*)(uintptr_t)wq;
    const void* S = (const void*)(uintptr_t)ws;
    void* Y = (void*)(uintptr_t)c;
    hipStream_t st = (hipStream_t)stream;
    if (act == 0) {
        if (f32out) dispatch_mv<true, true>(w, X, W, S, Y, M, nt, K, a_ld, c_ld, st);
        else dispatch_mv<true, false>(w, X, W, S, Y, M, nt, K, a_ld, c_ld, st);
    } else {
        if (f32out) dispatch_mv<false, true>(w, X, W, S, Y, M, nt, K, a_ld, c_ld, st);
        else dispatch_mv<false, false>(w, X, W, S, Y, M, nt, K, a_ld, c_ld, st);
    }
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int halo_gemm_q80a16_nt_wmma(long a, long wq, long ws, long c, int M, int K, int N,
                                        int a_ld, int c_ld, int tt, int f32out, long stream) {
    if (!a || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (K % 32) return -4;
    if (N % 16) return -5;
    if (tt != 0 && tt != 1 && tt != 2 && tt != 4 && tt != 8) return -7;
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    if ((a_ld % 8) || (a % 16)) return -7;       /* 16-byte activation chunks */
    const void* X = (const void*)(uintptr_t)a;
    const void* W = (const void*)(uintptr_t)wq;
    const void* S = (const void*)(uintptr_t)ws;
    void* Y = (void*)(uintptr_t)c;
    hipStream_t st = (hipStream_t)stream;
    const int t = pick_tt(M, tt);
    if (f32out) dispatch_wmma<true>(t, X, W, S, Y, M, N / 16, K, a_ld, c_ld, st);
    else dispatch_wmma<false>(t, X, W, S, Y, M, N / 16, K, a_ld, c_ld, st);
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int halo_gemm_q80a16_nt_mv_max_m(void) { return 8; }

/* ---------------------------------------------------------------- the rows' launch shims */

static int q80_args(const RadArgs* a, const RadTensor** X, const RadTensor** W, const RadTensor** S,
                    const RadTensor** Y, long long* M, long long* N, long long* K) {
    *X = halo_opt(a, 0);
    *W = halo_opt(a, 2);
    *S = halo_opt(a, 3);
    *Y = halo_opt(a, 4);
    if (!*X || !*W || !*S || !*Y) return HALO_NO(RAD_E_INVAL);
    if ((*X)->dtype != RAD_BF16 || (*W)->dtype != RAD_I8 || (*S)->dtype != RAD_F16 ||
        (*Y)->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    const char* dt = halo_gets(a, "dtype");
    if (!dt || std::strcmp(dt, "q80a16")) return HALO_NO(RAD_E_DTYPE);
    if (halo_geti_or(a, "group", 32) != 32) return HALO_NO(RAD_E_SHAPE);
    *K = halo_geti_or(a, "K", 0);
    *N = halo_geti_or(a, "N", 0);
    if (*K <= 0 || *N <= 0 || !halo_q80::shape_ok(*N, *K)) return HALO_NO(RAD_E_SHAPE);
    *M = halo_numel(*X) / *K;
    if (*M < 1 || *M > halo_geti_or(a, "M", *M)) return HALO_NO(RAD_E_SHAPE);
    if (halo_numel(*X) < *M * *K || halo_numel(*W) < *N * *K || halo_numel(*S) < *N * (*K / 32) ||
        halo_numel(*Y) < *M * *N)
        return HALO_NO(RAD_E_SHAPE);
    if (!halo_contig(*X) || !halo_contig(*W) || !halo_contig(*S) || !halo_contig(*Y))
        return HALO_NO(RAD_E_STRIDE);
    return RAD_OK;
}

extern "C" int halo_rad_gemm_q80a16_mv(const RadArgs* a, RadStream stream) {
    const RadTensor *X, *W, *S, *Y;
    long long M, N, K;
    const int rc = q80_args(a, &X, &W, &S, &Y, &M, &N, &K);
    if (rc != RAD_OK) return rc;
    if (M > 8) return HALO_NO(RAD_E_SHAPE);
    return halo_gemm_q80a16_nt_mv((long)(uintptr_t)halo_p(X), (long)(uintptr_t)halo_p(W),
                                  (long)(uintptr_t)halo_p(S), (long)(uintptr_t)halo_p(Y), (int)M,
                                  (int)K, (int)N, 0, 0, 0, 0, 0, (long)stream) == 0
               ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int halo_rad_gemm_q80a16_wmma(const RadArgs* a, RadStream stream) {
    const RadTensor *X, *W, *S, *Y;
    long long M, N, K;
    const int rc = q80_args(a, &X, &W, &S, &Y, &M, &N, &K);
    if (rc != RAD_OK) return rc;
    return halo_gemm_q80a16_nt_wmma((long)(uintptr_t)halo_p(X), (long)(uintptr_t)halo_p(W),
                                    (long)(uintptr_t)halo_p(S), (long)(uintptr_t)halo_p(Y), (int)M,
                                    (int)K, (int)N, 0, 0, 0, 0, (long)stream) == 0
               ? RAD_OK : RAD_E_DEVICE;
}
