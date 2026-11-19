// halo_gemm_f32_nt_skinny.hip -- gemm_nt at f32 for a handful of rows: Y[M, N] = A[M, K] @ W[N, K]^T,
// f32 activation, f32 weight, f32 accumulation and output, M <= 8.
//
// A decode-height projection reads every weight once and uses it for at most eight rows, so it is
// bound by the weights' bytes, not by arithmetic. This kernel is therefore a streaming dot product:
// one wave owns up to eight output columns at a time, its 32 lanes stride K in float4s (a wave
// reads 512 contiguous bytes of each weight row per step, with RPW rows in flight to cover the
// memory latency), the activation vector is re-read from cache, and the weights are loaded
// non-temporally so a 1.4 GB stream does not push the activations out of the caches. Each lane keeps
// its own partial sums and the wave reduces them once at the end.
//
// The Haru MeanFlow head and audio projectors run here: their weights are widened from the
// checkpoint's bf16 to f32 at load, M is the lane count (1 to 8), N and K are 64 to 7680.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_emit.h"

#include <cstring>

namespace {

constexpr uint32_t kWaves = 8;
constexpr uint32_t kThr = kWaves * 32;

typedef float f32x4 __attribute__((ext_vector_type(4)));

// MR rows of A, RPW output columns per wave in flight. VECA: A's rows are 16-byte aligned.
template <uint32_t MR, uint32_t RPW, bool VECA, uint32_t UNR>
__global__ __launch_bounds__(kThr) void gemv_f32_kernel(
    const float* __restrict__ a, uint32_t lda, const float* __restrict__ w, uint32_t K,
    float* __restrict__ y, uint32_t ldy, uint32_t N, uint32_t M) {
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    const uint32_t n0 = (blockIdx.x * kWaves + wave) * RPW;
    if (n0 >= N) return;
    const uint32_t K4 = K >> 2;

    const f32x4* wr[RPW];
#pragma unroll
    for (uint32_t r = 0; r < RPW; ++r) {
        const uint32_t n = n0 + r < N ? n0 + r : N - 1;   // a clamped row is computed and not stored
        wr[r] = (const f32x4*)(w + (size_t)n * K);
    }
    const float* ar[MR];
#pragma unroll
    for (uint32_t m = 0; m < MR; ++m) ar[m] = a + (size_t)(m < M ? m : M - 1) * lda;

    float acc[RPW][MR];
#pragma unroll
    for (uint32_t r = 0; r < RPW; ++r)
#pragma unroll
        for (uint32_t m = 0; m < MR; ++m) acc[r][m] = 0.0f;

    // UNR consecutive 16-byte chunks per lane per step are loaded together (UNR * RPW loads in flight),
    // so a wave reads UNR * 512 contiguous bytes of a row before it moves on.
    for (uint32_t kb = lane; kb < K4; kb += 32 * UNR) {
        f32x4 wv[UNR][RPW];
#pragma unroll
        for (uint32_t u = 0; u < UNR; ++u) {
            const uint32_t k4 = kb + 32 * u;
#pragma unroll
            for (uint32_t r = 0; r < RPW; ++r)
                wv[u][r] = k4 < K4 ? __builtin_nontemporal_load(wr[r] + k4) : f32x4{ 0, 0, 0, 0 };
        }
#pragma unroll
        for (uint32_t u = 0; u < UNR; ++u) {
            const uint32_t k4 = kb + 32 * u;
            if (k4 >= K4) break;
#pragma unroll
            for (uint32_t m = 0; m < MR; ++m) {
                f32x4 xv;
                if (VECA) {
                    xv = *(const f32x4*)(ar[m] + 4u * k4);
                } else {
                    xv = f32x4{ ar[m][4u * k4], ar[m][4u * k4 + 1], ar[m][4u * k4 + 2], ar[m][4u * k4 + 3] };
                }
#pragma unroll
                for (uint32_t r = 0; r < RPW; ++r)
                    acc[r][m] = fmaf(wv[u][r].x, xv.x, fmaf(wv[u][r].y, xv.y, fmaf(wv[u][r].z, xv.z,
                                     fmaf(wv[u][r].w, xv.w, acc[r][m]))));
            }
        }
    }

#pragma unroll
    for (uint32_t r = 0; r < RPW; ++r)
#pragma unroll
        for (uint32_t m = 0; m < MR; ++m) {
            const float s = halo_wave_sum(acc[r][m]);
            if (lane == 0 && n0 + r < N && m < M) y[(size_t)m * ldy + n0 + r] = s;
        }
}

}  // namespace

enum { GS_A = 0, GS_W, GS_Y };

extern "C" int halo_rad_gemm_nt_f32_skinny(const RadArgs* a, RadStream stream) {
    const RadTensor* A = halo_opt(a, GS_A);
    const RadTensor* W = halo_opt(a, GS_W);
    const RadTensor* Y = halo_opt(a, GS_Y);
    if (!A || !W || !Y) return HALO_NO(RAD_E_INVAL);
    if (A->dtype != RAD_F32 || W->dtype != RAD_F32 || Y->dtype != RAD_F32) return HALO_NO(RAD_E_DTYPE);
    if (!halo_contig(W)) return HALO_NO(RAD_E_STRIDE);

    const long long K = halo_geti_or(a, "K", halo_dim(W, (int)W->rank - 1));
    const long long N = halo_geti_or(a, "N", halo_numel(W) / (K > 0 ? K : 1));
    if (K <= 0 || N <= 0 || halo_numel(W) < N * K) return HALO_NO(RAD_E_SHAPE);
    if (K % 4 != 0 || !halo_aligned(W->data, 16)) return HALO_NO(RAD_E_ALIGN);
    long long M = 0, Ma = 0;
    const long long ldy = halo_rows_pitch(Y, N, &M);
    const long long lda = halo_rows_pitch(A, K, &Ma);
    if (ldy < 0 || lda < 0) return HALO_NO(RAD_E_STRIDE);
    if (M <= 0 || M > 8 || Ma < M) return HALO_NO(RAD_E_SHAPE);
    if (N * K >= (1ll << 40) || lda > 0x7fffffffll || ldy > 0x7fffffffll || K > 0x7fffffffll)
        return HALO_NO(RAD_E_SHAPE);

    const bool veca = (lda % 4 == 0) && halo_aligned(A->data, 16);
    const unsigned mr = M <= 1 ? 1u : M <= 2 ? 2u : M <= 4 ? 4u : 8u;
    // One column a wave, UNR chunks a lane in flight: measured on the card, more columns a wave
    // (fewer waves, each with more rows open) is slower at every shape the head issues.
    const unsigned rpw = 1;
    const unsigned unr = K / 4 >= 32 * 8 ? 8u : K / 4 >= 32 * 4 ? 4u : K / 4 >= 32 * 2 ? 2u : 1u;
    const unsigned cols_per_block = kWaves * rpw;
    const dim3 grid((unsigned)((N + cols_per_block - 1) / cols_per_block));
    const dim3 block(kThr);

#define GS_L1(MR_, RPW_, UNR_)                                                                    \
    do {                                                                                          \
        if (veca)                                                                                 \
            return HALO_EMIT(stream, (gemv_f32_kernel<MR_, RPW_, true, UNR_>), grid, block, 0,     \
                             (const float*)A->data, (uint32_t)lda, (const float*)W->data,         \
                             (uint32_t)K, (float*)Y->data, (uint32_t)ldy, (uint32_t)N,            \
                             (uint32_t)M);                                                        \
        return HALO_EMIT(stream, (gemv_f32_kernel<MR_, RPW_, false, UNR_>), grid, block, 0,       \
                         (const float*)A->data, (uint32_t)lda, (const float*)W->data,             \
                         (uint32_t)K, (float*)Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M);  \
    } while (0)
#define GS_L(MR_, RPW_)                                                                           \
    do { if (unr == 8) GS_L1(MR_, RPW_, 8); if (unr == 4) GS_L1(MR_, RPW_, 4);                    \
         if (unr == 2) GS_L1(MR_, RPW_, 2); GS_L1(MR_, RPW_, 1); } while (0)
    if (mr == 1) {
        if (rpw == 8) GS_L(1, 8);
        if (rpw == 4) GS_L(1, 4);
        if (rpw == 2) GS_L(1, 2);
        GS_L(1, 1);
    }
    if (mr == 2) {
        if (rpw >= 4) GS_L(2, 4);
        if (rpw == 2) GS_L(2, 2);
        GS_L(2, 1);
    }
    if (mr == 4) {
        if (rpw >= 2) GS_L(4, 2);
        GS_L(4, 1);
    }
    GS_L(8, 1);
#undef GS_L1
#undef GS_L
}
