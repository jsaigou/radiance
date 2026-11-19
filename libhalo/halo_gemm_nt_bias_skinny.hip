// halo_gemm_nt_bias_skinny.hip -- gemm_nt_bias for a handful of rows: C[M, N] = res + act(A[M, K] @ W[N, K]^T + bias),
// M <= 16 (bf16) or M <= 8 (f32). The streaming dot product of halo_gemm_f32_nt_skinny.hip, with the
// bias, activation and residual store of the tiled kernels.
//
// A decode step feeds these ops one row. The 128-row tile kernels that serve large M read the weight
// at about a fifth of the memory bandwidth there, because a 128-row tile with one live row spends its
// time on staging. Here a wave owns up to eight output columns, its 32 lanes stride K in 16-byte
// chunks, the weights are loaded non-temporally with several columns in flight, and the wave reduces
// once at the end: the cost is the weight's bytes at the card's read bandwidth.
//
// Epilogues are the tiled rows' own, so the two are interchangeable:
//   bf16: t = bf16(acc + bias); if act: t = bf16(act(t)); if res: t = bf16(res + t)      (RNE each step)
//   f32:  t = acc + bias; if act: t = act(t); if res: t = res + t; an f32 C is exact, a bf16 C rounds
//         each step as the unfused chain's tensors would.
// The Haru audio codec's FFNs (N 2048..8192, K 2048..8192) and the audio projectors run here.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_emit.h"

#include <cmath>
#include <cstring>

namespace {

constexpr uint32_t kWaves = 8;
constexpr uint32_t kThr = kWaves * 32;

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ float act_apply(int act, float t) {
    if (act == 1) return 0.5f * t * (1.0f + erff(t * 0.70710678118654752f));
    if (act == 2) {
        const float u = 0.7978845608028654f * (t + 0.044715f * t * t * t);
        return 0.5f * t * (1.0f + tanhf(u));
    }
    return t;
}

__device__ __forceinline__ float rnd_bf16(float v) { return bf16_to_f32(f32_to_bf16(v)); }

// ---- bf16 A, W, C; the bias bf16 or f32 --------------------------------------------------------
template <uint32_t MR, uint32_t RPW, bool VECA, bool BIAS_F32, uint32_t UNR>
__global__ __launch_bounds__(kThr) void gemv_bias_bf16_kernel(
    const uint16_t* __restrict__ a, uint32_t lda, const uint16_t* __restrict__ w, uint32_t K,
    const void* __restrict__ bias, const uint16_t* __restrict__ res, uint32_t ldr,
    uint16_t* __restrict__ y, uint32_t ldy, uint32_t N, uint32_t M, int act) {
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    const uint32_t n0 = (blockIdx.x * kWaves + wave) * RPW;
    if (n0 >= N) return;
    const uint32_t K8 = K >> 3;

    const u32x4* wr[RPW];
#pragma unroll
    for (uint32_t r = 0; r < RPW; ++r) {
        const uint32_t n = n0 + r < N ? n0 + r : N - 1;
        wr[r] = (const u32x4*)(w + (size_t)n * K);
    }
    const uint16_t* ar[MR];
#pragma unroll
    for (uint32_t m = 0; m < MR; ++m) ar[m] = a + (size_t)(m < M ? m : M - 1) * lda;

    float acc[RPW][MR];
#pragma unroll
    for (uint32_t r = 0; r < RPW; ++r)
#pragma unroll
        for (uint32_t m = 0; m < MR; ++m) acc[r][m] = 0.0f;

    // UNR consecutive 16-byte chunks per lane per step are issued together: UNR * RPW loads in flight
    // and a wave reads UNR * 512 contiguous bytes of each row before moving on (fewer DRAM page opens).
    for (uint32_t kb = lane; kb < K8; kb += 32 * UNR) {
        u32x4 wv[UNR][RPW];
#pragma unroll
        for (uint32_t u = 0; u < UNR; ++u) {
            const uint32_t k8 = kb + 32 * u;
#pragma unroll
            for (uint32_t r = 0; r < RPW; ++r)
                wv[u][r] = k8 < K8 ? __builtin_nontemporal_load(wr[r] + k8) : u32x4{ 0, 0, 0, 0 };
        }
#pragma unroll
        for (uint32_t u = 0; u < UNR; ++u) {
            const uint32_t k8 = kb + 32 * u;
            if (k8 >= K8) break;
#pragma unroll
            for (uint32_t m = 0; m < MR; ++m) {
                u32x4 xv;
                if (VECA) {
                    xv = *(const u32x4*)(ar[m] + 8u * k8);
                } else {
                    const uint16_t* p = ar[m] + 8u * k8;
                    xv = u32x4{ (uint32_t)p[0] | ((uint32_t)p[1] << 16), (uint32_t)p[2] | ((uint32_t)p[3] << 16),
                                (uint32_t)p[4] | ((uint32_t)p[5] << 16), (uint32_t)p[6] | ((uint32_t)p[7] << 16) };
                }
#pragma unroll
                for (uint32_t r = 0; r < RPW; ++r) {
                    float s = acc[r][m];
                    s = fmaf(bf_lo(wv[u][r].x), bf_lo(xv.x), s); s = fmaf(bf_hi(wv[u][r].x), bf_hi(xv.x), s);
                    s = fmaf(bf_lo(wv[u][r].y), bf_lo(xv.y), s); s = fmaf(bf_hi(wv[u][r].y), bf_hi(xv.y), s);
                    s = fmaf(bf_lo(wv[u][r].z), bf_lo(xv.z), s); s = fmaf(bf_hi(wv[u][r].z), bf_hi(xv.z), s);
                    s = fmaf(bf_lo(wv[u][r].w), bf_lo(xv.w), s); s = fmaf(bf_hi(wv[u][r].w), bf_hi(xv.w), s);
                    acc[r][m] = s;
                }
            }
        }
    }

#pragma unroll
    for (uint32_t r = 0; r < RPW; ++r)
#pragma unroll
        for (uint32_t m = 0; m < MR; ++m) {
            const float s = halo_wave_sum(acc[r][m]);
            const uint32_t n = n0 + r;
            if (lane == 0 && n < N && m < M) {
                const float b = BIAS_F32 ? ((const float*)bias)[n] : bf16_to_f32(((const uint16_t*)bias)[n]);
                float t = rnd_bf16(s + b);
                if (act) t = rnd_bf16(act_apply(act, t));
                if (res) t = rnd_bf16(bf16_to_f32(res[(size_t)m * ldr + n]) + t);
                y[(size_t)m * ldy + n] = f32_to_bf16(t);
            }
        }
}

// ---- f32 A, W; C f32 or bf16 (the residual C's dtype); the bias f32 or bf16 -----------------------
template <uint32_t MR, uint32_t RPW, bool VECA, bool BIAS_F32, bool OUT_BF16>
__global__ __launch_bounds__(kThr) void gemv_bias_f32_kernel(
    const float* __restrict__ a, uint32_t lda, const float* __restrict__ w, uint32_t K,
    const void* __restrict__ bias, const void* __restrict__ res, uint32_t ldr,
    void* __restrict__ y, uint32_t ldy, uint32_t N, uint32_t M, int act) {
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    const uint32_t n0 = (blockIdx.x * kWaves + wave) * RPW;
    if (n0 >= N) return;
    const uint32_t K4 = K >> 2;

    const f32x4* wr[RPW];
#pragma unroll
    for (uint32_t r = 0; r < RPW; ++r) {
        const uint32_t n = n0 + r < N ? n0 + r : N - 1;
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

    for (uint32_t k4 = lane; k4 < K4; k4 += 32) {
        f32x4 wv[RPW];
#pragma unroll
        for (uint32_t r = 0; r < RPW; ++r) wv[r] = __builtin_nontemporal_load(wr[r] + k4);
#pragma unroll
        for (uint32_t m = 0; m < MR; ++m) {
            f32x4 xv;
            if (VECA) xv = *(const f32x4*)(ar[m] + 4u * k4);
            else      xv = f32x4{ ar[m][4u * k4], ar[m][4u * k4 + 1], ar[m][4u * k4 + 2], ar[m][4u * k4 + 3] };
#pragma unroll
            for (uint32_t r = 0; r < RPW; ++r)
                acc[r][m] = fmaf(wv[r].x, xv.x, fmaf(wv[r].y, xv.y, fmaf(wv[r].z, xv.z,
                                 fmaf(wv[r].w, xv.w, acc[r][m]))));
        }
    }

#pragma unroll
    for (uint32_t r = 0; r < RPW; ++r)
#pragma unroll
        for (uint32_t m = 0; m < MR; ++m) {
            const float s = halo_wave_sum(acc[r][m]);
            const uint32_t n = n0 + r;
            if (lane == 0 && n < N && m < M) {
                const float b = BIAS_F32 ? ((const float*)bias)[n] : bf16_to_f32(((const uint16_t*)bias)[n]);
                if (OUT_BF16) {
                    float t = rnd_bf16(s + b);
                    if (act) t = rnd_bf16(act_apply(act, t));
                    if (res) t = rnd_bf16(bf16_to_f32(((const uint16_t*)res)[(size_t)m * ldr + n]) + t);
                    ((uint16_t*)y)[(size_t)m * ldy + n] = f32_to_bf16(t);
                } else {
                    float t = s + b;
                    if (act) t = act_apply(act, t);
                    if (res) t = ((const float*)res)[(size_t)m * ldr + n] + t;
                    ((float*)y)[(size_t)m * ldy + n] = t;
                }
            }
        }
}

struct Plan { unsigned mr, rpw; dim3 grid; };
static Plan plan(long long M, long long N) {
    Plan p;
    p.mr = M <= 1 ? 1u : M <= 2 ? 2u : M <= 4 ? 4u : M <= 8 ? 8u : 16u;
    // One column a wave: measured on the card, more columns a wave (fewer waves, each with more rows
    // open at once) is 1.5 to 2.3 times slower at the codec's shapes. Parallelism comes from the waves.
    p.rpw = 1;
    p.grid = dim3((unsigned)((N + kWaves - 1) / kWaves));
    return p;
}

int parse_act(const RadArgs* a, int* act) {
    const char* act_s = halo_gets(a, "act");
    *act = 0;
    if (act_s && *act_s) {
        if (!std::strcmp(act_s, "gelu")) *act = 1;
        else if (!std::strcmp(act_s, "gelu_tanh")) *act = 2;
        else if (std::strcmp(act_s, "none")) return RAD_E_SCHEMA;
    }
    return RAD_OK;
}

}  // namespace

enum { SB_A = 0, SB_W, SB_BIAS, SB_RES, SB_Y };

extern "C" int halo_rad_gemm_nt_bias_skinny_bf16(const RadArgs* a, RadStream stream) {
    const RadTensor* A = halo_opt(a, SB_A);
    const RadTensor* W = halo_opt(a, SB_W);
    const RadTensor* B = halo_opt(a, SB_BIAS);
    const RadTensor* R = halo_opt(a, SB_RES);
    const RadTensor* Y = halo_opt(a, SB_Y);
    if (!A || !W || !B || !Y) return HALO_NO(RAD_E_INVAL);
    if (A->dtype != RAD_BF16 || W->dtype != RAD_BF16 || Y->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_F32 && B->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    const bool has_res = R && R->data;
    if (has_res && R->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if (!halo_contig(W) || !halo_contig(B)) return HALO_NO(RAD_E_STRIDE);
    int act = 0;
    if (parse_act(a, &act) != RAD_OK) return HALO_NO(RAD_E_SCHEMA);

    const long long K = halo_geti_or(a, "K", halo_dim(W, (int)W->rank - 1));
    const long long N = halo_geti_or(a, "N", K > 0 ? halo_numel(W) / K : 0);
    if (K <= 0 || N <= 0 || halo_numel(W) < N * K || halo_numel(B) < N) return HALO_NO(RAD_E_SHAPE);
    if (K % 8 != 0 || !halo_aligned(W->data, 16)) return HALO_NO(RAD_E_ALIGN);
    long long M = 0, Ma = 0, Mr = 0;
    const long long ldy = halo_rows_pitch(Y, N, &M);
    const long long lda = halo_rows_pitch(A, K, &Ma);
    const long long ldr = has_res ? halo_rows_pitch(R, N, &Mr) : 0;
    if (ldy < 0 || lda < 0 || ldr < 0) return HALO_NO(RAD_E_STRIDE);
    if (M <= 0 || M > 16 || Ma < M || (has_res && Mr < M)) return HALO_NO(RAD_E_SHAPE);
    if (N * K >= (1ll << 40) || lda > 0x7fffffffll || ldy > 0x7fffffffll || ldr > 0x7fffffffll)
        return HALO_NO(RAD_E_SHAPE);

    const bool veca = (lda % 8 == 0) && halo_aligned(A->data, 16);
    const bool bf32 = B->dtype == RAD_F32;
    const Plan p = plan(M, N);
    const dim3 block(kThr);
    // Chunks a lane keeps in flight, by how many the row has: 8 from K = 2048 (bf16).
    const unsigned unr = K / 8 >= 32 * 8 ? 8u : K / 8 >= 32 * 4 ? 4u : K / 8 >= 32 * 2 ? 2u : 1u;
#define SBB1(MR_, RPW_, UNR_)                                                                          \
    do {                                                                                              \
        if (veca && bf32)                                                                             \
            return HALO_EMIT(stream, (gemv_bias_bf16_kernel<MR_, RPW_, true, true, UNR_>), p.grid, block, 0, \
                (const uint16_t*)A->data, (uint32_t)lda, (const uint16_t*)W->data, (uint32_t)K,       \
                B->data, has_res ? (const uint16_t*)R->data : (const uint16_t*)nullptr, (uint32_t)ldr, \
                (uint16_t*)Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act);                    \
        if (veca)                                                                                     \
            return HALO_EMIT(stream, (gemv_bias_bf16_kernel<MR_, RPW_, true, false, UNR_>), p.grid, block, 0, \
                (const uint16_t*)A->data, (uint32_t)lda, (const uint16_t*)W->data, (uint32_t)K,       \
                B->data, has_res ? (const uint16_t*)R->data : (const uint16_t*)nullptr, (uint32_t)ldr, \
                (uint16_t*)Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act);                    \
        if (bf32)                                                                                     \
            return HALO_EMIT(stream, (gemv_bias_bf16_kernel<MR_, RPW_, false, true, UNR_>), p.grid, block, 0, \
                (const uint16_t*)A->data, (uint32_t)lda, (const uint16_t*)W->data, (uint32_t)K,       \
                B->data, has_res ? (const uint16_t*)R->data : (const uint16_t*)nullptr, (uint32_t)ldr, \
                (uint16_t*)Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act);                    \
        return HALO_EMIT(stream, (gemv_bias_bf16_kernel<MR_, RPW_, false, false, UNR_>), p.grid, block, 0, \
            (const uint16_t*)A->data, (uint32_t)lda, (const uint16_t*)W->data, (uint32_t)K,           \
            B->data, has_res ? (const uint16_t*)R->data : (const uint16_t*)nullptr, (uint32_t)ldr,     \
            (uint16_t*)Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act);                        \
    } while (0)
#define SBB(MR_, RPW_)                                                                                \
    do { if (unr == 8) SBB1(MR_, RPW_, 8); if (unr == 4) SBB1(MR_, RPW_, 4); if (unr == 1) SBB1(MR_, RPW_, 1); SBB1(MR_, RPW_, 2); } while (0)
    if (p.mr == 1) { if (p.rpw == 8) SBB(1, 8); if (p.rpw == 4) SBB(1, 4); if (p.rpw == 2) SBB(1, 2); SBB(1, 1); }
    if (p.mr == 2) { if (p.rpw >= 8) SBB(2, 8); if (p.rpw >= 4) SBB(2, 4); if (p.rpw == 2) SBB(2, 2); SBB(2, 1); }
    if (p.mr == 4) { if (p.rpw >= 4) SBB(4, 4); if (p.rpw == 2) SBB(4, 2); SBB(4, 1); }
    if (p.mr == 8) { if (p.rpw >= 2) SBB(8, 2); SBB(8, 1); }
    SBB(16, 1);
#undef SBB1
#undef SBB
}

extern "C" int halo_rad_gemm_nt_bias_skinny_f32(const RadArgs* a, RadStream stream) {
    const RadTensor* A = halo_opt(a, SB_A);
    const RadTensor* W = halo_opt(a, SB_W);
    const RadTensor* B = halo_opt(a, SB_BIAS);
    const RadTensor* R = halo_opt(a, SB_RES);
    const RadTensor* Y = halo_opt(a, SB_Y);
    if (!A || !W || !B || !Y) return HALO_NO(RAD_E_INVAL);
    if (A->dtype != RAD_F32 || W->dtype != RAD_F32) return HALO_NO(RAD_E_DTYPE);
    if (Y->dtype != RAD_F32 && Y->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_F32 && B->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    const bool has_res = R && R->data;
    if (has_res && R->dtype != Y->dtype) return HALO_NO(RAD_E_DTYPE);
    if (!halo_contig(W) || !halo_contig(B)) return HALO_NO(RAD_E_STRIDE);
    int act = 0;
    if (parse_act(a, &act) != RAD_OK) return HALO_NO(RAD_E_SCHEMA);

    const long long K = halo_geti_or(a, "K", halo_dim(W, (int)W->rank - 1));
    const long long N = halo_geti_or(a, "N", K > 0 ? halo_numel(W) / K : 0);
    if (K <= 0 || N <= 0 || halo_numel(W) < N * K || halo_numel(B) < N) return HALO_NO(RAD_E_SHAPE);
    if (K % 4 != 0 || !halo_aligned(W->data, 16)) return HALO_NO(RAD_E_ALIGN);
    long long M = 0, Ma = 0, Mr = 0;
    const long long ldy = halo_rows_pitch(Y, N, &M);
    const long long lda = halo_rows_pitch(A, K, &Ma);
    const long long ldr = has_res ? halo_rows_pitch(R, N, &Mr) : 0;
    if (ldy < 0 || lda < 0 || ldr < 0) return HALO_NO(RAD_E_STRIDE);
    if (M <= 0 || M > 8 || Ma < M || (has_res && Mr < M)) return HALO_NO(RAD_E_SHAPE);
    if (N * K >= (1ll << 40) || lda > 0x7fffffffll || ldy > 0x7fffffffll || ldr > 0x7fffffffll)
        return HALO_NO(RAD_E_SHAPE);

    const bool veca = (lda % 4 == 0) && halo_aligned(A->data, 16);
    const bool bf32 = B->dtype == RAD_F32;
    const bool obf = Y->dtype == RAD_BF16;
    Plan p = plan(M, N);
    if (p.mr == 16) p.mr = 8;
    const unsigned budget = 8u / p.mr;       // f32 weights are twice the registers per load
    if (p.rpw > budget) p.rpw = budget;
    p.grid = dim3((unsigned)((N + kWaves * p.rpw - 1) / (kWaves * p.rpw)));
    const dim3 block(kThr);
    const void* resp = has_res ? R->data : nullptr;
#define SBF(MR_, RPW_)                                                                              \
    do {                                                                                            \
        const void* rr_ = resp;                                                                     \
        if (veca && bf32 && !obf) return HALO_EMIT(stream, (gemv_bias_f32_kernel<MR_, RPW_, true, true, false>), p.grid, block, 0, (const float*)A->data, (uint32_t)lda, (const float*)W->data, (uint32_t)K, B->data, rr_, (uint32_t)ldr, Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act); \
        if (veca && bf32 && obf)  return HALO_EMIT(stream, (gemv_bias_f32_kernel<MR_, RPW_, true, true, true>), p.grid, block, 0, (const float*)A->data, (uint32_t)lda, (const float*)W->data, (uint32_t)K, B->data, rr_, (uint32_t)ldr, Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act); \
        if (veca && !bf32 && !obf) return HALO_EMIT(stream, (gemv_bias_f32_kernel<MR_, RPW_, true, false, false>), p.grid, block, 0, (const float*)A->data, (uint32_t)lda, (const float*)W->data, (uint32_t)K, B->data, rr_, (uint32_t)ldr, Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act); \
        if (veca && !bf32 && obf)  return HALO_EMIT(stream, (gemv_bias_f32_kernel<MR_, RPW_, true, false, true>), p.grid, block, 0, (const float*)A->data, (uint32_t)lda, (const float*)W->data, (uint32_t)K, B->data, rr_, (uint32_t)ldr, Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act); \
        if (!veca && bf32 && !obf) return HALO_EMIT(stream, (gemv_bias_f32_kernel<MR_, RPW_, false, true, false>), p.grid, block, 0, (const float*)A->data, (uint32_t)lda, (const float*)W->data, (uint32_t)K, B->data, rr_, (uint32_t)ldr, Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act); \
        if (!veca && bf32 && obf)  return HALO_EMIT(stream, (gemv_bias_f32_kernel<MR_, RPW_, false, true, true>), p.grid, block, 0, (const float*)A->data, (uint32_t)lda, (const float*)W->data, (uint32_t)K, B->data, rr_, (uint32_t)ldr, Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act); \
        if (!veca && !bf32 && !obf) return HALO_EMIT(stream, (gemv_bias_f32_kernel<MR_, RPW_, false, false, false>), p.grid, block, 0, (const float*)A->data, (uint32_t)lda, (const float*)W->data, (uint32_t)K, B->data, rr_, (uint32_t)ldr, Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act); \
        return HALO_EMIT(stream, (gemv_bias_f32_kernel<MR_, RPW_, false, false, true>), p.grid, block, 0, (const float*)A->data, (uint32_t)lda, (const float*)W->data, (uint32_t)K, B->data, rr_, (uint32_t)ldr, Y->data, (uint32_t)ldy, (uint32_t)N, (uint32_t)M, act); \
    } while (0)
    if (p.mr == 1) { if (p.rpw == 8) SBF(1, 8); if (p.rpw == 4) SBF(1, 4); if (p.rpw == 2) SBF(1, 2); SBF(1, 1); }
    if (p.mr == 2) { if (p.rpw >= 4) SBF(2, 4); if (p.rpw == 2) SBF(2, 2); SBF(2, 1); }
    if (p.mr == 4) { if (p.rpw >= 2) SBF(4, 2); SBF(4, 1); }
    SBF(8, 1);
#undef SBF
}
