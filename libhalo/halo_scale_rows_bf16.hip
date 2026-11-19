/* Derived from libr4d/r4d_moe.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_scale_rows_bf16.hip -- the shared expert's one-scalar row gate, gfx1151.
//
// y[m, i] = x[m, i] * act(s[m]); with the optional `ad`, the fused gate-and-add form
// y = bf16(x * act(s)) + ad. No fragment, no LDS: one workgroup per row, the scalar read once.
// Extracted from libr4d's r4d_moe.hip, which other lanes own.
#include "halo.h"
#include "halo_args.h"
#include "halo_emit.h"
#include "halo_common.h"

#include <cstdint>
#include <cstring>

namespace {

/* This unit's own error check: libr4d's R4D_MOE_CHK returns a raw negative hipError_t, the halo
 * rows answer RAD_E_* codes. */
#define HALO_SR_CHK() do { hipError_t e_ = hipGetLastError();                    \
                           if (e_ != hipSuccess) return HALO_NO(RAD_E_DEVICE); } while (0)

/* ================================================================== scale_rows
 *
 * y[m, i] = x[m, i] * act(s[m]). One workgroup per row, grid-strided along it, with the scalar read
 * once. Used only for the shared expert's gate; docs/OPS.md explains why `mul` cannot serve it. */
/* With `ad`, the fused gate-and-add form `y = bf16(x * act(s)) + ad`. The middle narrow matches the
 * unfused pair, which stores the product as bf16 before adding; an f32 product would change the
 * last bit of every channel. */
template <uint32_t NTHR>
__global__ __launch_bounds__(NTHR) void scale_rows_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ sc,
    const uint16_t* __restrict__ ad, uint16_t* __restrict__ y,
    uint32_t M, uint32_t n, uint32_t x_ld, uint32_t s_ld, uint32_t a_ld, uint32_t y_ld, int sig) {
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const float v0 = bf16_to_f32(sc[(size_t)m * s_ld]);
    const float v  = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    for (uint32_t i = threadIdx.x; i < n; i += NTHR) {
        const uint16_t p = f32_to_bf16(bf16_to_f32(x[(size_t)m * x_ld + i]) * v);
        y[(size_t)m * y_ld + i] =
            ad ? f32_to_bf16(bf16_to_f32(p) + bf16_to_f32(ad[(size_t)m * a_ld + i])) : p;
    }
}


/* The f32 form (the head's reciprocal broadcast, `x` a row of ones): y = x * act(s) [+ ad]. `s` is
 * one f32 or bf16 scalar a row. float4 when n % 4 == 0 and every pitch and base is 16-byte aligned,
 * and the block is as narrow as the row allows. The fused-add form multiplies, rounds, then adds
 * (__fmul_rn) so it is never contracted into an fma: it is bit-identical to scale_rows then add. */
template <uint32_t THR, bool SC_BF16, bool VEC>
__global__ __launch_bounds__(THR) void scale_rows_f32_kernel(
    const float* __restrict__ x, const void* __restrict__ sc, const float* __restrict__ ad,
    float* __restrict__ y, uint32_t M, uint32_t n, uint32_t x_ld, uint32_t s_ld, uint32_t a_ld,
    uint32_t y_ld, int sig) {
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const float v0 = SC_BF16 ? bf16_to_f32(((const uint16_t*)sc)[(size_t)m * s_ld])
                             : ((const float*)sc)[(size_t)m * s_ld];
    const float v = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    if (VEC) {
        const float4* __restrict__ xr = (const float4*)(x + (size_t)m * x_ld);
        float4* __restrict__ yr = (float4*)(y + (size_t)m * y_ld);
        const float4* __restrict__ ar = ad ? (const float4*)(ad + (size_t)m * a_ld) : nullptr;
        for (uint32_t i = threadIdx.x; i < n / 4u; i += THR) {
            const float4 u = xr[i];
            float4 r = make_float4(__fmul_rn(u.x, v), __fmul_rn(u.y, v), __fmul_rn(u.z, v),
                                   __fmul_rn(u.w, v));
            if (ar) { const float4 d = ar[i]; r.x += d.x; r.y += d.y; r.z += d.z; r.w += d.w; }
            yr[i] = r;
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR) {
            float r = __fmul_rn(x[(size_t)m * x_ld + i], v);
            if (ad) r += ad[(size_t)m * a_ld + i];
            y[(size_t)m * y_ld + i] = r;
        }
    }
}

template <bool SC_BF16, bool VEC>
static void launch_scale_rows_f32(const float* x, const void* sc, const float* ad, float* y,
                                  uint32_t M, uint32_t n, uint32_t x_ld, uint32_t s_ld,
                                  uint32_t a_ld, uint32_t y_ld, int sig, hipStream_t s) {
    const uint32_t units = VEC ? n / 4u : n;
    if (units <= 32)
        hipLaunchKernelGGL((scale_rows_f32_kernel<32, SC_BF16, VEC>), dim3(M), dim3(32), 0, s, x, sc, ad, y, M, n, x_ld, s_ld, a_ld, y_ld, sig);
    else if (units <= 64)
        hipLaunchKernelGGL((scale_rows_f32_kernel<64, SC_BF16, VEC>), dim3(M), dim3(64), 0, s, x, sc, ad, y, M, n, x_ld, s_ld, a_ld, y_ld, sig);
    else if (units <= 128)
        hipLaunchKernelGGL((scale_rows_f32_kernel<128, SC_BF16, VEC>), dim3(M), dim3(128), 0, s, x, sc, ad, y, M, n, x_ld, s_ld, a_ld, y_ld, sig);
    else
        hipLaunchKernelGGL((scale_rows_f32_kernel<256, SC_BF16, VEC>), dim3(M), dim3(256), 0, s, x, sc, ad, y, M, n, x_ld, s_ld, a_ld, y_ld, sig);
}

}  /* namespace */

extern "C" int halo_rad_scale_rows(const RadArgs* a, RadStream s) {
    const RadTensor* x  = halo_opt(a, 0);
    const RadTensor* sc = halo_opt(a, 1);
    const RadTensor* ad = halo_opt(a, 2);          /* optional: the fused gate-and-add form */
    const RadTensor* y  = halo_opt(a, 3);
    if (!x || !sc || !y || !x->data || !sc->data || !y->data) return HALO_NO(RAD_E_INVAL);
    if (ad && !ad->data) ad = nullptr;
    /* Two legs: all bf16 (the shared expert's gate), or f32 x, y (and `ad`) with a bf16 or f32 scalar. */
    const bool leg_f32 = x->dtype == RAD_F32 && y->dtype == RAD_F32 &&
                         (sc->dtype == RAD_F32 || sc->dtype == RAD_BF16) &&
                         (!ad || ad->dtype == RAD_F32);
    if (!leg_f32) {
        if (x->dtype != RAD_BF16 || sc->dtype != RAD_BF16 || y->dtype != RAD_BF16)
            return HALO_NO(RAD_E_DTYPE);
        if (ad && ad->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    }
    const long long n = halo_geti_or(a, "n", 0);
    if (n <= 0) return HALO_NO(RAD_E_SHAPE);
    /* Read as a name, not via halo_act_code (whose vocabulary is silu/sigmoid, not sigmoid/none), so
     * "none" is distinguishable from a typo. An unknown name is refused. */
    const char* act = halo_gets(a, "act");
    const int sig = (act && !std::strcmp(act, "sigmoid")) ? 1 : 0;
    if (act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return HALO_NO(RAD_E_INVAL);

    long long M = 0;
    const long long y_ld = halo_rows_pitch(y, n, &M);
    if (y_ld < 0 || M <= 0) return HALO_NO(RAD_E_STRIDE);
    long long Mx = 0;
    const long long x_ld = halo_rows_pitch(x, n, &Mx);
    if (x_ld < 0 || Mx < M) return HALO_NO(RAD_E_STRIDE);
    if (halo_numel(sc) < M) return HALO_NO(RAD_E_SHAPE);
    long long a_ld = 0;
    if (ad) {
        long long Ma = 0;
        a_ld = halo_rows_pitch(ad, n, &Ma);
        if (a_ld < 0 || Ma < M) return HALO_NO(RAD_E_STRIDE);
    }
    /* One scalar per row, as [M] or [M, 1]; the row stride is dim 0's in both. */
    const long long s_ld = halo_stride(sc, 0) > 0 ? halo_stride(sc, 0) : 1;

    if (leg_f32) {
        const bool vec = (n % 4 == 0) && (x_ld % 4 == 0) && (y_ld % 4 == 0) && (!ad || a_ld % 4 == 0) &&
                         (((uintptr_t)x->data | (uintptr_t)y->data | (ad ? (uintptr_t)ad->data : 0)) & 15u) == 0;
        const uint32_t sl = (uint32_t)(s_ld ? s_ld : 1);
        const float* xp = (const float*)x->data;
        const float* ap = ad ? (const float*)ad->data : nullptr;
        float* yp = (float*)y->data;
        hipStream_t st = (hipStream_t)s;
        if (sc->dtype == RAD_BF16) {
            if (vec) launch_scale_rows_f32<true, true>(xp, sc->data, ap, yp, (uint32_t)M, (uint32_t)n, (uint32_t)x_ld, sl, (uint32_t)a_ld, (uint32_t)y_ld, sig, st);
            else     launch_scale_rows_f32<true, false>(xp, sc->data, ap, yp, (uint32_t)M, (uint32_t)n, (uint32_t)x_ld, sl, (uint32_t)a_ld, (uint32_t)y_ld, sig, st);
        } else {
            if (vec) launch_scale_rows_f32<false, true>(xp, sc->data, ap, yp, (uint32_t)M, (uint32_t)n, (uint32_t)x_ld, sl, (uint32_t)a_ld, (uint32_t)y_ld, sig, st);
            else     launch_scale_rows_f32<false, false>(xp, sc->data, ap, yp, (uint32_t)M, (uint32_t)n, (uint32_t)x_ld, sl, (uint32_t)a_ld, (uint32_t)y_ld, sig, st);
        }
        HALO_SR_CHK();
        return RAD_OK;
    }
    hipLaunchKernelGGL((scale_rows_kernel<256>), dim3((unsigned)M), dim3(256), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (const uint16_t*)sc->data,
                       ad ? (const uint16_t*)ad->data : nullptr, (uint16_t*)y->data,
                       (uint32_t)M, (uint32_t)n, (uint32_t)x_ld, (uint32_t)(s_ld ? s_ld : 1),
                       (uint32_t)a_ld, (uint32_t)y_ld, sig);
    HALO_SR_CHK();
    return RAD_OK;
}
