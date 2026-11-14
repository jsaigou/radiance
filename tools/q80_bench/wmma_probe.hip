// wmma_probe -- is gfx1151's 16x16x16 bf16/f16 WMMA exact on small integers? One block, one wave.
#include <hip/hip_runtime.h>
#include "halo_wmma.h"
#include <cstdio>
#include <cstring>

template <typename FR, typename E>
__global__ void k(const float* a, const float* b, float* d, int mode) {
    const int lane = threadIdx.x;
    E ea[16], eb[16];
    for (int e = 0; e < 16; ++e) { ea[e] = (E)a[(lane & 15) * 16 + e]; eb[e] = (E)b[e * 16 + (lane & 15)]; }
    FR fa, fb; memcpy(&fa, ea, 32); memcpy(&fb, eb, 32);
    v8f c; for (int e = 0; e < 8; ++e) c[e] = 0.0f;
    v8f r = halo::mma(fa, fb, c);
    for (int e = 0; e < 8; ++e) d[(2 * e + (lane >> 4)) * 16 + (lane & 15)] = r[e];
}
static uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
int main() {
    float ha[256], hb[256], hd[256], *da, *db, *dd;
    hipMalloc(&da, 1024); hipMalloc(&db, 1024); hipMalloc(&dd, 1024);
    for (int variant = 0; variant < 4; ++variant) {
        // variant 0: A one-hot (row m has 1.0 at k=m), B ints -128..127 ; 1: A ints -8..8, B ints ; 2: A one-hot, B = 3 ; 3: A = 1, B ints
        for (int i = 0; i < 256; ++i) {
            int m = i / 16, kk = i % 16;
            ha[i] = variant == 0 || variant == 2 ? (m == kk ? 1.0f : 0.0f) : (variant == 3 ? 1.0f : (float)((int)(mix(i) % 17) - 8));
            hb[i] = variant == 2 ? 3.0f : (float)((int)(mix(i + 999) & 0xFF) - 128);
        }
        hipMemcpy(da, ha, 1024, hipMemcpyHostToDevice); hipMemcpy(db, hb, 1024, hipMemcpyHostToDevice);
        for (int ty = 0; ty < 2; ++ty) {
            if (ty == 0) k<v16bf, __bf16><<<1, 32>>>(da, db, dd, 0); else k<v16hf, __fp16><<<1, 32>>>(da, db, dd, 0);
            hipDeviceSynchronize();
            hipMemcpy(hd, dd, 1024, hipMemcpyDeviceToHost);
            int bad = 0; double worst = 0;
            for (int m = 0; m < 16; ++m) for (int n = 0; n < 16; ++n) {
                double s = 0; for (int kk = 0; kk < 16; ++kk) s += (double)ha[m * 16 + kk] * hb[kk * 16 + n];
                if (hd[m * 16 + n] != (float)s) { ++bad; worst = fmax(worst, fabs(hd[m * 16 + n] - s)); }
            }
            printf("variant %d %s: %d of 256 inexact, worst abs err %g\n", variant, ty ? "f16 " : "bf16", bad, worst);
        }
    }
    return 0;
}
