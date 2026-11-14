/* halo_q80.h -- ggml Q8_0 as libhalo stores and reads it, for dense weights.
 *
 * Q8_0 is llama.cpp's block_q8_0: 32 weights = one f16 scale d and 32 int8 codes q, dequantised as
 * `w = (float)d * (float)q` (ggml's dequantize_row_q8_0). The container holds it as two canonical
 * planes (core/format/ggml_planes.h split_q8_0, libquant/lq_ggml.cpp): `codes` i8 [N][K] and
 * `scale` f16 [N][K/32], both byte copies of the GGUF block fields. Nothing here rounds: d has 11
 * significant bits and q 8, so d*q has at most 19 and is exact in f32. The dequantised weight is
 * therefore bit for bit ggml's, in every layout below, and only the ORDER in which a kernel sums
 * products can differ from llama.cpp (which also quantises the activation to Q8_0; libhalo does not).
 *
 * The stored form ("halo.q80.tile16x32") is one for both kernels. A TILE is 16 weight rows by one
 * 32-wide K block, 512 bytes of codes in row order, so a wave reading a tile with 16 bytes a lane
 * reads 512 contiguous bytes, and a WMMA B fragment (one row's 16 consecutive k per lane) is one
 * 16-byte load. Tiles run K-fastest within a row tile:
 *
 *   codes   i8 [N/16][K/32][16 rows][32]      byte (rt*KB + kb)*512 + r*32 + j,   KB = K/32
 *   scales  f16 [N/16][K/32][16 rows]         half (rt*KB + kb)*16  + r
 *
 * Constraints (the layout hooks and the kernel rows refuse anything else): K a multiple of 32 (the
 * block), N a multiple of 16 (the tile). Pure host/device-neutral C++, no HIP include.
 */
#pragma once
#include <cstdint>
#include <cstring>

namespace halo_q80 {

constexpr int kBlock = 32;       /* K values per scale */
constexpr int kTileRows = 16;    /* weight rows per tile */
constexpr int kTileBytes = kTileRows * kBlock;
constexpr int64_t kMaxElems = int64_t(1) << 40;   /* N*K bound: far past any tensor, keeps all offset arithmetic in int64 */

/* Is (N, K) a shape the tiles can address? */
inline bool shape_ok(int64_t N, int64_t K) {
    if (N <= 0 || K <= 0) return false;
    if ((N % kTileRows) || (K % kBlock)) return false;
    if (N > kMaxElems / K) return false;
    return true;
}
inline int64_t code_bytes(int64_t N, int64_t K) { return N * K; }
inline int64_t scale_bytes(int64_t N, int64_t K) { return N * (K / kBlock) * 2; }

/* Offsets of element (n, k) / scale (n, kb) in the tiled planes. */
inline int64_t code_off(int64_t n, int64_t k, int64_t K) {
    const int64_t KB = K / kBlock;
    return ((n / kTileRows) * KB + k / kBlock) * kTileBytes + (n % kTileRows) * kBlock + k % kBlock;
}
inline int64_t scale_idx(int64_t n, int64_t kb, int64_t K) {
    return ((n / kTileRows) * (K / kBlock) + kb) * kTileRows + n % kTileRows;
}

/* f16 -> f32, exact (every f16 is an f32), including subnormals, inf and NaN. */
inline float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, u;
    if (e == 0) {
        if (m == 0) u = s;
        else {                                   /* subnormal: renormalise */
            e = 1;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            m &= 0x3FFu;
            u = s | ((e + 112u) << 23) | (m << 13);
        }
    } else if (e == 31) u = s | 0x7F800000u | (m << 13);
    else u = s | ((e + 112u) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/* ggml's dequantize_row_q8_0 for one value: d * q in f32. */
inline float dequant(uint16_t d, int8_t q) { return f16_to_f32(d) * (float)q; }

}  /* namespace halo_q80 */
