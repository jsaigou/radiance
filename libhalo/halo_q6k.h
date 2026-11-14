/* halo_q6k.h -- ggml Q6_K as libhalo stores and reads it, for dense weights.
 *
 * Q6_K is llama.cpp's block_q6_K: 256 weights in one block. The block holds ql u8[128] and
 * qh u8[64] (the 6-bit codes, q - 32 in two's complement), scales i8[16] (one per 16 values) and
 * d f16 (one per block). The container holds it as three canonical planes
 * (core/format/ggml_planes.h split_q6_K, libquant/lq_ggml.cpp encoding_of):
 *
 *   codes    i6 [N][K]       q - 32, packed 6 bits per value, lowest bits first
 *   scale    i8 [N][K/16]    sc, one per 16 values
 *   scale.1  f16 [N][K/256]  d, one per 256 values
 *
 * The dequantised weight is w = (d * sc) * (q - 32). d has 11 significant bits and sc 7, so
 * d * sc is exact in f32 (18 bits); times the code (6 bits, |c| <= 32) the 24-bit f32
 * significand still holds it, so w is bit for bit (float)d * (float)sc * (float)code. Only the
 * ORDER in which a kernel sums products can differ from llama.cpp (which also quantises the
 * activation; libhalo does not).
 *
 * The stored form ("halo.q6k.*") is one for both kernels. The 6-bit code bytes are only moved,
 * never expanded: a TILE is 16 weight rows by one 64-wide K pass, 16 rows * 48 bytes of codes,
 * and the four 16-value sub-blocks of a row stay in order inside the row's 48 bytes. A wave
 * reading a tile with 12 bytes a lane reads 768 contiguous bytes, and a WMMA B fragment (one
 * row's 16 consecutive k per lane) is one 12-byte load:
 *
 *   codes    i6 [N][K]        tile (rt, pass), row r, sub s: byte ((rt*NP + pass)*16 + r)*48 + 12*s
 *   scale    i8 [N/16][K/16][16]   byte (rt*(K/16) + kb)*16 + r
 *   scale.1  f16 [N/16][K/256][16] half (rt*(K/256) + g)*16 + r
 *
 * where NP = K/64. Constraints (the layout hooks and the kernel rows refuse anything else): K a
 * multiple of 256 (the super-scale block), N a multiple of 16 (the tile). Pure host/device-neutral
 * C++, no HIP include.
 */
#pragma once
#include <cstdint>
#include <cstring>

namespace halo_q6k {

constexpr int kBlock = 256;      /* K values per f16 super-scale */
constexpr int kSub = 16;         /* K values per int8 scale */
constexpr int kPass = 64;        /* K values per stored code pass (four sub-blocks) */
constexpr int kTileRows = 16;    /* weight rows per tile */
constexpr int kSubBytes = kSub * 6 / 8;                  /* 12 bytes of codes per sub-block row */
constexpr int kPassBytes = kTileRows * (kPass * 6 / 8);  /* 768 bytes per (tile, pass) */
constexpr int64_t kMaxElems = int64_t(1) << 40;   /* N*K bound: far past any tensor, keeps all offset arithmetic in int64 */

/* Is (N, K) a shape the tiles can address? */
inline bool shape_ok(int64_t N, int64_t K) {
    if (N <= 0 || K <= 0) return false;
    if ((N % kTileRows) || (K % kBlock)) return false;
    if (N > kMaxElems / K) return false;
    return true;
}
inline int64_t code_bytes(int64_t N, int64_t K) { return N * (K * 6 / 8); }
inline int64_t scale_bytes(int64_t N, int64_t K) { return N * (K / kSub); }
inline int64_t d_bytes(int64_t N, int64_t K) { return N * (K / kBlock) * 2; }

/* Byte offset of element (n, k) in the tiled codes plane. */
inline int64_t code_off(int64_t n, int64_t k, int64_t K) {
    const int64_t NP = K / kPass;
    const int64_t pass = k / kPass, s = (k % kPass) / kSub, i = k % kSub;
    return ((n / kTileRows) * NP + pass) * kPassBytes + (n % kTileRows) * (kPass * 6 / 8) +
           s * kSubBytes + (6 * i) / 8;
}
/* Element index of scale (n, kb) (kb a 16-value sub-block) in the tiled scale plane. */
inline int64_t scale_idx(int64_t n, int64_t kb, int64_t K) {
    return ((n / kTileRows) * (K / kSub) + kb) * kTileRows + n % kTileRows;
}
/* Element index of super-scale (n, g) (g a 256-value block) in the tiled d plane. */
inline int64_t d_idx(int64_t n, int64_t g, int64_t K) {
    return ((n / kTileRows) * (K / kBlock) + g) * kTileRows + n % kTileRows;
}

/* A stored 6-bit code, two's complement: 0..31 stay, 32..63 are -32..-1. */
inline int code6(uint32_t raw) { return (raw & 32u) ? (int)(raw | ~63u) : (int)raw; }

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

/* ggml's dequantize_row_q6_K for one value: (d * sc) * (q - 32), each product exact in f32. */
inline float dequant(uint16_t d, int8_t sc, int code) {
    return (f16_to_f32(d) * (float)sc) * (float)code;
}

}  /* namespace halo_q6k */
