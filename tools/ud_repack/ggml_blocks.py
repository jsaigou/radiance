"""ggml_blocks -- llama.cpp's block formats as numpy arrays: dequantisation, and the split of each
block into the planes a GPU layout is built from.

Everything here is CPU-only and dependency-free apart from numpy. The dequantisers are written from
llama.cpp's dequantize_row_<type> (ggml-quants.c) with the same f32 operation order; ud_repack's
tests also compare them with the `gguf` python package's independent implementation.

A "block array" is uint8 of shape (nb, bytes_per_block). Dequantised output is float32 (nb, qk).
"""
import re
import numpy as np

KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.int8)

# name -> (elements per block, bytes per block)
BLOCK = {
    'Q8_0': (32, 34), 'Q4_K': (256, 144), 'Q5_K': (256, 176), 'Q6_K': (256, 210),
    'Q5_0': (32, 22), 'IQ4_NL': (32, 18), 'IQ4_XS': (256, 136),
    'Q2_0': (64, 18), 'IQ3_XXS': (256, 98), 'IQ2_S': (256, 82),
}

# llama.cpp's ksigns_iq2xs, computed: a 7-bit index's eight sign bits, the eighth equal to the
# index's parity (the grid stores seven sign bits; the eighth is implied).
KSIGNS_IQ2XS = np.array([i | (((bin(i).count('1')) & 1) << 7) for i in range(128)], np.uint8)


def _ksign(v):
    """ksigns_iq2xs for every 7-bit index in v (n,)."""
    v = np.asarray(v, np.uint8)
    return (v | ((np.bitwise_xor.reduce(
        np.stack([(v >> b) & 1 for b in range(7)], axis=0).astype(np.uint8), axis=0) & 1) << 7)).astype(np.uint8)


def _f16(b, lo):
    """The f16 at byte offset lo of every block, as float32 (n,)."""
    return np.ascontiguousarray(b[:, lo:lo + 2]).view('<f2').reshape(-1).astype(np.float32)


# ------------------------------------------------------------------ Q4_K / Q5_K scale+min
def k4_scale_min(s12):
    """llama.cpp get_scale_min_k4 for all eight sub-blocks: (n,12) uint8 -> sc (n,8), mm (n,8) uint8."""
    q = s12.astype(np.uint8)
    sc = np.empty((q.shape[0], 8), np.uint8)
    mm = np.empty_like(sc)
    for j in range(4):
        sc[:, j] = q[:, j] & 63
        mm[:, j] = q[:, j + 4] & 63
    for j in range(4, 8):
        sc[:, j] = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
        mm[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return sc, mm


def k4_pack_scale_min(sc, mm):
    """Inverse of k4_scale_min."""
    n = sc.shape[0]
    q = np.zeros((n, 12), np.uint8)
    for j in range(4):
        q[:, j] = (sc[:, j] & 63) | ((sc[:, j + 4] >> 4) << 6)
        q[:, j + 4] = (mm[:, j] & 63) | ((mm[:, j + 4] >> 4) << 6)
        q[:, j + 8] = (sc[:, j + 4] & 0xF) | ((mm[:, j + 4] & 0xF) << 4)
    return q


# ------------------------------------------------------------------ plane splits (blocks -> planes)
# Each returns a dict of arrays whose first axis is the block. 'codes' is (n, qk) in VALUE ORDER
# (element i of the block at column i), so a layout can permute it freely.
def split_q8_0(b):
    return dict(codes=np.ascontiguousarray(b[:, 2:34]).view(np.int8), d=np.ascontiguousarray(b[:, 0:2]).view('<f2').reshape(-1))


def split_q4_K(b):
    n = b.shape[0]
    sc, mm = k4_scale_min(b[:, 4:16])
    qs = b[:, 16:144].reshape(n, 4, 32)
    codes = np.stack([qs & 15, qs >> 4], axis=2).reshape(n, 256)   # chunk c: low nibbles = 64c+l, high = 64c+32+l
    return dict(codes=codes.astype(np.uint8), sc=sc, mm=mm,
                d=np.ascontiguousarray(b[:, 0:2]).view('<f2').reshape(-1),
                dmin=np.ascontiguousarray(b[:, 2:4]).view('<f2').reshape(-1))


def split_q5_K(b):
    n = b.shape[0]
    sc, mm = k4_scale_min(b[:, 4:16])
    qh = b[:, 16:48]                                      # (n,32)
    qs = b[:, 48:176].reshape(n, 4, 32)
    nib = np.stack([qs & 15, qs >> 4], axis=2)            # (n,4,2,32)
    shift = (2 * np.arange(4)[:, None] + np.arange(2)[None, :]).astype(np.uint8)   # bit 2c+half
    hb = (qh[:, None, None, :] >> shift[None, :, :, None]) & 1
    codes = (nib | (hb << 4)).reshape(n, 256)
    return dict(codes=codes.astype(np.uint8), sc=sc, mm=mm,
                d=np.ascontiguousarray(b[:, 0:2]).view('<f2').reshape(-1),
                dmin=np.ascontiguousarray(b[:, 2:4]).view('<f2').reshape(-1))


def split_q6_K(b):
    n = b.shape[0]
    ql = b[:, 0:128].reshape(n, 2, 2, 32)                 # (n, half, 0/1 -> ql[l] / ql[l+32], l)
    qh = b[:, 128:192].reshape(n, 2, 32)
    q1 = (ql[:, :, 0] & 0xF) | (((qh >> 0) & 3) << 4)
    q2 = (ql[:, :, 1] & 0xF) | (((qh >> 2) & 3) << 4)
    q3 = (ql[:, :, 0] >> 4) | (((qh >> 4) & 3) << 4)
    q4 = (ql[:, :, 1] >> 4) | (((qh >> 6) & 3) << 4)
    codes = np.stack([q1, q2, q3, q4], axis=2).reshape(n, 256)   # value order: half, group, l
    sc = np.ascontiguousarray(b[:, 192:208]).view(np.int8)       # (n,16): sub-block j of 16 values
    return dict(codes=codes.astype(np.uint8), sc=sc, d=np.ascontiguousarray(b[:, 208:210]).view('<f2').reshape(-1))


def split_q5_0(b):
    n = b.shape[0]
    qh = np.ascontiguousarray(b[:, 2:6]).view('<u4').reshape(-1)
    qs = b[:, 6:22]
    j = np.arange(16)
    lo = (qs & 15) | (((qh[:, None] >> j) << 4) & 0x10).astype(np.uint8)
    hi = (qs >> 4) | (((qh[:, None] >> (j + 12))) & 0x10).astype(np.uint8)
    return dict(codes=np.concatenate([lo, hi], axis=1).astype(np.uint8),
                d=np.ascontiguousarray(b[:, 0:2]).view('<f2').reshape(-1))


def split_iq4_nl(b):
    qs = b[:, 2:18]
    return dict(codes=np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.uint8),
                d=np.ascontiguousarray(b[:, 0:2]).view('<f2').reshape(-1))


def split_q2_0(b):
    # d f16 @0; qs[16] @2, 2 bits a value (byte j/4, bits (j%4)*2), value q -> (q-1)*d
    qs = b[:, 2:18]
    codes = np.empty((b.shape[0], 64), np.uint8)
    for j in range(64):
        codes[:, j] = (qs[:, j // 4] >> ((j % 4) * 2)) & 3
    return dict(codes=codes, d=np.ascontiguousarray(b[:, 0:2]).view('<f2').reshape(-1))


def split_iq4_xs(b):
    n = b.shape[0]
    sh = np.ascontiguousarray(b[:, 2:4]).view('<u2').reshape(-1)
    sl = b[:, 4:8]
    ib = np.arange(8)
    ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh[:, None] >> (2 * ib)) & 3) << 4)
    qs = b[:, 8:136].reshape(n, 8, 16)
    codes = np.concatenate([qs & 15, qs >> 4], axis=2).reshape(n, 256)
    return dict(codes=codes.astype(np.uint8), ls=ls.astype(np.uint8),
                d=np.ascontiguousarray(b[:, 0:2]).view('<f2').reshape(-1))


def split_iq3_xxs(b):
    # d f16 @0; qs[64] codes @2 (one 8-bit index per four values); scales_and_signs[32] @66:
    # for each 32 values a uint32 whose low 28 bits are four 7-bit sign indexes and top 4 bits the
    # scale ls (the scale is 2*ls+1). The sign plane is the expanded ksigns byte, one per 8 values.
    n = b.shape[0]
    codes = np.ascontiguousarray(b[:, 2:66])
    aux = np.ascontiguousarray(b[:, 66:98]).view('<u4').reshape(n, 8)
    signs = np.empty((n, 32), np.uint8)
    scales = np.empty((n, 8), np.uint8)
    for ib in range(8):
        a = aux[:, ib]
        for l in range(4):
            signs[:, 4 * ib + l] = _ksign((a >> (7 * l)) & 127)
        scales[:, ib] = (2 * (a >> 28) + 1).astype(np.uint8)
    return dict(codes=codes, signs=signs, scale=scales,
                d=np.ascontiguousarray(b[:, 0:2]).view('<f2').reshape(-1))


def split_iq2_s(b):
    # d f16 @0; qs[64] @2: the first 32 bytes are four 8-bit code lows per 32 values, the next 32
    # are the signs (one byte per 8 values); qh[8] @66 the two high code bits per l; scales[8] @74
    # the two 4-bit ls per 32 values, low nibble first. The codes plane is u10, four per 32.
    n = b.shape[0]
    qs = b[:, 2:66]
    qh = b[:, 66:74]
    scales_in = b[:, 74:82]
    bits = np.zeros((n, 32, 10), np.uint8)
    for ib in range(8):
        for l in range(4):
            lo = qs[:, 4 * ib + l].astype(np.uint16)
            hi = ((qh[:, ib].astype(np.uint16) << (8 - 2 * l)) & 0x300)
            bits[:, 4 * ib + l, :] = ((lo | hi)[:, None] >> np.arange(10, dtype=np.uint16)) & 1
    codes = np.packbits(bits.reshape(n, 320), axis=1, bitorder='little')
    signs = np.ascontiguousarray(qs[:, 32:64])
    scales = np.empty((n, 16), np.uint8)
    for ib in range(8):
        scales[:, 2 * ib] = (2 * (scales_in[:, ib] & 0xf) + 1).astype(np.uint8)
        scales[:, 2 * ib + 1] = (2 * (scales_in[:, ib] >> 4) + 1).astype(np.uint8)
    return dict(codes=codes, signs=signs, scale=scales,
                d=np.ascontiguousarray(b[:, 0:2]).view('<f2').reshape(-1))


SPLIT = {'Q8_0': split_q8_0, 'Q4_K': split_q4_K, 'Q5_K': split_q5_K, 'Q6_K': split_q6_K,
         'Q5_0': split_q5_0, 'IQ4_NL': split_iq4_nl, 'IQ4_XS': split_iq4_xs, 'Q2_0': split_q2_0,
         'IQ3_XXS': split_iq3_xxs, 'IQ2_S': split_iq2_s}


# ------------------------------------------------------------------ plane decoders (the kernel's arithmetic)
# These take the PLANES, not the bytes. They are what a GPU kernel computes before it rounds to the
# matrix unit's input type; every product is exact in f32 (see the margin notes) so there is no
# order ambiguity, and the single rounding is the min subtraction of Q4_K/Q5_K.
f32 = np.float32


def decode_q8_0(p):
    return p['d'].astype(f32)[:, None] * p['codes'].astype(f32)


def decode_q4_K(p):          # also Q5_K: the code is 4 or 5 bits wide, the arithmetic is the same
    n = p['codes'].shape[0]
    d1 = p['d'].astype(f32)[:, None] * p['sc'].astype(f32)          # 11+6 bits, exact
    m1 = p['dmin'].astype(f32)[:, None] * p['mm'].astype(f32)
    q = p['codes'].reshape(n, 8, 32).astype(f32)                    # q<=31: 17+5 bits, exact
    return (d1[:, :, None] * q - m1[:, :, None]).reshape(n, 256)


decode_q5_K = decode_q4_K


def decode_q6_K(p):
    n = p['codes'].shape[0]
    ds = p['d'].astype(f32)[:, None] * p['sc'].astype(f32)          # 11+7 bits, exact
    q = p['codes'].reshape(n, 16, 16).astype(f32) - f32(32)
    return (ds[:, :, None] * q).reshape(n, 256)


def decode_q5_0(p):
    return p['d'].astype(f32)[:, None] * (p['codes'].astype(f32) - f32(16))


def decode_q2_0(p):
    return p['d'].astype(f32)[:, None] * (p['codes'].astype(f32) - f32(1))


def decode_iq4_nl(p):
    return p['d'].astype(f32)[:, None] * KVALUES_IQ4NL[p['codes']].astype(f32)


def decode_iq4_xs(p):
    n = p['codes'].shape[0]
    dl = p['d'].astype(f32)[:, None] * (p['ls'].astype(f32) - f32(32))
    return (dl[:, :, None] * KVALUES_IQ4NL[p['codes']].astype(f32).reshape(n, 8, 32)).reshape(n, 256)


def _grid_rows(n, name):
    """The grid table's rows as (n, width) uint8, lowest byte first, from libhalo's copy of
    llama.cpp's tables (halo_moe_grids.h)."""
    if name == 'iq3xxs':
        v = np.array([int(x, 0) for x in _grid_values('HALO_IQ3XXS_GRID_VALUES')], np.uint32)
        return v.view(np.uint8).reshape(-1, 4)
    v = np.array([int(x, 0) for x in _grid_values('HALO_IQ2S_GRID_VALUES')], np.uint64)
    return v.view(np.uint8).reshape(-1, 8)


def _grid_values(macro):
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, '..', '..', 'libhalo', 'halo_moe_grids.h')
    text = open(path).read()
    m = re.search(r'#define\s+' + macro + r'\s+(.*?)(?=\n\s*#define|\n\s*(?:static|__|#endif)|\Z)',
                  text, re.S)
    if not m:
        raise RuntimeError('halo_moe_grids.h has no ' + macro)
    return re.findall(r'0x[0-9a-fA-F]+', m.group(1))


GRID_IQ3XXS = _grid_rows(256, 'iq3xxs')
GRID_IQ2S = _grid_rows(1024, 'iq2s')


def decode_iq3_xxs(p):
    n = p['codes'].shape[0]
    y = np.empty((n, 256), f32)
    for ib in range(8):
        c = p['codes'][:, 8 * ib:8 * ib + 8]               # eight 4-value grid rows
        sg = p['signs'][:, 4 * ib:4 * ib + 4]              # four sign bytes, one per 8 values
        s = p['d'].astype(f32)[:, None] * p['scale'][:, ib].astype(f32).astype(f32)[:, None] * f32(0.25)
        for l in range(4):
            for half in range(2):
                g = GRID_IQ3XXS[c[:, 2 * l + half]].astype(f32)
                bits = (sg[:, l:l + 1] >> np.arange(4 * half, 4 * half + 4, dtype=np.uint8)) & 1
                g = np.where(bits.astype(bool), -g, g)
                y[:, 32 * ib + 8 * l + 4 * half:32 * ib + 8 * l + 4 * half + 4] = s * g
    return y


def decode_iq2_s(p):
    n = p['codes'].shape[0]
    y = np.empty((n, 256), f32)
    bits = np.unpackbits(p['codes'], axis=1, bitorder='little')[:, :320].reshape(n, 32, 10)
    codes = (bits << np.arange(10, dtype=np.uint16)).sum(axis=2)
    for ib in range(8):
        for l in range(4):
            g = GRID_IQ2S[codes[:, 4 * ib + l]].astype(f32)
            sb = p['signs'][:, 4 * ib + l:4 * ib + l + 1]
            b = (sb >> np.arange(8, dtype=np.uint8)) & 1
            g = np.where(b.astype(bool), -g, g)
            s = (p['d'].astype(f32) * p['scale'][:, 2 * ib + l // 2].astype(f32) * f32(0.125))[:, None]
            y[:, 32 * ib + 8 * l:32 * ib + 8 * l + 8] = s * g
    return y


DECODE = {'Q8_0': decode_q8_0, 'Q4_K': decode_q4_K, 'Q5_K': decode_q5_K, 'Q6_K': decode_q6_K,
          'Q5_0': decode_q5_0, 'IQ4_NL': decode_iq4_nl, 'IQ4_XS': decode_iq4_xs, 'Q2_0': decode_q2_0,
          'IQ3_XXS': decode_iq3_xxs, 'IQ2_S': decode_iq2_s}


# ------------------------------------------------------------------ reference dequantisers (llama.cpp order, from bytes)
def dequant_ref(tname, b):
    """llama.cpp dequantize_row_<type> transcribed directly from the block bytes, with loops over
    the same indices; independent of the SPLIT/DECODE path above. Slow-ish but vectorised over blocks."""
    n = b.shape[0]
    if tname == 'Q8_0':
        return _f16(b, 0)[:, None] * b[:, 2:34].view(np.int8).astype(f32)
    if tname in ('Q4_K', 'Q5_K'):
        d = _f16(b, 0); mn = _f16(b, 2)
        s12 = b[:, 4:16]
        if tname == 'Q4_K':
            ql = b[:, 16:144]; qh = None
        else:
            qh = b[:, 16:48]; ql = b[:, 48:176]
        y = np.empty((n, 256), f32)
        sc, mm = k4_scale_min(s12)
        u1, u2 = 1, 2
        for c in range(4):
            q = ql[:, 32 * c:32 * c + 32]
            for half in (0, 1):
                is_ = 2 * c + half
                d1 = d * sc[:, is_].astype(f32); m1 = mn * mm[:, is_].astype(f32)
                nib = (q & 15) if half == 0 else (q >> 4)
                v = nib.astype(f32)
                if qh is not None:
                    bit = (u1 if half == 0 else u2)
                    v = v + ((qh & bit) != 0).astype(f32) * f32(16)
                y[:, 64 * c + 32 * half:64 * c + 32 * half + 32] = d1[:, None] * v - m1[:, None]
            u1 <<= 2; u2 <<= 2
        return y
    if tname == 'Q6_K':
        d = _f16(b, 208)
        ql = b[:, 0:128]; qh = b[:, 128:192]; sc = b[:, 192:208].view(np.int8)
        y = np.empty((n, 256), f32)
        for h in range(2):
            qlh = ql[:, 64 * h:64 * h + 64]; qhh = qh[:, 32 * h:32 * h + 32]; sch = sc[:, 8 * h:8 * h + 8]
            l = np.arange(32); is_ = l // 16
            q1 = ((qlh[:, l] & 0xF) | (((qhh[:, l] >> 0) & 3) << 4)).astype(np.int32) - 32
            q2 = ((qlh[:, l + 32] & 0xF) | (((qhh[:, l] >> 2) & 3) << 4)).astype(np.int32) - 32
            q3 = ((qlh[:, l] >> 4) | (((qhh[:, l] >> 4) & 3) << 4)).astype(np.int32) - 32
            q4 = ((qlh[:, l + 32] >> 4) | (((qhh[:, l] >> 6) & 3) << 4)).astype(np.int32) - 32
            for g, q in enumerate((q1, q2, q3, q4)):
                y[:, 128 * h + 32 * g:128 * h + 32 * g + 32] = (d[:, None] * sch[:, 2 * g + is_].astype(f32)) * q.astype(f32)
        return y
    if tname == 'Q5_0':
        d = _f16(b, 0); qh = np.ascontiguousarray(b[:, 2:6]).view('<u4').reshape(-1); qs = b[:, 6:22]
        y = np.empty((n, 32), f32)
        for j in range(16):
            x0 = ((qs[:, j] & 0xF) | (((qh >> j) << 4) & 0x10)).astype(np.int32) - 16
            x1 = ((qs[:, j] >> 4) | (((qh >> (j + 12))) & 0x10)).astype(np.int32) - 16
            y[:, j] = x0.astype(f32) * d; y[:, j + 16] = x1.astype(f32) * d
        return y
    if tname == 'IQ4_NL':
        d = _f16(b, 0); qs = b[:, 2:18]
        y = np.empty((n, 32), f32)
        y[:, :16] = d[:, None] * KVALUES_IQ4NL[qs & 15].astype(f32)
        y[:, 16:] = d[:, None] * KVALUES_IQ4NL[qs >> 4].astype(f32)
        return y
    if tname == 'IQ4_XS':
        d = _f16(b, 0); sh = np.ascontiguousarray(b[:, 2:4]).view('<u2').reshape(-1); sl = b[:, 4:8]; qs = b[:, 8:136]
        y = np.empty((n, 256), f32)
        for ib in range(8):
            ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF).astype(np.int32) | (((sh >> (2 * ib)) & 3).astype(np.int32) << 4)
            dl = d * (ls - 32).astype(f32)
            q = qs[:, 16 * ib:16 * ib + 16]
            y[:, 32 * ib:32 * ib + 16] = dl[:, None] * KVALUES_IQ4NL[q & 15].astype(f32)
            y[:, 32 * ib + 16:32 * ib + 32] = dl[:, None] * KVALUES_IQ4NL[q >> 4].astype(f32)
        return y
    if tname == 'Q2_0':
        d = _f16(b, 0); qs = b[:, 2:18]
        y = np.empty((n, 64), f32)
        for j in range(64):
            q = ((qs[:, j // 4] >> ((j % 4) * 2)) & 3).astype(np.int32) - 1
            y[:, j] = d * q.astype(f32)
        return y
    if tname == 'IQ3_XXS':
        d = _f16(b, 0); qs = b[:, 2:66]; aux = np.ascontiguousarray(b[:, 66:98]).view('<u4').reshape(n, 8)
        y = np.empty((n, 256), f32)
        for ib in range(8):
            a = aux[:, ib]
            db = d * (f32(0.5) + (a >> 28).astype(f32)) * f32(0.5)
            for l in range(4):
                sg = KSIGNS_IQ2XS[((a >> (7 * l)) & 127)]
                for half in range(2):
                    g = GRID_IQ3XXS[qs[:, 8 * ib + 2 * l + half]].astype(f32)
                    bmask = ((sg[:, None] >> np.arange(4 * half, 4 * half + 4)) & 1).astype(bool)
                    v = np.where(bmask, -g, g)
                    y[:, 32 * ib + 8 * l + 4 * half:32 * ib + 8 * l + 4 * half + 4] = db[:, None] * v
        return y
    if tname == 'IQ2_S':
        d = _f16(b, 0); qs = b[:, 2:66]; qh = b[:, 66:74]; sc = b[:, 74:82]
        signs = qs[:, 32:64]
        y = np.empty((n, 256), f32)
        for ib in range(8):
            db = [d * (f32(0.5) + (sc[:, ib] & 0xf).astype(f32)) * f32(0.25),
                  d * (f32(0.5) + (sc[:, ib] >> 4).astype(f32)) * f32(0.25)]
            for l in range(4):
                idx = qs[:, 4 * ib + l].astype(np.uint16) | ((qh[:, ib].astype(np.uint16) << (8 - 2 * l)) & 0x300)
                g = GRID_IQ2S[idx].astype(f32)
                bmask = ((signs[:, 4 * ib + l:4 * ib + l + 1] >> np.arange(8)) & 1).astype(bool)
                v = np.where(bmask, -g, g)
                y[:, 32 * ib + 8 * l:32 * ib + 8 * l + 8] = db[l // 2][:, None] * v
        return y
    raise KeyError(tname)


# ------------------------------------------------------------------ plane merges (planes -> blocks), the inverse of SPLIT
def _h(a):
    return np.ascontiguousarray(a.astype('<f2')).view(np.uint8).reshape(-1, 2)


def merge_q8_0(p):
    return np.concatenate([_h(p['d']), np.ascontiguousarray(p['codes']).view(np.uint8)], axis=1)


def _merge_k45(p, five):
    n = p['codes'].shape[0]
    c = p['codes'].reshape(n, 4, 2, 32).astype(np.uint8)
    qs = ((c[:, :, 0] & 15) | ((c[:, :, 1] & 15) << 4)).reshape(n, 128)
    parts = [_h(p['d']), _h(p['dmin']), k4_pack_scale_min(p['sc'], p['mm'])]
    if five:
        shift = (2 * np.arange(4)[:, None] + np.arange(2)[None, :]).astype(np.uint8)
        hb = (c >> 4) & 1
        qh = np.bitwise_or.reduce((hb << shift[None, :, :, None]).reshape(n, 8, 32), axis=1).astype(np.uint8)
        parts.append(qh)
    parts.append(qs)
    return np.concatenate(parts, axis=1)


def merge_q4_K(p):
    return _merge_k45(p, False)


def merge_q5_K(p):
    return _merge_k45(p, True)


def merge_q6_K(p):
    n = p['codes'].shape[0]
    c = p['codes'].reshape(n, 2, 4, 32).astype(np.uint8)
    q1, q2, q3, q4 = c[:, :, 0], c[:, :, 1], c[:, :, 2], c[:, :, 3]
    ql = np.stack([(q1 & 0xF) | ((q3 & 0xF) << 4), (q2 & 0xF) | ((q4 & 0xF) << 4)], axis=2).reshape(n, 128)
    qh = ((q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)).reshape(n, 64).astype(np.uint8)
    return np.concatenate([ql, qh, np.ascontiguousarray(p['sc']).view(np.uint8), _h(p['d'])], axis=1)


def merge_q5_0(p):
    c = p['codes'].astype(np.uint32)
    lo, hi = c[:, :16], c[:, 16:]
    qs = ((lo & 15) | ((hi & 15) << 4)).astype(np.uint8)
    j = np.arange(16, dtype=np.uint32)
    qh = (np.bitwise_or.reduce(((lo >> 4) & 1) << j, axis=1) | np.bitwise_or.reduce(((hi >> 4) & 1) << (j + 16), axis=1)).astype('<u4')
    return np.concatenate([_h(p['d']), qh.view(np.uint8).reshape(-1, 4), qs], axis=1)


def merge_q2_0(p):
    c = p['codes'].astype(np.uint32)
    qs = np.zeros((c.shape[0], 16), np.uint8)
    for j in range(64):
        qs[:, j // 4] |= ((c[:, j] & 3) << ((j % 4) * 2)).astype(np.uint8)
    return np.concatenate([_h(p['d']), qs], axis=1)


def merge_iq4_nl(p):
    c = p['codes'].astype(np.uint8)
    return np.concatenate([_h(p['d']), (c[:, :16] & 15) | ((c[:, 16:] & 15) << 4)], axis=1)


def merge_iq4_xs(p):
    n = p['codes'].shape[0]
    c = p['codes'].reshape(n, 8, 2, 16).astype(np.uint8)
    qs = ((c[:, :, 0] & 15) | ((c[:, :, 1] & 15) << 4)).reshape(n, 128)
    ls = p['ls'].astype(np.uint32)
    sl = np.zeros((n, 4), np.uint8)
    for ib in range(8):
        sl[:, ib // 2] |= ((ls[:, ib] & 0xF) << (4 * (ib % 2))).astype(np.uint8)
    sh = np.bitwise_or.reduce(((ls >> 4) & 3) << (2 * np.arange(8, dtype=np.uint32)), axis=1).astype('<u2')
    return np.concatenate([_h(p['d']), sh.view(np.uint8).reshape(-1, 2), sl, qs], axis=1)


def merge_iq3_xxs(p):
    n = p['codes'].shape[0]
    aux = np.zeros((n, 8), np.uint32)
    for ib in range(8):
        for l in range(4):
            # the sign plane holds ksigns_iq2xs's expanded byte; the 7-bit index is its low bits
            aux[:, ib] |= (p['signs'][:, 4 * ib + l].astype(np.uint32) & 127) << (7 * l)
        aux[:, ib] |= ((p['scale'][:, ib].astype(np.uint32) - 1) // 2) << 28
    return np.concatenate([_h(p['d']), np.ascontiguousarray(p['codes']), aux.view(np.uint8)], axis=1)


def merge_iq2_s(p):
    n = p['codes'].shape[0]
    bits = np.unpackbits(p['codes'], axis=1, bitorder='little')[:, :320].reshape(n, 32, 10)
    codes = (bits << np.arange(10, dtype=np.uint16)).sum(axis=2).astype(np.uint32)
    qs = np.zeros((n, 64), np.uint8)
    qh = np.zeros((n, 8), np.uint8)
    for ib in range(8):
        for l in range(4):
            c = codes[:, 4 * ib + l]
            qs[:, 4 * ib + l] = (c & 0xFF).astype(np.uint8)
            qh[:, ib] |= (((c >> 8) & 3) << (2 * l)).astype(np.uint8)
    qs[:, 32:64] = p['signs']
    sc = np.zeros((n, 8), np.uint8)
    for ib in range(8):
        lo = (p['scale'][:, 2 * ib].astype(np.uint32) - 1) // 2
        hi = (p['scale'][:, 2 * ib + 1].astype(np.uint32) - 1) // 2
        sc[:, ib] = (lo | (hi << 4)).astype(np.uint8)
    return np.concatenate([_h(p['d']), qs, qh, sc], axis=1)


MERGE = {'Q8_0': merge_q8_0, 'Q4_K': merge_q4_K, 'Q5_K': merge_q5_K, 'Q6_K': merge_q6_K,
         'Q5_0': merge_q5_0, 'IQ4_NL': merge_iq4_nl, 'IQ4_XS': merge_iq4_xs, 'Q2_0': merge_q2_0,
         'IQ3_XXS': merge_iq3_xxs, 'IQ2_S': merge_iq2_s}
