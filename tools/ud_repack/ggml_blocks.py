"""ggml_blocks -- llama.cpp's block formats as numpy arrays: dequantisation, and the split of each
block into the planes a GPU layout is built from.

Everything here is CPU-only and dependency-free apart from numpy. The dequantisers are written from
llama.cpp's dequantize_row_<type> (ggml-quants.c) with the same f32 operation order; ud_repack's
tests also compare them with the `gguf` python package's independent implementation.

A "block array" is uint8 of shape (nb, bytes_per_block). Dequantised output is float32 (nb, qk).
"""
import numpy as np

KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.int8)

# name -> (elements per block, bytes per block)
BLOCK = {
    'Q8_0': (32, 34), 'Q4_K': (256, 144), 'Q5_K': (256, 176), 'Q6_K': (256, 210),
    'Q5_0': (32, 22), 'IQ4_NL': (32, 18), 'IQ4_XS': (256, 136),
    'Q2_0': (64, 18),
}


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


SPLIT = {'Q8_0': split_q8_0, 'Q4_K': split_q4_K, 'Q5_K': split_q5_K, 'Q6_K': split_q6_K,
         'Q5_0': split_q5_0, 'IQ4_NL': split_iq4_nl, 'IQ4_XS': split_iq4_xs, 'Q2_0': split_q2_0}


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


DECODE = {'Q8_0': decode_q8_0, 'Q4_K': decode_q4_K, 'Q5_K': decode_q5_K, 'Q6_K': decode_q6_K,
          'Q5_0': decode_q5_0, 'IQ4_NL': decode_iq4_nl, 'IQ4_XS': decode_iq4_xs, 'Q2_0': decode_q2_0}


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


MERGE = {'Q8_0': merge_q8_0, 'Q4_K': merge_q4_K, 'Q5_K': merge_q5_K, 'Q6_K': merge_q6_K,
         'Q5_0': merge_q5_0, 'IQ4_NL': merge_iq4_nl, 'IQ4_XS': merge_iq4_xs, 'Q2_0': merge_q2_0}
