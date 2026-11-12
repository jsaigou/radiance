"""kq_layout -- the proposed GPU-side row layouts that mirror llama.cpp's blocks ("kq" family).

A row of K weights becomes a sequence of bit-stream planes (LSB first, as rad_encoding.h's bit
streams are), each plane covering the whole row, so a kernel streams the code plane with wide loads
and fetches the small scale planes once per 32/256 of K. Nothing is requantised: every plane holds
a ggml block field verbatim, widened or split but never rounded, so merge(unpack(serialize(x))) is
the original GGUF block bytes.

Variants: 'P' keeps the 6-bit K-quant scales packed (the same bits per weight as the GGUF block) and
'A' widens them to a byte (+0.125 bpw for Q4_K/Q5_K, 0.0 for Q6_K/Q8_0, whose scales are already
bytes), the choice being a decode-cost-versus-bandwidth question for the kernel author. The order of
elements inside a plane is the canonical value order; a lane-order permutation for a particular
kernel is a pure index shuffle and does not change what this file proves.
"""
import numpy as np
from ggml_blocks import BLOCK, SPLIT, MERGE

# plane: (name, bits_P, bits_A, elements per block, field in the ggml_blocks split dict, part)
# part: None = whole field, 'lo4' = low four bits, 'hi' = the bits above the low four
SPEC = {
    'Q8_0':   [('codes', 8, 8, 32, 'codes', None), ('d', 16, 16, 1, 'd', None)],
    'Q4_K':   [('codes', 4, 4, 256, 'codes', None), ('sc', 6, 8, 8, 'sc', None), ('mm', 6, 8, 8, 'mm', None),
               ('d', 16, 16, 1, 'd', None), ('dmin', 16, 16, 1, 'dmin', None)],
    'Q5_K':   [('lo4', 4, 4, 256, 'codes', 'lo4'), ('hi1', 1, 1, 256, 'codes', 'hi'), ('sc', 6, 8, 8, 'sc', None),
               ('mm', 6, 8, 8, 'mm', None), ('d', 16, 16, 1, 'd', None), ('dmin', 16, 16, 1, 'dmin', None)],
    'Q6_K':   [('lo4', 4, 4, 256, 'codes', 'lo4'), ('hi2', 2, 2, 256, 'codes', 'hi'), ('sc', 8, 8, 16, 'sc', None),
               ('d', 16, 16, 1, 'd', None)],
    'Q5_0':   [('lo4', 4, 4, 32, 'codes', 'lo4'), ('hi1', 1, 1, 32, 'codes', 'hi'), ('d', 16, 16, 1, 'd', None)],
    'IQ4_NL': [('codes', 4, 4, 32, 'codes', None), ('d', 16, 16, 1, 'd', None)],
    'IQ4_XS': [('codes', 4, 4, 256, 'codes', None), ('ls', 6, 8, 8, 'ls', None), ('d', 16, 16, 1, 'd', None)],
}


def bits_per_weight(tname, variant='P'):
    qk, _ = BLOCK[tname]
    col = 1 if variant == 'P' else 2
    return sum(pl[col] * pl[3] for pl in SPEC[tname]) / qk


def pack_bits(v, bits):
    """(rows, m) unsigned ints -> (rows, ceil(m*bits/8)) uint8, LSB first."""
    rows, m = v.shape
    if bits == 8:
        return v.astype(np.uint8)
    if bits == 16:
        return np.ascontiguousarray(v.astype('<u2')).view(np.uint8).reshape(rows, 2 * m)
    b = ((v.astype(np.uint32)[:, :, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8).reshape(rows, m * bits)
    return np.packbits(b, axis=1, bitorder='little')


def unpack_bits(buf, bits, m):
    rows = buf.shape[0]
    if bits == 8:
        return buf[:, :m].astype(np.uint32)
    if bits == 16:
        return np.ascontiguousarray(buf[:, :2 * m]).view('<u2').astype(np.uint32)
    b = np.unpackbits(buf, axis=1, bitorder='little')[:, :m * bits].reshape(rows, m, bits).astype(np.uint32)
    return (b << np.arange(bits, dtype=np.uint32)).sum(axis=2)


def _raw(field, arr):
    """The field as the unsigned integers a plane stores (int8 scales and Q8_0 codes as two's complement bytes, f16 as bits)."""
    if arr.dtype == np.int8:
        return arr.view(np.uint8).astype(np.uint32)
    if arr.dtype == np.float16:
        return arr.view(np.uint16).astype(np.uint32)
    return arr.astype(np.uint32)


def serialize(tname, planes, rows, variant='P'):
    """planes: SPLIT output for rows*nb blocks in row-major order -> (rows, row_bytes) uint8."""
    col = 1 if variant == 'P' else 2
    out = []
    for name, bp, ba, cnt, field, part in SPEC[tname]:
        bits = bp if variant == 'P' else ba
        v = _raw(field, planes[field])
        if part == 'lo4':
            v = v & 15
        elif part == 'hi':
            v = v >> 4
        out.append(pack_bits(v.reshape(rows, -1), bits))
    return np.concatenate(out, axis=1)


def row_bytes(tname, K, variant='P'):
    qk, _ = BLOCK[tname]
    nb = K // qk
    col = 1 if variant == 'P' else 2
    return sum(-(-(pl[col] * pl[3] * nb) // 8) for pl in SPEC[tname])


def deserialize(tname, buf, K, variant='P'):
    """(rows, row_bytes) -> SPLIT-style planes with first axis rows*nb."""
    qk, _ = BLOCK[tname]
    nb = K // qk
    rows = buf.shape[0]
    pos = 0
    got = {}
    for name, bp, ba, cnt, field, part in SPEC[tname]:
        bits = bp if variant == 'P' else ba
        nbytes = -(-(bits * cnt * nb) // 8)
        v = unpack_bits(np.ascontiguousarray(buf[:, pos:pos + nbytes]), bits, cnt * nb)
        pos += nbytes
        got.setdefault(field, []).append((part, v))
    planes = {}
    for field, parts in got.items():
        v = parts[0][1] if len(parts) == 1 else sum((p[1] << (4 if p[0] == 'hi' else 0)) for p in parts)
        per = [c for (_, _, _, c, f, _) in SPEC[tname] if f == field][0]
        v = v.reshape(rows * nb, per) if per > 1 else v.reshape(rows * nb)
        planes[field] = v
    # restore dtypes
    ref = {'Q8_0': dict(codes=np.int8, d=np.float16), 'Q6_K': dict(sc=np.int8, d=np.float16)}.get(tname, {})
    for k in planes:
        if k in ('d', 'dmin'):
            planes[k] = planes[k].astype(np.uint16).view(np.float16)
        elif ref.get(k) == np.int8:
            planes[k] = planes[k].astype(np.uint8).view(np.int8)
        else:
            planes[k] = planes[k].astype(np.uint8)
    return planes


def plane_bytes(tname, planes, rows, variant='P'):
    """Like serialize, but per plane: {name: (rows, bytes) uint8} -- for comparing plane by plane."""
    out = {}
    for name, bp, ba, cnt, field, part in SPEC[tname]:
        bits = bp if variant == 'P' else ba
        v = _raw(field, planes[field])
        if part == 'lo4': v = v & 15
        elif part == 'hi': v = v >> 4
        out[name] = pack_bits(v.reshape(rows, -1), bits)
    return out
