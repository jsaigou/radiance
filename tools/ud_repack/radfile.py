"""radfile -- a dependency-light (numpy only) reader for the .rad container, written from
abi/rad_format.h and abi/rad_encoding.h, independent of the C++ reader in core/format/radfile.cpp.

It exists so a converted container can be checked by something that shares no code with the writer:
tools/ud_repack/rad_verify.py decodes the planes of a lossless GGUF import with it and compares the
result with the GGUF's own dequantisation.

Layout (little-endian): RadFileHeader, then tables at the offsets it names. See the C++ headers for
the field meanings; every struct size below is asserted against the header's counts at open.
"""
import struct
import numpy as np

HDR = struct.Struct('<II Q 5Q 14Q 8Q')       # magic, version, file_bytes, 5 string offsets, 7 (off,count) pairs, reserved[8]
KV = struct.Struct('<QIIq')                  # key, type, pad, value (int64 / double / stroff share 8 bytes)
ENTRY = struct.Struct('<QQQ II 6q iii I QQQ 2Q')
PLANE = struct.Struct('<QQ')
ENC_SZ = 24 + 24 + 8 + 8 * (24 + 4 + 4 + 16 + 16)
ENCPL = struct.Struct('<24s II 2q 2q')

DT = {1: ('f32', 32), 2: ('f16', 16), 3: ('bf16', 16), 4: ('fp8_e4m3', 8), 5: ('fp8_e5m2', 8), 8: ('i8', 8),
      9: ('u8', 8), 10: ('i16', 16), 11: ('i32', 32), 12: ('i64', 64), 13: ('u32', 32), 14: ('bool', 8),
      16: ('i4', 4), 17: ('i2', 2), 18: ('fp4_e2m1', 4), 19: ('u4', 4), 20: ('u2', 2), 21: ('e8m0', 8),
      22: ('u1', 1), 23: ('i3', 3), 24: ('u3', 3), 25: ('i5', 5), 26: ('u5', 5), 27: ('i6', 6), 28: ('u6', 6),
      29: ('i7', 7), 30: ('u7', 7), 31: ('u9', 9), 32: ('u10', 10), 33: ('u11', 11), 34: ('u12', 12),
      35: ('fp6_e2m3', 6), 36: ('fp6_e3m2', 6), 37: ('u16', 16)}
DT_BY_NAME = {v[0]: k for k, v in DT.items()}


def dt_bits(dt):
    return DT[dt][1]


class Plane:
    def __init__(self, role, dtype, kind, block, extent):
        self.role, self.dtype, self.kind, self.block, self.extent = role, dtype, kind, block, extent

    def dims(self, rows, cols):
        if self.kind == 1:
            return self.extent
        pr = (rows + self.block[0] - 1) // self.block[0] if self.block[0] > 0 else 1
        pc = (cols + self.block[1] - 1) // self.block[1] if self.block[1] > 0 else 1
        return pr, pc

    def __repr__(self):
        return f'{self.role}:{DT[self.dtype][0]}' + (f'[{self.block[0]}x{self.block[1]}]' if self.kind == 0 else f'<table {self.extent}>')


class Encoding:
    def __init__(self, raw):
        self.scheme = raw[0:24].split(b'\0')[0].decode()
        self.transform = raw[24:48].split(b'\0')[0].decode()
        (n, _) = struct.unpack_from('<ii', raw, 48)
        self.planes = []
        for i in range(n):
            role, dtype, kind, b0, b1, e0, e1 = ENCPL.unpack_from(raw, 56 + i * ENCPL.size)
            self.planes.append(Plane(role.split(b'\0')[0].decode(), dtype, kind, (b0, b1), (e0, e1)))

    def name(self):
        if self.scheme == 'plain':
            return DT[self.planes[0].dtype][0]
        return self.scheme + ':' + ','.join(repr(p) for p in self.planes) + (('/' + self.transform) if self.transform else '')


class Entry:
    pass


class RadFile:
    def __init__(self, path):
        self.path = path
        self.mm = np.memmap(path, dtype=np.uint8, mode='r')
        b = self.mm
        h = HDR.unpack_from(b, 0)
        (magic, ver, fbytes, arch, model, quant, recipe, created, str_off, str_bytes, meta_off, meta_n, dir_off, dir_n,
         plane_off, plane_n, enc_off, enc_n, vocab_off, vocab_bytes, prof_off, prof_n, data_off, data_bytes) = h[:24]
        if magic != 0x31444152 or ver != 2:
            raise ValueError(f'{path}: not a RAD2 container (magic {magic:#x}, version {ver})')
        if fbytes != len(b):
            raise ValueError(f'{path}: header says {fbytes} bytes, file has {len(b)}')
        self.strs = bytes(b[str_off:str_off + str_bytes])
        self.arch, self.model, self.quant, self.recipe, self.created_by = (self._s(x) for x in (arch, model, quant, recipe, created))
        self.meta = {}
        for i in range(meta_n):
            k, t, _, v = KV.unpack_from(b, meta_off + i * KV.size)
            if t == 100:
                v = struct.unpack('<d', struct.pack('<q', v))[0]
            elif t != 0 and t != 1 and t != 2:
                pass
            self.meta[self._s(k)] = (t, v)
        self.encs = [Encoding(bytes(b[enc_off + i * ENC_SZ: enc_off + (i + 1) * ENC_SZ])) for i in range(enc_n)]
        self.planes = [PLANE.unpack_from(b, plane_off + i * PLANE.size) for i in range(plane_n)]
        self.entries = []
        self.by_name = {}
        for i in range(dir_n):
            f = ENTRY.unpack_from(b, dir_off + i * ENTRY.size)
            e = Entry()
            (name, quantizer, options, enc, rank) = f[0:5]
            e.name, e.quantizer, e.options = self._s(name), self._s(quantizer), self._s(options)
            e.enc_i, e.rank = enc, rank
            e.shape = tuple(f[5:11][:rank])
            e.layer, e.expert, e.slot = f[11], f[12], f[13]
            e.n_planes, e.plane_first, e.offset, e.bytes = f[14], f[15], f[16], f[17]
            e.enc = self.encs[e.enc_i]
            self.entries.append(e)
            self.by_name[e.name] = e

    def _s(self, off):
        return self.strs[off:self.strs.index(b'\0', off)].decode('utf8', 'replace')

    def plane_bytes(self, e, k):
        off, nbytes = self.planes[e.plane_first + k]
        return self.mm[off:off + nbytes]

    def plane_rows(self, e, k):
        """Plane k as a (prows, row_bytes) uint8 view and its element dims (prows, pcols)."""
        rows = int(np.prod(e.shape[:-1])) if len(e.shape) > 1 else 1
        cols = e.shape[-1] if e.shape else 1
        pl = e.enc.planes[k]
        pr, pc = pl.dims(rows, cols)
        rb = (pc * dt_bits(pl.dtype) + 7) // 8
        return self.plane_bytes(e, k).reshape(pr, rb), pr, pc


def unpack_stream(rows_u8, bits, n):
    """(R, row_bytes) bit-stream rows (LSB first, rows start on a byte) -> (R, n) uint32 values."""
    R = rows_u8.shape[0]
    if bits == 8:
        return rows_u8[:, :n].astype(np.uint32)
    if bits == 16:
        return np.ascontiguousarray(rows_u8[:, :2 * n]).view('<u2').astype(np.uint32)
    if bits == 32:
        return np.ascontiguousarray(rows_u8[:, :4 * n]).view('<u4').astype(np.uint32)
    if bits == 4:
        lo = rows_u8 & 15
        hi = rows_u8 >> 4
        out = np.empty((R, rows_u8.shape[1] * 2), np.uint32)
        out[:, 0::2] = lo
        out[:, 1::2] = hi
        return out[:, :n]
    b = np.unpackbits(np.ascontiguousarray(rows_u8), axis=1, bitorder='little')[:, :n * bits].reshape(R, n, bits).astype(np.uint32)
    return (b << np.arange(bits, dtype=np.uint32)).sum(axis=2, dtype=np.uint32)


def plane_values(rf, e, k, row0=0, nrows=None):
    """Plane k's elements as a numpy array (rows row0..row0+nrows of the plane): floats as float32/float64-exact
    (f16 -> float32, bf16 -> float32), integers as int64 (signed dtypes sign-extended)."""
    rows_u8, pr, pc = rf.plane_rows(e, k)
    if nrows is None:
        nrows = pr - row0
    sub = rows_u8[row0:row0 + nrows]
    dt = e.enc.planes[k].dtype
    name, bits = DT[dt]
    if name == 'f32':
        return np.ascontiguousarray(sub[:, :4 * pc]).view('<f4').copy()
    if name == 'f16':
        return np.ascontiguousarray(sub[:, :2 * pc]).view('<f2').astype(np.float32)
    if name == 'bf16':
        u = np.ascontiguousarray(sub[:, :2 * pc]).view('<u2').astype(np.uint32) << 16
        return u.view('<f4')
    v = unpack_stream(sub, bits, pc).astype(np.int64)
    if name.startswith('i'):
        sign = 1 << (bits - 1)
        v = np.where(v >= sign, v - (1 << bits), v)
    return v


def decode_affine(rf, e, row0=0, nrows=None):
    """The core's affine decode (rad_encoding.h), v = ((q [- zero]) * scale * scale.1 ...) - min * min.1 ...,
    in f32 with the chain multiplied left to right, for logical rows row0..row0+nrows of a rank-2 weight.
    Plain encodings return the stored values."""
    enc = e.enc
    rows = int(np.prod(e.shape[:-1])) if len(e.shape) > 1 else 1
    cols = e.shape[-1] if e.shape else 1
    if nrows is None:
        nrows = rows - row0
    f32 = np.float32
    if enc.scheme == 'plain':
        return plane_values(rf, e, 0, row0, nrows).astype(f32)
    if enc.scheme != 'affine' or enc.transform:
        raise NotImplementedError(f'{enc.name()}')
    roles = {p.role: i for i, p in enumerate(enc.planes)}
    if any(p.kind == 1 for p in enc.planes):
        raise NotImplementedError('table planes')

    def expand(k):
        pl = enc.planes[k]
        br, bc = pl.block
        br = br if br > 0 else rows
        bc = bc if bc > 0 else cols
        r0 = row0 // br
        r1 = (row0 + nrows - 1) // br + 1
        v = plane_values(rf, e, k, r0, r1 - r0).astype(f32 if pl.dtype in (1, 2, 3) else np.float64)
        v = np.repeat(v, bc, axis=1)[:, :cols]
        if br > 1:
            v = np.repeat(v, br, axis=0)
        off = row0 - r0 * br
        return v[off:off + nrows]

    q = expand(roles['codes']).astype(f32)
    if 'zero' in roles:
        q = q - expand(roles['zero']).astype(f32)
    out = q
    lv = 0
    while True:
        r = 'scale' if lv == 0 else f'scale.{lv}'
        if r not in roles:
            break
        out = out * expand(roles[r]).astype(f32)
        lv += 1
    lv = 0
    sub = None
    while True:
        r = 'min' if lv == 0 else f'min.{lv}'
        if r not in roles:
            break
        m = expand(roles[r]).astype(f32)
        sub = m if sub is None else sub * m
        lv += 1
    if sub is not None:
        out = out - sub
    return out.astype(f32)
