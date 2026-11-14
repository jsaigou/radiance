#!/usr/bin/env python3
"""gguf_q80_dump.py -- list a GGUF's Q8_0 tensors, or dump the first rows of chosen ones as raw
block bytes (34 bytes/block, exactly as stored) for halo_q80_bench's `real` mode. Read-only on the
GGUF. No third-party imports (own header parser, GGUF v3).

  gguf_q80_dump.py list  <gguf>
  gguf_q80_dump.py dump  <gguf> <outdir> <max_rows> <tensor>...      writes <outdir>/<tensor>.q80 + .shape
"""
import struct, sys, os

def rd(f, fmt):
    n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))

def rstr(f):
    (n,) = rd(f, '<Q'); return f.read(n).decode('utf8', 'replace')

def skip_val(f, t):
    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in sizes: f.seek(sizes[t], 1)
    elif t == 8: rstr(f)
    elif t == 9:
        (et,) = rd(f, '<I'); (n,) = rd(f, '<Q')
        for _ in range(n): skip_val(f, et)
    else: raise SystemExit('bad kv type %d' % t)

def header(path):
    f = open(path, 'rb')
    magic, ver, nt, nkv = rd(f, '<4sIQQ')
    assert magic == b'GGUF' and ver == 3, (magic, ver)
    align = 32
    for _ in range(nkv):
        k = rstr(f); (t,) = rd(f, '<I')
        if k == 'general.alignment' and t == 4:
            (align,) = rd(f, '<I')
        else: skip_val(f, t)
    ts = []
    for _ in range(nt):
        name = rstr(f); (nd,) = rd(f, '<I'); dims = rd(f, '<%dQ' % nd)
        ty, off = rd(f, '<IQ'); ts.append((name, dims, ty, off))
    pos = f.tell(); data0 = (pos + align - 1) // align * align
    return f, ts, data0

def main():
    mode, path = sys.argv[1], sys.argv[2]
    f, ts, data0 = header(path)
    if mode == 'list':
        for name, dims, ty, off in ts:
            if ty == 8: print(name, dims, off)
        return
    outdir, maxrows, names = sys.argv[3], int(sys.argv[4]), sys.argv[5:]
    os.makedirs(outdir, exist_ok=True)
    for name, dims, ty, off in ts:
        if name in names:
            assert ty == 8 and len(dims) == 2, (name, ty, dims)
            K, N = dims
            assert K % 32 == 0
            rows = min(N, maxrows)
            f.seek(data0 + off)
            blob = f.read(rows * (K // 32) * 34)
            open(os.path.join(outdir, name + '.q80'), 'wb').write(blob)
            open(os.path.join(outdir, name + '.shape'), 'w').write('%d %d %d\n' % (rows, K, N))
            print('dumped', name, 'rows', rows, 'of', N, 'K', K, len(blob), 'bytes')

main()
