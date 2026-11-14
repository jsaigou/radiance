#!/usr/bin/env python3
"""extract.py -- the Unsloth UD release's IQ4_NL n-gram table as the container's two planes, and a
small sample with the reference's decode for rad-avx-iq4 and halo_selftest.

The PLE table is one GGUF tensor, `per_layer_token_embd.weight`, IQ4_NL. A row is `n_embd / 32`
blocks of 18 bytes: two f16 `d` bytes then sixteen `qs` bytes, element j of the block the low
nibble of qs[j] and j+16 the high one (tools/ud_repack/ggml_blocks.py split_iq4_nl). The container
stores the value-order nibbles packed two a byte (`codes`, n_embd / 2 bytes a row) and the f16
`d` of each block (`scales`, one per 32 values), which is what rad_encoding.h's IQ4_NL encoding
names.

  extract.py sample  --gguf SHARD --out DIR [--rows 4096] [--seed 1]
      sample/codes.bin, sample/scales.bin, sample/ids.bin, sample/expected.bf16
      `expected.bf16` is u16 bf16 of d * kvalues_iq4nl[code] rounded to nearest even, the value
      the device kernel and libavx must both write.

  extract.py full --gguf SHARD --out DIR [--rows N]
      codes.bin, scales.bin for the whole table (26.8 GiB + 3.0 GiB), streamed.

The reference arithmetic is tools/ud_repack/ggml_blocks.py's decode_iq4_nl; this file re-derives the
planes from the block bytes and uses the same KVALUES table.
"""
import argparse
import os
import sys

import numpy as np

KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                          1, 13, 25, 38, 53, 69, 89, 113], np.int8)


def bf16_rne(f32):
    """rad_f32_to_bf16 (abi/rad_plugin.h): nearest even, NaN kept quiet, overflow to infinity."""
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    a = u & np.uint32(0x7fffffff)
    out = np.where((a > np.uint32(0x7f800000)),
                   (u >> np.uint32(16)) | np.uint32(0x0040),
                   (u + np.uint32(0x7fff) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16))
    return out.astype(np.uint16)


def split_row(b, n_embd):
    """A row's blocks (nb, 18) -> (codes (nb, 32) value order, d (nb,) f16 bits as uint16)."""
    nb = n_embd // 32
    blocks = np.ascontiguousarray(b[:nb * 18]).reshape(nb, 18)
    d = np.ascontiguousarray(blocks[:, 0:2]).view('<f2').reshape(nb)
    qs = blocks[:, 2:18]
    codes = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.uint8)   # value order
    return codes, d


def pack_codes(codes):
    """(rows, n_embd) value-order codes -> (rows, n_embd / 2) packed two a byte."""
    c = np.ascontiguousarray(codes, dtype=np.uint8)
    return (c[:, 0::2] & 15) | ((c[:, 1::2] & 15) << 4)


def decode(codes, d):
    """(rows, nb, 32) codes and (rows, nb) f16 bits -> (rows, n_embd) f32, through tools/ud_repack's
    own decode_iq4_nl (the Python reference), exact in f32."""
    try:
        import ggml_blocks
    except ImportError:
        sys.exit("ggml_blocks.py not found: keep tools/ud_repack beside tools/ple_iq4")
    rows, nb, _ = codes.shape
    p = {'codes': np.ascontiguousarray(codes).reshape(rows * nb, 32),
         'd': np.ascontiguousarray(d).reshape(rows * nb)}
    return ggml_blocks.decode_iq4_nl(p).reshape(rows, nb * 32)


def find_table(path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ud_repack'))
    try:
        import ggufhdr
    except ImportError:
        sys.exit("ggufhdr.py not found: keep tools/ud_repack beside tools/ple_iq4, or pass the "
                 "GGUF header fields yourself")
    h = ggufhdr.read_header(path)
    for t in h['tensors']:
        if t['name'] == 'per_layer_token_embd.weight':
            if t['type'] != 'IQ4_NL':
                sys.exit(f"per_layer_token_embd.weight is {t['type']}, not IQ4_NL")
            n_embd = int(t['shape'][0])
            rows = int(t['shape'][1])
            return h['data_start'] + int(t['offset']), rows, n_embd
    sys.exit(f"{path}: no per_layer_token_embd.weight")


def sample(args):
    base, rows, n_embd = find_table(args.gguf)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n = min(args.rows, rows)
    # The table rows to draw, then the sample is the small image: row k of the image is a random
    # real table row, and the ids the gather reads index the image (0..n-1).
    ids = rng.choice(rows, size=n, replace=False).astype(np.int64)
    nb = n_embd // 32
    row_bytes = nb * 18
    codes = np.empty((n, n_embd), np.uint8)
    d = np.empty((n, nb), '<f2')
    with open(args.gguf, 'rb') as f:
        for k, rid in enumerate(ids):
            f.seek(base + int(rid) * row_bytes)
            b = np.frombuffer(f.read(row_bytes), np.uint8)
            c, dd = split_row(b, n_embd)
            codes[k] = c.reshape(-1)
            d[k] = dd
    packed = pack_codes(codes)
    vals = decode(codes.reshape(n, nb, 32), d)
    expected = bf16_rne(vals)
    open(os.path.join(args.out, 'codes.bin'), 'wb').write(packed.tobytes())
    open(os.path.join(args.out, 'scales.bin'), 'wb').write(d.view(np.uint16).reshape(-1).tobytes())
    open(os.path.join(args.out, 'ids.bin'), 'wb').write(np.arange(n, dtype=np.int32).tobytes())
    open(os.path.join(args.out, 'expected.bf16'), 'wb').write(expected.tobytes())
    print(f"sample: {n} rows of {n_embd} from {args.gguf} (table {rows} rows) -> {args.out}")
    print(f"  codes {packed.nbytes} B, scales {d.nbytes} B, ids {ids.nbytes} B, "
          f"expected {expected.nbytes} B")


def full(args):
    base, rows, n_embd = find_table(args.gguf)
    total = min(args.rows, rows) if args.rows else rows
    nb = n_embd // 32
    row_bytes = nb * 18
    os.makedirs(args.out, exist_ok=True)
    cpath = os.path.join(args.out, 'codes.bin')
    spath = os.path.join(args.out, 'scales.bin')
    chunk = 1 << 18
    done = 0
    with open(args.gguf, 'rb') as f, open(cpath, 'wb') as fc, open(spath, 'wb') as fs:
        f.seek(base)
        while done < total:
            n = min(chunk, total - done)
            b = np.frombuffer(f.read(n * row_bytes), np.uint8)
            if b.size != n * row_bytes:
                sys.exit(f"short read at row {done}")
            blk = b.reshape(n, nb, 18)
            qs = blk[:, :, 2:18]
            # value order: low nibbles of qs[j] are values 0..15, high nibbles 16..31
            lo = qs & 15
            hi = qs >> 4
            # pack value 2k low, 2k+1 high within each 32: pairs come from [lo, hi] interleaved
            both = np.concatenate([lo, hi], axis=2).reshape(n, nb, 32)
            packed = (both[:, :, 0::2] & 15) | ((both[:, :, 1::2] & 15) << 4)
            fc.write(packed.reshape(n, -1).tobytes())
            fs.write(np.ascontiguousarray(blk[:, :, 0:2]).view('<f2').reshape(n, nb).view(np.uint16).tobytes())
            done += n
            if done % (chunk * 16) == 0 or done == total:
                print(f"  {done}/{total} rows", flush=True)
    print(f"full: {total} rows -> {cpath} ({os.path.getsize(cpath)} B), {spath} "
          f"({os.path.getsize(spath)} B)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('sample', 'full'):
        p = sub.add_parser(name)
        p.add_argument('--gguf', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--rows', type=int, default=4096 if name == 'sample' else 0)
        if name == 'sample':
            p.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    (sample if a.cmd == 'sample' else full)(a)


if __name__ == '__main__':
    main()
