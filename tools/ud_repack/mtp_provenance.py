#!/usr/bin/env python3
"""mtp_provenance -- how far is the UD GGUF's weight from the official checkpoint's, for one expert?

Reads one expert's fp8 planes straight out of a rad container that was converted from the official
FP8 checkpoint (planes are `checkpoint` copies: the official bits) at absolute offsets taken from
`rad-info -v`, and compares them with the GGUF's dequantised values of the same expert.
Usage: mtp_provenance.py GGUF RAD LAYER EXPERT CODES_OFF SCALE_OFF   (gate_up plane, 1024x2048 fp8, bf16 [8x16] scales)
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ud_repack import load_all
from fit_experiments import dequant_rows


def e4m3_table():
    t = np.zeros(256, np.float64)
    for b in range(256):
        s = -1.0 if b & 128 else 1.0; e = (b >> 3) & 15; m = b & 7
        t[b] = np.nan if (e == 15 and m == 7) else s * ((m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return t


def main():
    gguf, rad, layer, ex, coff, soff = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    N, K = 1024, 2048
    codes = np.fromfile(rad, dtype=np.uint8, count=N * K, offset=coff).reshape(N, K)
    sc = np.fromfile(rad, dtype='<u2', count=(N // 128) * (K // 128), offset=soff).astype(np.uint32) << 16
    sc = sc.view(np.float32).reshape(N // 128, K // 128).astype(np.float64)
    W = e4m3_table()[codes] * np.repeat(np.repeat(sc, 128, 0), 128, 1)
    kv, ts = load_all([gguf])
    by = {t['name']: t for t in ts}
    g = by[f'blk.{layer}.ffn_gate_exps.weight']; u = by[f'blk.{layer}.ffn_up_exps.weight']
    G = dequant_rows(g, ex, 512).astype(np.float64); U = dequant_rows(u, ex, 512).astype(np.float64)
    out = dict(layer=layer, expert=ex, gguf_type=g['type'])
    for name, ref, off in (('gate', G, 0), ('up', U, 512)):
        for oname, o in (('as_given', off), ('swapped', 512 - off)):
            d = W[o:o + 512] - ref
            out[f'{name}_{oname}_rel_rmse'] = float(np.sqrt((d ** 2).sum() / (ref ** 2).sum()))
    out['frac_gguf_equal_official_gate'] = float((W[:512].astype(np.float32) == G.astype(np.float32)).mean())
    print(json.dumps(out))


if __name__ == '__main__':
    main()
