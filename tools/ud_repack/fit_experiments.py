#!/usr/bin/env python3
"""fit_experiments -- how close can an EXISTING rad weight format get to a GGUF tensor's dequantised values?

Reference is always the GGUF's own f32 dequantisation (not bf16), so the numbers are the cost of the
format change alone, on top of whatever the GGUF already lost. Quantisation here is RTN with a
searched block scale (the "RTN+scale search" of docs/radiance-3bit-experts.md stage 1), no GPTQ and
no calibration; with those the codebook errors would be lower (unmeasured).

Candidates (bits/weight counted with their scale planes):
  w3nl64 3.125  w4nl64 4.125  w4nl32 4.25  w5nl64 5.125   codebook, E4M3 scale per 64 (or 32) under the fixed 2^-13,
                                                          optionally after the fwht128 rotation the real kernels use
  u4a128 4.1875  u4 codes, f16 scale and u8 zero per 128   (the plugin's asymmetric 4-bit trunk form)
  i8g128 8.125   int8, bf16 scale per 128
  fp8b   8.0     E4M3 with a bf16 scale per 128x128 block   (what the shipped FP8 .rad uses)
Prints one JSON object per (tensor, candidate).
"""
import argparse, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ud_repack import load_all, tensor_bytes, layer_of
from ggml_blocks import SPLIT, DECODE, dequant_ref
from ggufhdr import BLK as BLOCK

TABLES = {
    'w4nl': [1.125, 3.25, 5.5, 8.0, 11.0, 14.0, 18.0, 22.0],
    'w5nl': [0.8125, 2.5, 4.0, 5.5, 7.5, 9.0, 11.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 30.0],
    'w3nl': [3.25, 10.0, 18.0, 28.0],
}  # magnitudes; a sign bit doubles them, exactly as lq_grid.cpp's tables (magnitudes ascending then negatives)


def e4m3_positive():
    v = []
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7: continue    # NaN
            v.append((m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
    return np.array(sorted(set(v[1:])), np.float64)   # drop 0


E4 = e4m3_positive()


def fwht128(x):
    """Orthonormal Walsh-Hadamard transform over groups of 128 along the last axis."""
    sh = x.shape
    y = x.reshape(-1, 128).astype(np.float64)
    h = 1
    while h < 128:
        y = y.reshape(-1, 128 // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = np.stack([a + b, a - b], axis=2)
        h *= 2
    return (y.reshape(-1, 128) / np.sqrt(128)).reshape(sh)


def fit_codebook(X, mags, G):
    """X (N,K) float -> reconstruction (N,K) float64 using sign+magnitude table, E4M3 scale per G of K under 2^-13."""
    N, K = X.shape
    B = X.reshape(-1, G).astype(np.float64)
    t = np.array(mags, np.float64)
    levels = np.concatenate([-t[::-1], t])            # sorted ascending
    mids = (levels[1:] + levels[:-1]) / 2
    amax = np.abs(B).max(axis=1)
    est = np.maximum(amax * 8192.0 / t[-1], E4[0])
    i0 = np.clip(np.searchsorted(E4, est), 0, len(E4) - 1)
    best = np.full(B.shape[0], np.inf); bestrec = np.zeros_like(B)
    for o in range(-5, 4):
        s = E4[np.clip(i0 + o, 0, len(E4) - 1)] * 2.0 ** -13
        idx = np.searchsorted(mids, B / s[:, None])
        rec = levels[idx] * s[:, None]
        sse = ((rec - B) ** 2).sum(axis=1)
        m = sse < best
        best[m] = sse[m]; bestrec[m] = rec[m]
    z = amax == 0
    bestrec[z] = 0.0
    return bestrec.reshape(N, K)


def bf16_round(a):
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def fit_u4_asym(X, G=128):
    B = X.reshape(-1, G).astype(np.float64)
    mn, mx = B.min(axis=1), B.max(axis=1)
    s = ((mx - mn) / 15.0).astype(np.float16).astype(np.float64)
    s[s == 0] = 1.0
    z = np.clip(np.round(-mn / s), 0, 255)
    q = np.clip(np.round(B / s[:, None] + z[:, None]), 0, 15)
    return (s[:, None] * (q - z[:, None])).reshape(X.shape)


def fit_i8(X, G=128):
    B = X.reshape(-1, G).astype(np.float64)
    s = bf16_round((np.abs(B).max(axis=1) / 127.0).astype(np.float32))
    s[s == 0] = 1.0
    return (np.clip(np.round(B / s[:, None]), -127, 127) * s[:, None]).reshape(X.shape)


def fit_fp8_block(X, R=128, C=128):
    N, K = X.shape
    if N % R or K % C: return None
    B = X.reshape(N // R, R, K // C, C).astype(np.float64)
    amax = np.abs(B).max(axis=(1, 3), keepdims=True)
    s = bf16_round((amax / 448.0).astype(np.float32)); s[s == 0] = 1.0
    y = np.abs(B) / s
    levels = np.concatenate([[0.0], E4])
    mids = (levels[1:] + levels[:-1]) / 2
    q = levels[np.searchsorted(mids, y)]
    return (np.sign(B) * q * s).reshape(N, K)


def fit_uniform_affine_exact_test(planes_dequant_blocks):
    pass


def metrics(ref, rec):
    ref = ref.astype(np.float64); rec = rec.astype(np.float64)
    d = np.abs(ref - rec)
    nz = ref != 0
    r32, c32 = ref.astype(np.float32), rec.astype(np.float32)
    return dict(rel_rmse=float(np.sqrt((d ** 2).sum() / (ref ** 2).sum())), max_abs=float(d.max()),
                frac_zero_in_ref=float(1 - nz.mean()),
                frac_equal_f32_nonzero=float((c32[nz] == r32[nz]).mean()),
                max_abs_over_absmax=float(d.max() / np.abs(ref).max()),
                frac_equal_f32=float((rec.astype(np.float32) == ref.astype(np.float32)).mean()),
                frac_equal_f16=float((rec.astype(np.float16) == ref.astype(np.float16)).mean()))


def dequant_rows(t, expert, nrows, gguf_lib=True):
    ty = t['type']; K = t['shape'][0]; N = t['shape'][1] if len(t['shape']) > 1 else 1
    E = t['shape'][2] if len(t['shape']) > 2 else 1
    qk, bb = BLOCK[ty]
    raw = tensor_bytes(t).reshape(E, N, K // qk, bb)[expert if E > 1 else 0, :nrows]
    blocks = np.ascontiguousarray(raw).reshape(-1, bb)
    if ty in SPLIT:
        w = dequant_ref(ty, blocks)
    else:
        import gguf
        from gguf import quants
        w = quants.dequantize(blocks, getattr(gguf.GGMLQuantizationType, ty)).astype(np.float32)
    return w.reshape(nrows, K)


CANDS = {
    'w3nl64': lambda X, rot: fit_codebook(X, TABLES['w3nl'], 64),
    'w4nl64': lambda X, rot: fit_codebook(X, TABLES['w4nl'], 64),
    'w4nl32': lambda X, rot: fit_codebook(X, TABLES['w4nl'], 32),
    'w5nl64': lambda X, rot: fit_codebook(X, TABLES['w5nl'], 64),
    'u4a128': lambda X, rot: fit_u4_asym(X, 128),
    'i8g128': lambda X, rot: fit_i8(X, 128),
    'fp8b':   lambda X, rot: fit_fp8_block(X),
}
BPW = {'w3nl64': 3.125, 'w4nl64': 4.125, 'w4nl32': 4.25, 'w5nl64': 5.125, 'u4a128': 4.1875, 'i8g128': 8.125, 'fp8b': 8.0 + 16 / 16384}
ROTATED = {'w3nl64', 'w4nl64', 'w4nl32', 'w5nl64'}   # the real kernels apply fwht128 to weights and activations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf', nargs='+'); ap.add_argument('--tensor', required=True)
    ap.add_argument('--expert', type=int, default=0); ap.add_argument('--rows', type=int, default=512)
    ap.add_argument('--cands', default='w4nl64,w5nl64,u4a128,i8g128,fp8b')
    ap.add_argument('--no-rot', action='store_true')
    a = ap.parse_args()
    kv, ts = load_all(a.gguf)
    t = [x for x in ts if x['name'] == a.tensor][0]
    nrows = min(a.rows, t['shape'][1] if len(t['shape']) > 1 else 1)
    X = dequant_rows(t, a.expert, nrows)
    for c in a.cands.split(','):
        Y = X
        rot = c in ROTATED and not a.no_rot
        if rot: Y = fwht128(X)
        rec = CANDS[c](Y, rot)
        if rec is None: continue
        if rot: rec = fwht128(rec)      # orthonormal and involutive on 128: back to the weight basis
        m = metrics(X, rec)
        m.update(tensor=a.tensor, ggml=t['type'], expert=a.expert, cand=c, bpw=BPW[c], rotated=rot, ggml_bpw=8 * BLOCK[t['type']][1] / BLOCK[t['type']][0] if t['type'] in BLOCK else None)
        print(json.dumps(m), flush=True)


if __name__ == '__main__':
    main()
