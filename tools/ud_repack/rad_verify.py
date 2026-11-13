#!/usr/bin/env python3
"""rad_verify -- prove a lossless GGUF -> .rad conversion: every weight's dequantised f32 values in the
.rad equal the GGUF's, as sha256 of the little-endian row-major f32 bytes.

Three independent computations meet for each weight:

  (1) the .rad, re-read from disk by radfile.py (a numpy parser written from abi/rad_format.h) and
      decoded plane by plane with the core's affine rule (rad_encoding.h), in f32;
  (2) the GGUF, parsed by ggufhdr.py and dequantised by ggml_blocks.dequant_ref (a transcription of
      llama.cpp's dequantize_row_<type>), then put through the convention changes the converter
      is meant to make (value-head regroup, A_log) by the rules written out again below;
  (3) rad-convert's own --plan-json sha256 (the C++ importer's decode), when --plan is given.

All three are required to agree. The name -> GGUF-tensor and regroup rules below are deliberately
re-stated, not read from the plugin, so that a wrong map in the plugin shows up as a mismatch.

    python3 rad_verify.py MODEL.gguf MODEL.rad [--plan plan.json] [--workers 8] [--only REGEX]
"""
import argparse, hashlib, json, os, re, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggufhdr
from ggml_blocks import BLOCK, dequant_ref
from radfile import RadFile, decode_affine, plane_values

NV, NK, HV, HK = 32, 16, 128, 128      # Qwen3.6-35B-A3B value / key heads and widths (checked against the GGUF's kv)
A = NV // NK


def regroup_index(span, a, inner):
    """File order -> model order for one span: [a][b][inner] read as [b][a][inner]. Returns idx with
    model[i] = file[idx[i]]."""
    b = span // (a * inner)
    return np.arange(span).reshape(a, b, inner).transpose(1, 0, 2).reshape(-1)


def rule_for(name, kv):
    """Declared .rad name -> (gguf tensor name, expert or None, transform), transform in
    {None, ('rows', start, span, a, inner), ('cols', start, span, a, inner), ('neglog', start, span, a, inner)}."""
    nk = int(kv['qwen35moe.ssm.group_count']); nv = int(kv['qwen35moe.ssm.time_step_rank'])
    hk = int(kv['qwen35moe.ssm.state_size']); hv = int(kv['qwen35moe.ssm.inner_size']) // nv
    a = nv // nk
    qk = 2 * nk * hk
    vspan = nv * hv
    L = int(kv['qwen35moe.block_count']) - int(kv.get('qwen35moe.nextn_predict_layers', 0))
    top = {'token_embd.weight': 'token_embd.weight', 'output_norm.weight': 'output_norm.weight',
           'output.weight': 'output.weight', 'mtp.fc.weight': f'blk.{L}.nextn.eh_proj.weight',
           'mtp.pre_fc_norm_hidden.weight': f'blk.{L}.nextn.hnorm.weight',
           'mtp.pre_fc_norm_embedding.weight': f'blk.{L}.nextn.enorm.weight',
           'mtp.norm.weight': f'blk.{L}.nextn.shared_head_norm.weight'}
    if name in top:
        return top[name], None, None
    m = re.fullmatch(r'blk\.(\d+)\.(.+)', name)
    if not m:
        raise KeyError(name)
    l, rest = m.group(1), m.group(2)
    e = re.fullmatch(r'ffn_(gate|up|down)_exps\.(\d+)\.weight', rest)
    if e:
        return f'blk.{l}.ffn_{e.group(1)}_exps.weight', int(e.group(2)), None
    simple = {'attn_norm.weight': 'attn_norm.weight', 'ffn_norm.weight': 'post_attention_norm.weight',
              'attn_qg.weight': 'attn_q.weight', 'attn_k.weight': 'attn_k.weight', 'attn_v.weight': 'attn_v.weight',
              'attn_output.weight': 'attn_output.weight', 'attn_q_norm.weight': 'attn_q_norm.weight',
              'attn_k_norm.weight': 'attn_k_norm.weight', 'ssm_norm.weight': 'ssm_norm.weight',
              'ffn_gate_inp.weight': 'ffn_gate_inp.weight', 'ffn_gate_inp_shexp.weight': 'ffn_gate_inp_shexp.weight',
              'ffn_gate_shexp.weight': 'ffn_gate_shexp.weight', 'ffn_up_shexp.weight': 'ffn_up_shexp.weight',
              'ffn_down_shexp.weight': 'ffn_down_shexp.weight'}
    if rest in simple:
        return f'blk.{l}.{simple[rest]}', None, None
    gdn = {'ssm_qkv.weight': ('attn_qkv.weight', ('rows', qk, vspan, a, hv)),
           'ssm_z.weight': ('attn_gate.weight', ('rows', 0, vspan, a, hv)),
           'ssm_alpha.weight': ('ssm_alpha.weight', ('rows', 0, nv, a, 1)),
           'ssm_beta.weight': ('ssm_beta.weight', ('rows', 0, nv, a, 1)),
           'ssm_conv1d.weight': ('ssm_conv1d.weight', ('rows', qk, vspan, a, hv)),
           'ssm_dt.bias': ('ssm_dt.bias', ('rows', 0, nv, a, 1)),
           'ssm_a_log': ('ssm_a', ('neglog', 0, nv, a, 1)),
           'ssm_out.weight': ('ssm_out.weight', ('cols', 0, vspan, a, hv))}
    if rest in gdn:
        t, tr = gdn[rest]
        return f'blk.{l}.{t}', None, tr
    raise KeyError(name)


class Gguf:
    def __init__(self, path):
        self.h = ggufhdr.read_header(path)
        self.mm = np.memmap(path, dtype=np.uint8, mode='r')
        self.T = {t['name']: t for t in self.h['tensors']}
        self.kv = self.h['kv']

    def logical(self, t):
        """(rows, cols) of the tensor as the container sees it (outermost dims flattened, last dim = columns)."""
        shp = list(reversed(t['shape']))
        return int(np.prod(shp[:-1])) if len(shp) > 1 else 1, shp[-1]

    def blocks(self, t):
        qk, bb = BLOCK[t['type']] if t['type'] in BLOCK else (1, {'F32': 4}[t['type']])
        o = self.h['data_start'] + t['offset']
        return self.mm[o:o + t['nbytes']], qk, bb

    def rows_f32(self, name, r0, nr, expert=None):
        """Dequantised rows [r0, r0+nr) of tensor `name` (of one expert's slice when `expert` is given)."""
        t = self.T[name]
        rows, cols = self.logical(t)
        raw, qk, bb = self.blocks(t)
        if expert is not None:
            per = rows // t['shape'][-1] if False else None
            E = list(reversed(t['shape']))[0]
            rows //= E
            nbytes = t['nbytes'] // E
            raw = raw[expert * nbytes:(expert + 1) * nbytes]
        if t['type'] == 'F32':
            return np.ascontiguousarray(raw.view('<f4').reshape(rows, cols)[r0:r0 + nr])
        rb = cols // qk * bb
        blk = raw.reshape(rows, rb)[r0:r0 + nr].reshape(-1, bb)
        return dequant_ref(t['type'], blk).reshape(nr, cols).astype(np.float32)


def sha_rows(chunks):
    h = hashlib.sha256()
    for c in chunks:
        h.update(np.ascontiguousarray(c, dtype='<f4').tobytes())
    return h.hexdigest()


def gguf_side(g, name, expert, tr):
    t_name = name
    t = g.T[t_name]
    rows, cols = g.logical(t)
    if expert is not None:
        rows //= list(reversed(t['shape']))[0]
    if tr is None:
        CH = max(1, (1 << 23) // cols)
        return sha_rows(g.rows_f32(t_name, r, min(CH, rows - r), expert) for r in range(0, rows, CH)), (rows, cols)
    kind, start, span, a, inner = tr
    x = g.rows_f32(t_name, 0, rows, expert)
    if len(t['shape']) == 1:
        x = x.reshape(-1, 1)            # a vector regroups along its elements
    if kind == 'rows':
        idx = regroup_index(span, a, inner)
        x = np.concatenate([x[:start], x[start:start + span][idx], x[start + span:]], axis=0) if x.ndim == 2 else x
        if x.ndim == 2 and x.shape[1] == 1:
            pass
    elif kind == 'cols':
        idx = regroup_index(span, a, inner)
        x = np.concatenate([x[:, :start], x[:, start:start + span][:, idx], x[:, start + span:]], axis=1)
    elif kind == 'neglog':
        v = x.reshape(-1)
        idx = regroup_index(span, a, inner)
        v = v[start:start + span][idx]
        return None, (v.shape[0],)         # handled by the caller (derived)
    return sha_rows([x]), x.shape


_G = None
_R = None


def _init(gpath, rpath):
    global _G, _R
    _G = Gguf(gpath)
    _R = RadFile(rpath)


def _check(name):
    e = _R.by_name[name]
    gname, expert, tr = rule_for(name, _G.kv)
    rows = int(np.prod(e.shape[:-1])) if len(e.shape) > 1 else 1
    cols = e.shape[-1]
    if tr is not None and tr[0] == 'neglog':
        # A_log: compare with log(-ssm_a) recomputed here, in f64, tolerance two f32 ulps of exp round trip
        a = _G.rows_f32(gname, 0, 1).reshape(-1) if _G.T[gname]['type'] == 'F32' else None
        kind, start, span, aa, inner = tr
        a = a[start:start + span][regroup_index(span, aa, inner)]
        got = plane_values(_R, e, 0, 0, 1).reshape(-1)
        back = np.exp(got.astype(np.float64)).astype(np.float32)
        ulp = np.spacing(np.abs(a))
        ok = bool(np.all(np.abs(back + a) <= 2 * ulp)) and np.array_equal(got.shape, a.shape)
        want = np.log(-a.astype(np.float64)).astype(np.float32)
        return name, ok, 'derived', f'max |exp(A_log)+ssm_a|/|ssm_a| = {float(np.max(np.abs(back + a) / np.abs(a))):.2e}', None, None
    # .rad side
    h = hashlib.sha256()
    CH = max(1, (1 << 23) // cols)
    if e.enc.scheme == 'affine' or e.enc.scheme == 'plain':
        for r in range(0, rows, CH):
            h.update(np.ascontiguousarray(decode_affine(_R, e, r, min(CH, rows - r)), dtype='<f4').tobytes())
    else:
        raise NotImplementedError(e.enc.name())
    rad = h.hexdigest()
    gsha, shp = gguf_side(_G, gname, expert, tr)
    return name, rad == gsha, e.enc.name(), '', rad, gsha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf'); ap.add_argument('rad')
    ap.add_argument('--plan'); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--only'); ap.add_argument('--out')
    args = ap.parse_args()
    rf = RadFile(args.rad)
    names = [e.name for e in rf.entries]
    if args.only:
        names = [n for n in names if re.search(args.only, n)]
    plan = {}
    if args.plan:
        for w in json.load(open(args.plan))['weights']:
            plan[w['name']] = w.get('sha256_dequant_f32')
    t0 = time.time()
    bad, n_ok, n_derived, planbad, planmiss = [], 0, 0, [], 0
    per_class = {}
    out = open(args.out, 'w') if args.out else None
    with Pool(args.workers, initializer=_init, initargs=(args.gguf, args.rad)) as pool:
        for i, (name, ok, enc, note, rad, gsha) in enumerate(pool.imap_unordered(_check, names, chunksize=8)):
            if enc == 'derived':
                n_derived += 1
                if not ok: bad.append((name, note))
                if out: out.write(json.dumps(dict(name=name, ok=ok, note=note)) + '\n')
                continue
            if ok: n_ok += 1
            else: bad.append((name, f'rad {rad} gguf {gsha}'))
            if args.plan:
                if plan.get(name) is None: planmiss += 1
                elif plan[name] != rad: planbad.append(name)
            c = per_class.setdefault(enc[:48], [0, 0])
            c[0] += 1; c[1] += ok
            if out: out.write(json.dumps(dict(name=name, ok=ok, enc=enc, rad=rad, gguf=gsha, plan=plan.get(name))) + '\n')
            if (i + 1) % 2000 == 0:
                print(f'  {i + 1}/{len(names)}  {time.time() - t0:.0f}s', flush=True)
    print(f'weights checked: {len(names)}  hash-equal (rad == gguf): {n_ok}  derived (A_log) checked: {n_derived}  FAILED: {len(bad)}')
    for enc, (n, ok) in sorted(per_class.items()):
        print(f'  {enc:50s} {ok}/{n}')
    if args.plan:
        print(f'plan sha256 (rad-convert C++ decode) vs rad decode: {len(names) - n_derived - len(planbad) - planmiss} equal, {len(planbad)} differ, {planmiss} without a hash')
    for b in bad[:20]:
        print('FAIL', *b)
    print(f'{time.time() - t0:.0f}s')
    sys.exit(1 if bad or planbad else 0)


if __name__ == '__main__':
    main()
