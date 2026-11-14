#!/usr/bin/env python3
"""route_b -- the GGUF as a MAP, not as the source (docs/radiance-gguf-plugin.md, "Route B").

Route A writes the GGUF's own numbers. Route B keeps only what the GGUF decided -- which tensors are
how many bits -- and converts the ORIGINAL weights with the formats the engine already has fast
kernels for. This tool does the two things that can be done without a GPU:

  plan   GGUF -> a rad-convert recipe (the existing recipe language, patterns over the declared
         names of the qwen35moe_fp8 plugin) and a size table, Route A against Route B, per role.
  err    relative RMSE against the official bf16 weights of: the GGUF's own dequantisation (Route A's
         values), each candidate Route B format fitted to the bf16 (RTN with a searched scale, the
         method of fit_experiments.py), and the official FP8 container. Sampled tensors.

    python3 route_b.py plan GGUF [--recipe-out F]
    python3 route_b.py err  GGUF --bf16 DIR --fp8 FP8.rad [--out F]
"""
import argparse, glob, json, os, re, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggufhdr
from ggml_blocks import BLOCK, dequant_ref
from fit_experiments import CANDS, ROTATED, BPW, fwht128
from radfile import RadFile, plane_values

GG_BPW = {t: 8.0 * bb / qk for t, (qk, bb) in BLOCK.items()}
GG_BPW.update({'F32': 32.0, 'F16': 16.0, 'BF16': 16.0})

# ---------------------------------------------------------------- the mapping GGUF type -> existing format
# (type, role) -> (format name, bits per weight, recipe options). Formats are the ones with kernels in
# libr4d today (docs/MOE-W4.md "What is served", docs/GUIDE.md): w4nl64a8h / w5nl64a8h routed experts,
# i8*bf16[1x128] dense trunk, block fp8 as the official checkpoint ships it.
W4 = 'gptq table=w4nl group=64 scale=fp8_e4m3 scale2=f32 block2=*x* scale2_value=0.0001220703125 transform=fwht128 rule=search cd=3 calib=$CALIB'
W5 = W4.replace('w4nl', 'w5nl')
RTN_W4 = W4.replace('gptq', 'rtn').replace(' cd=3 calib=$CALIB', '')
RTN_W5 = RTN_W4.replace('w4nl', 'w5nl')
I8 = 'rtn codes=i8 group=128 scale=bf16 clamp=sym rule=search'
ROUTE_B = {
    'Q4_K': ('w4nl64', 4.125, W4), 'Q5_K': ('w5nl64', 5.125, W5), 'Q6_K': ('w5nl64', 5.125, W5),
    'Q8_0': ('i8g128', 8.125, I8),
}
ALT_B = {'Q6_K': ('fp8b', 8.0 + 16 / 16384, None), 'Q8_0': ('fp8b', 8.0 + 16 / 16384, None)}


def role_of(name):
    n = re.sub(r'^blk\.\d+\.', '', name)
    return n


def declared(name, L):
    """GGUF tensor name -> the declared name(s) qwen35moe_fp8 uses for it (rule patterns), or None for 'keep'."""
    m = re.fullmatch(r'blk\.(\d+)\.(.+)', name)
    if name in ('output.weight',):
        return 'output.weight'
    if name == 'token_embd.weight':
        return None
    if not m:
        return None
    l, r = int(m.group(1)), m.group(2)
    t = {'attn_q.weight': 'attn_qg.weight', 'attn_k.weight': 'attn_k.weight', 'attn_v.weight': 'attn_v.weight',
         'attn_output.weight': 'attn_output.weight', 'attn_qkv.weight': 'ssm_inz.weight', 'attn_gate.weight': 'ssm_inz.weight',
         'ssm_out.weight': 'ssm_out.weight', 'ffn_gate_shexp.weight': 'ffn_gate_up_shexp.weight',
         'ffn_up_shexp.weight': 'ffn_gate_up_shexp.weight', 'ffn_down_shexp.weight': 'ffn_down_shexp.weight',
         'ffn_gate_exps.weight': 'ffn_gate_up_exps.*.weight', 'ffn_up_exps.weight': 'ffn_gate_up_exps.*.weight',
         'ffn_down_exps.weight': 'ffn_down_exps.*.weight'}.get(r)
    return f'blk.{l}.{t}' if t else None


def load_gguf(path):
    h = ggufhdr.read_header(path)
    return h


def plan(args):
    h = load_gguf(args.gguf)
    L = int(h['kv']['qwen35moe.block_count']) - int(h['kv'].get('qwen35moe.nextn_predict_layers', 0))
    rows = {}            # role -> [n_elem, A_bytes, B_bytes, types]
    rules = []           # (pattern, rule text, comment) in order; first match wins
    seen = set()
    for t in h['tensors']:
        ty, ne, nm = t['type'], t['nelem'], t['name']
        a_bytes = t['nbytes']
        layer = int(re.match(r'blk\.(\d+)\.', nm).group(1)) if nm.startswith('blk.') else -1
        role = role_of(nm) if layer >= 0 else nm
        mtp = layer >= L
        if ty in ROUTE_B and not (mtp and 'exps' in nm):
            fmt, bpw, rule = ROUTE_B[ty]
            b_bytes = ne * bpw / 8
        elif ty == 'Q8_0' and mtp:                       # the MTP layer's 256 Q8_0 experts: the official fp8 bits
            fmt, bpw, rule = 'fp8b(official)', 8.0 + 16 / 16384, None
            b_bytes = ne * bpw / 8
        elif ty == 'F32' and ('ffn_gate_inp' in nm):
            fmt, bpw, rule = 'bf16', 16.0, None          # the router is bf16-exact (Measured): halve it
            b_bytes = ne * 2
        else:
            fmt, bpw, rule = ty.lower(), GG_BPW.get(ty, 0), None
            b_bytes = a_bytes
        if nm == 'token_embd.weight':                    # a row lookup: bf16 in every container
            fmt, bpw, rule, b_bytes = 'bf16', 16.0, None, ne * 2
        key = (re.sub(r'^blk\.\d+\.', 'blk.*.', nm), ty, fmt)
        r = rows.setdefault(key, [0, 0, 0, 0])
        r[0] += ne; r[1] += a_bytes; r[2] += b_bytes; r[3] += 1
        d = declared(nm, L)
        if d and rule and (d, ty) not in seen and not (mtp and 'exps' in nm):
            seen.add((d, ty))
            rules.append((d, rule, f'{ty} -> {fmt}'))
    # experts: per-layer rules (the mix differs per layer), plus the MTP layer left to the checkpoint
    out = ['# Route B recipe, generated from %s by tools/ud_repack/route_b.py' % os.path.basename(args.gguf),
           '# The GGUF is a MAP: each rule gives a tensor the existing fast format with about the bits Unsloth chose.',
           '# Source: the official bf16 (or fp8) safetensors of Qwen3.6-35B-A3B, declared names of arch/qwen35moe_fp8.',
           '# The layer-40 (MTP) experts are not listed: no rule matches them, so they keep the checkpoint\'s fp8 block form.',
           '# Calibration: $CALIB for gptq (a RADIANCE_CALIB_DIR run); --imatrix <unsloth imatrix> weights rule=search.',
           '']
    ex = {}
    for t in h['tensors']:
        m = re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight', t['name'])
        if m and int(m.group(1)) < L:
            ex.setdefault((int(m.group(1)), 'down' if m.group(2) == 'down' else 'gate_up'), set()).add(t['type'])
    for (l, kind), tys in sorted(ex.items()):
        ty = sorted(tys)[0]
        if len(tys) > 1:
            ty = max(tys, key=lambda x: GG_BPW[x])       # gate and up differ: the wider one (fused tensor)
        fmt, bpw, rule = ROUTE_B[ty]
        out.append(f'blk.{l}.ffn_{kind}_exps.*.weight'.ljust(40) + ' ' + rule + f'   # {ty}')
    out.append('')
    out.append('# dense trunk (the GGUF keeps all of it at Q8_0)')
    for d, rule, c in rules:
        if '*' in d and 'exps' in d: continue
        if d.startswith('blk.') and d not in ('output.weight',):
            d = re.sub(r'^blk\.\d+\.', 'blk.*.', d)
        if (d, rule) in [(x, y) for x, y, _ in []]: continue
        line = d.ljust(40) + ' ' + rule + f'   # {c}'
        if line not in out: out.append(line)
    out.append('mtp.draft_head.weight'.ljust(40) + ' rtn codes=u2 zero=u8 group=128 scale=f16    # derived from output.weight, as the fp8 container does')
    txt = '\n'.join(out) + '\n'
    if args.recipe_out:
        open(args.recipe_out, 'w').write(txt)
    GiB = 1 << 30
    print(f'{"tensor role":38s} {"ggml":6s} {"-> B format":14s} {"A GiB":>8s} {"B GiB":>8s}  {"A bpw":>6s} {"B bpw":>6s}')
    ta = tb = 0
    for (role, ty, fmt), (ne, ab, bb, n) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
        if ab / GiB < 0.0005: continue
        print(f'{role:38s} {ty:6s} {fmt:14s} {ab / GiB:8.3f} {bb / GiB:8.3f}  {8 * ab / ne:6.3f} {8 * bb / ne:6.3f}')
    ta = sum(v[1] for v in rows.values()); tb = sum(v[2] for v in rows.values())
    print(f'{"TOTAL":38s} {"":6s} {"":14s} {ta / GiB:8.3f} {tb / GiB:8.3f}   Route B / Route A = {tb / ta:.4f}')
    # alternatives for Q6_K / Q8_0
    q6 = sum(v[0] for k, v in rows.items() if k[1] == 'Q6_K')
    print(f'Q6_K elements {q6 / 1e9:.3f} G: as w5nl64 {q6 * 5.125 / 8 / GiB:.3f} GiB, as fp8 block {q6 * 8.001 / 8 / GiB:.3f} GiB, as GGUF {q6 * 6.5625 / 8 / GiB:.3f} GiB')
    if not args.recipe_out:
        print('\n' + txt)


# ---------------------------------------------------------------- measurement against the official bf16
class Safetensors:
    def __init__(self, d):
        self.idx = {}
        for f in sorted(glob.glob(os.path.join(d, '*.safetensors'))):
            with open(f, 'rb') as fh:
                n = struct.unpack('<Q', fh.read(8))[0]
                hd = json.loads(fh.read(n))
            mm = np.memmap(f, dtype=np.uint8, mode='r')
            for k, v in hd.items():
                if k != '__metadata__':
                    self.idx[k] = (mm, 8 + n + v['data_offsets'][0], v['data_offsets'][1] - v['data_offsets'][0], v['dtype'], v['shape'])

    def get(self, name, sl=None):
        mm, off, nb, dt, shp = self.idx[name]
        assert dt == 'BF16', dt
        a = np.asarray(mm[off:off + nb]).view('<u2').reshape(shp)
        if sl is not None:
            a = a[sl]
        return (a.astype(np.uint32) << 16).view('<f4')


E4 = None


def e4m3_table():
    t = np.zeros(256, np.float32)
    for i in range(256):
        s = -1.0 if i & 0x80 else 1.0
        e, m = (i >> 3) & 15, i & 7
        t[i] = np.nan if (e == 15 and m == 7) else s * ((m / 8 * 2 ** -6) if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return t


def fp8_rad(rf, name, r0=0, nr=None):
    e = rf.by_name[name]
    codes, pr, pc = rf.plane_rows(e, 0)
    sc = plane_values(rf, e, 1)
    global E4
    if E4 is None: E4 = e4m3_table()
    R, C = codes.shape
    nr = R - r0 if nr is None else nr
    b0, b1 = r0 // 128, (r0 + nr - 1) // 128 + 1
    c = E4[np.asarray(codes[b0 * 128:b1 * 128])]
    s = np.repeat(np.repeat(sc[b0:b1], 128, axis=0), 128, axis=1)[:c.shape[0], :C]
    return (c * s)[r0 - b0 * 128:r0 - b0 * 128 + nr]


def untile(x, axis, span_start, span, a, inner):
    b = span // (a * inner)
    idx = np.arange(span).reshape(a, b, inner).transpose(1, 0, 2).reshape(-1)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(span_start, span_start + span)
    y = x.copy()
    y[tuple(sl)] = x[tuple(sl)].take(idx, axis=axis)
    return y


class GG:
    def __init__(self, path):
        self.h = load_gguf(path)
        self.mm = np.memmap(path, dtype=np.uint8, mode='r')
        self.T = {t['name']: t for t in self.h['tensors']}

    def deq(self, name, expert=None, r0=0, nr=None):
        t = self.T[name]
        shp = list(reversed(t['shape']))
        cols = shp[-1]
        rows = int(np.prod(shp[:-1])) if len(shp) > 1 else 1
        qk, bb = BLOCK[t['type']]
        o = self.h['data_start'] + t['offset']
        raw = self.mm[o:o + t['nbytes']]
        if expert is not None:
            E = shp[0]; rows //= E; per = t['nbytes'] // E
            raw = raw[expert * per:(expert + 1) * per]
        nr = rows - r0 if nr is None else nr
        blk = raw.reshape(rows, cols // qk * bb)[r0:r0 + nr].reshape(-1, bb)
        return dequant_ref(t['type'], blk).reshape(nr, cols)


def rel(ref, rec):
    ref = ref.astype(np.float64); rec = rec.astype(np.float64)
    return float(np.sqrt(((ref - rec) ** 2).sum() / (ref ** 2).sum()))


def fit(name, W):
    """Route B candidate `name` fitted to the bf16 weights W (rows, K)."""
    Y = fwht128(W) if name in ROTATED else W
    rec = CANDS[name](Y, name in ROTATED)
    if rec is None:
        return None
    return fwht128(rec) if name in ROTATED else rec


def err(args):
    g = GG(args.gguf)
    st = Safetensors(args.bf16)
    rf = RadFile(args.fp8)
    L = int(g.h['kv']['qwen35moe.block_count']) - int(g.h['kv'].get('qwen35moe.nextn_predict_layers', 0))
    rows_out = []

    def case(label, gname, bf, gtf=None, expert=None, r0=0, nr=None, fp8=None, cands=None):
        t = g.T[gname]
        A = g.deq(gname, expert, r0, nr)
        if gtf: A = gtf(A)
        W = bf
        res = dict(label=label, ggml=t['type'], a_bpw=GG_BPW[t['type']], err_gguf=rel(W, A), n=int(W.size))
        for c in (cands or []):
            R = fit(c, W)
            if R is not None:
                res[f'err_{c}'] = rel(W, R); res[f'bpw_{c}'] = BPW[c]
        if fp8 is not None:
            res['err_fp8rad'] = rel(W, fp8)
        rows_out.append(res)
        print(json.dumps(res), flush=True)

    pre = 'model.language_model.layers.%d.'
    # routed experts
    for l, es in [(0, (0, 77)), (2, (0, 99, 200)), (20, (5, 150)), (38, (3, 222)), (1, (0, 99)), (34, (4, 100)), (39, (2, 120))]:
        for e in es:
            gu = st.get(pre % l + 'mlp.experts.gate_up_proj', e)       # (1024, 2048): gate then up
            dn = st.get(pre % l + 'mlp.experts.down_proj', e)          # (2048, 512)
            ty = g.T[f'blk.{l}.ffn_gate_exps.weight']['type']; td = g.T[f'blk.{l}.ffn_down_exps.weight']['type']
            c_gu = ['w4nl64', 'w5nl64'] if ty == 'Q4_K' else ['w5nl64']
            c_dn = ['w4nl64', 'w5nl64'] if td == 'Q5_K' else (['w5nl64', 'fp8b'] if td == 'Q6_K' else ['w5nl64'])
            f_gu = fp8_rad(rf, f'blk.{l}.ffn_gate_up_exps.{e}.weight'); f_dn = fp8_rad(rf, f'blk.{l}.ffn_down_exps.{e}.weight')
            case(f'L{l} e{e} gate', f'blk.{l}.ffn_gate_exps.weight', gu[:512], expert=e, fp8=f_gu[:512], cands=c_gu)
            case(f'L{l} e{e} up', f'blk.{l}.ffn_up_exps.weight', gu[512:], expert=e, fp8=f_gu[512:], cands=c_gu)
            case(f'L{l} e{e} down', f'blk.{l}.ffn_down_exps.weight', dn, expert=e, fp8=f_dn, cands=c_dn)
    # layer 40 = the MTP layer: official bits are fp8, GGUF is Q8_0
    for e in (0, 130):
        gu = st.get('mtp.layers.0.mlp.experts.gate_up_proj', e)
        f_gu = fp8_rad(rf, f'blk.40.ffn_gate_up_exps.{e}.weight')
        case(f'L40(MTP) e{e} gate', 'blk.40.ffn_gate_exps.weight', gu[:512], expert=e, fp8=f_gu[:512], cands=['fp8b', 'i8g128'])
    # dense trunk: GDN layers (value heads un-tiled on the GGUF side), attention, shared expert, lm_head
    for l in (0, 1, 20):
        qkv = st.get(pre % l + 'linear_attn.in_proj_qkv.weight')
        z = st.get(pre % l + 'linear_attn.in_proj_z.weight'); out = st.get(pre % l + 'linear_attn.out_proj.weight')
        inz = fp8_rad(rf, f'blk.{l}.ssm_inz.weight', 0, 12288)
        case(f'L{l} in_proj_qkv', f'blk.{l}.attn_qkv.weight', qkv, gtf=lambda x: untile(x, 0, 4096, 4096, 2, 128), fp8=inz[:8192], cands=['i8g128', 'fp8b'])
        case(f'L{l} in_proj_z', f'blk.{l}.attn_gate.weight', z, gtf=lambda x: untile(x, 0, 0, 4096, 2, 128), fp8=inz[8192:], cands=['i8g128', 'fp8b'])
        case(f'L{l} out_proj', f'blk.{l}.ssm_out.weight', out, gtf=lambda x: untile(x, 1, 0, 4096, 2, 128), fp8=fp8_rad(rf, f'blk.{l}.ssm_out.weight'), cands=['i8g128', 'fp8b'])
    for l in (3, 39):
        for hf, gn, dn in [('q_proj', 'attn_q', 'attn_qg'), ('k_proj', 'attn_k', 'attn_k'), ('v_proj', 'attn_v', 'attn_v'), ('o_proj', 'attn_output', 'attn_output')]:
            case(f'L{l} {hf}', f'blk.{l}.{gn}.weight', st.get(pre % l + f'self_attn.{hf}.weight'), fp8=fp8_rad(rf, f'blk.{l}.{dn}.weight'), cands=['i8g128', 'fp8b'])
    for l in (2, 30):
        ge = st.get(pre % l + 'mlp.shared_expert.gate_proj.weight'); up = st.get(pre % l + 'mlp.shared_expert.up_proj.weight')
        gup = fp8_rad(rf, f'blk.{l}.ffn_gate_up_shexp.weight')
        case(f'L{l} shexp gate', f'blk.{l}.ffn_gate_shexp.weight', ge, fp8=gup[:512], cands=['i8g128', 'fp8b'])
        case(f'L{l} shexp down', f'blk.{l}.ffn_down_shexp.weight', st.get(pre % l + 'mlp.shared_expert.down_proj.weight'), fp8=fp8_rad(rf, f'blk.{l}.ffn_down_shexp.weight'), cands=['i8g128', 'fp8b'])
    N = 16384
    lm = st.get('lm_head.weight', slice(0, N))
    case('lm_head rows 0-16383', 'output.weight', lm, nr=N, fp8=fp8_rad(rf, 'output.weight', 0, N), cands=['i8g128', 'fp8b'])
    case('lm_head rows 100000-116383', 'output.weight', st.get('lm_head.weight', slice(100000, 100000 + N)), r0=100000, nr=N,
         fp8=fp8_rad(rf, 'output.weight', 100000, N), cands=['i8g128', 'fp8b'])
    # summary by (ggml type, role group)
    import collections
    grp = collections.defaultdict(list)
    for r in rows_out:
        role = 'experts' if 'gate' in r['label'] and ' e' in r['label'] or ' up' in r['label'] or ' down' in r['label'] and ' e' in r['label'] else 'trunk'
        grp[(r['ggml'], role)].append(r)
    print('\nmean relative RMSE against bf16 (RTN + searched scale, no GPTQ, no imatrix: an upper bound for Route B)')
    keys = sorted({k for r in rows_out for k in r if k.startswith('err_')})
    print(f'{"ggml":6s} {"role":8s} {"n":>3s} ' + ' '.join(f'{k:>11s}' for k in keys))
    for (ty, role), rs in sorted(grp.items()):
        print(f'{ty:6s} {role:8s} {len(rs):3d} ' + ' '.join(f'{np.mean([r[k] for r in rs if k in r]):11.4f}' if any(k in r for r in rs) else f'{"-":>11s}' for k in keys))
    if args.out:
        json.dump(rows_out, open(args.out, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('plan'); p.add_argument('gguf'); p.add_argument('--recipe-out')
    e = sp.add_parser('err'); e.add_argument('gguf'); e.add_argument('--bf16', required=True); e.add_argument('--fp8', required=True); e.add_argument('--out')
    a = ap.parse_args()
    {'plan': plan, 'err': err}[a.cmd](a)


if __name__ == '__main__':
    main()
