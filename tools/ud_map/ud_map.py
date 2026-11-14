#!/usr/bin/env python3
"""ud_map -- produce a .rad from the bf16 weights with an Unsloth Dynamic GGUF as the MAP.

The GGUF decides, per tensor, which format (and therefore how many bits) Unsloth chose. This tool
does not requantise to an approximation by default: the recipe it emits uses libquant's `ggml`
quantiser with that same llama.cpp type (`type=iq2_s`, `type=q6_K`, ...), so the container carries
Unsloth's format at Unsloth's size, quantised from the original bf16 (with --imatrix for quality).

  ud_map.py plan  MAP.gguf [MORE SHARDS...] [--declared PLAN.json] [--source DIR]
                  [--recipe-out FILE] [--json FILE]

`--declared` is the plan JSON `rad-convert --plan-only --plan-json` wrote for the bf16 source
(the arch plugin's declared names/shapes). It is used to check that every mapped tensor has a
declared weight and to size the weights the GGUF does not carry (the MTP head and the vision
tower, which the UD text GGUF leaves out).

Read-only: nothing is written except the requested recipe/plan files.
"""
import argparse
import collections
import json
import os
import re

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ud_repack"))
from ud_repack import load_all  # noqa: E402

# GGUF type -> (bits per weight, libquant `ggml type=` spelling, servable by build 23 + qwen4exp_fp8)
# servable False means: libquant can write the format, but no kernel in the deployed build reads it.
TYPES = {
    "Q8_0":    (8.5,      "q8_0",   False),
    "Q6_K":    (6.5625,   "q6_k",   False),
    "Q5_K":    (5.5,      "q5_k",   False),
    "Q4_K":    (4.5,      "q4_k",   False),
    "Q5_0":    (5.5,      "q5_0",   False),
    "IQ4_NL":  (4.5,      "iq4_nl", False),
    "IQ4_XS":  (4.25,     "iq4_xs", False),
    "IQ3_XXS": (3.0625,   "iq3_xxs", False),
    "IQ2_S":   (2.5625,   "iq2_s",  False),
    "F32":     (32.0,     None,     True),
    "F16":     (16.0,     None,     True),
    "BF16":    (16.0,     None,     True),
}
UNKNOWN_TYPE = ("?", None, False)

# ---- the qwen4exp map: GGUF tensor name -> what the architecture declares it as.
# kind: "one" (1:1), "fused" (two GGUF tensors make one declared weight), "fused_into" (this one is
# the second half of a fused weight; its type must match its partner), "special" (packed/sliced).
def qwen4exp(name):
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if name == "per_layer_token_embd.weight":
        return dict(role="ple-table", kind="special", target="blk.1.ple_ngram.weight",
                    note="the PLE n-gram table; its 128 source shards are one declared weight")
    if name in ("token_embd.weight", "output.weight") or name.startswith("output_hc_"):
        return dict(role="output" if name.startswith("output") else "embedding", kind="one", target=name)
    if not m:
        return dict(role="other", kind="unknown", target=None, note="no map rule")
    l, r = m.group(1), m.group(2)
    b = f"blk.{l}."
    one = {
        "attn_q.weight": "attn_qg.weight", "attn_k.weight": "attn_k.weight",
        "attn_v.weight": "attn_v.weight", "attn_output.weight": "attn_output.weight",
        "attn_q_norm.weight": "attn_q_norm.weight", "attn_k_norm.weight": "attn_k_norm.weight",
        "ffn_gate_inp.weight": "ffn_gate_inp.weight",
        "ffn_gate_inp_shexp.weight": "ffn_gate_shexp.weight",
        "ssm_a": "ssm_a_log", "ssm_conv1d.weight": "ssm_conv1d.weight",
        "ssm_dt.bias": "ssm_dt.bias", "ssm_norm.weight": "ssm_norm.weight",
        "ssm_out.weight": "ssm_out.weight",
        "ple_conv1d.weight": "ple_conv.weight", "ple_key.weight": "ple_key.weight",
        "ple_norm_conv.weight": "ple_norm_conv.weight", "ple_norm_key.weight": "ple_norm_key.weight",
        "ple_norm_query.weight": "ple_norm_query.weight", "ple_value.weight": "ple_value.weight",
        "indexer.q_norm.weight": "qsa_q_norm.weight", "indexer.k_norm.weight": "qsa_k_norm.weight",
    }
    if r in one:
        return dict(role=r.split(".")[0], kind="one", target=b + one[r])
    fused = {
        "ffn_gate_shexp.weight": ("ffn_gate_up_shexp.weight", "ffn_up_shexp.weight"),
        "ffn_up_shexp.weight": ("ffn_gate_up_shexp.weight", "ffn_gate_shexp.weight"),
        "ssm_alpha.weight": ("ssm_ab.weight", "ssm_beta.weight"),
        "ssm_beta.weight": ("ssm_ab.weight", "ssm_alpha.weight"),
        "indexer.q_proj.weight": ("qsa_qk.weight", "indexer.k_proj.weight"),
        "indexer.k_proj.weight": ("qsa_qk.weight", "indexer.q_proj.weight"),
    }
    if r in fused:
        return dict(role=r.split(".")[0], kind="fused", target=b + fused[r][0], partner=b + fused[r][1])
    if r in ("attn_qkv.weight", "attn_gate.weight"):
        partner = b + ("attn_gate.weight" if r == "attn_qkv.weight" else "attn_qkv.weight")
        return dict(role="delta-attn", kind="fused", target=b + "ssm_inz.weight", partner=partner)
    if r in ("ffn_gate_exps.weight", "ffn_up_exps.weight"):
        partner = b + ("ffn_up_exps.weight" if r == "ffn_gate_exps.weight" else "ffn_gate_exps.weight")
        return dict(role="routed-expert gate/up", kind="fused", target=b + "ffn_gate_up_exps.*.weight",
                    partner=partner)
    if r == "ffn_down_exps.weight":
        return dict(role="routed-expert down", kind="one", target=b + "ffn_down_exps.*.weight")
    if r == "ffn_down_shexp.weight":
        return dict(role="shared-expert", kind="one", target=b + r)
    hc = re.fullmatch(r"hc_(attn|ffn)_(down|inject|norm|up)\.weight", r)
    if hc:
        return dict(role="gated-residual", kind="one", target=b + f"{hc.group(1)}_hc_{hc.group(2)}.weight")
    return dict(role=r.split(".")[0], kind="unknown", target=None, note="no map rule")


ARCHES = {"qwen4exp": qwen4exp}


def _declared_index(path):
    ws = json.load(open(path))["weights"]
    names = [w["name"] for w in ws]
    return ws, names


def _covered(name, targets):
    """Does a declared name match any target pattern (with '*' as the expert index)?"""
    for t in targets:
        if "*" in t:
            rx = "^" + re.escape(t).replace(r"\*", r"\d+") + "$"
            if re.match(rx, name):
                return True
        elif t == name:
            return True
    return False


def plan(args):
    kv, ts = load_all(args.gguf)
    arch = kv.get("general.architecture", "?")
    mapper = ARCHES.get(arch)
    if mapper is None:
        sys.exit(f"ud_map: no name map for architecture '{arch}' (have: {', '.join(ARCHES)})")

    rows, fused = [], collections.defaultdict(list)
    for t in ts:
        m = mapper(t["name"])
        rows.append(dict(name=t["name"], type=t["type"], shape=t["shape"], bytes=t["nbytes"],
                         role=m.get("role"), target=m.get("target"), kind=m.get("kind"),
                         note=m.get("note", "")))
        if m.get("kind") in ("fused", "fused_into") and m.get("target"):
            fused[m["target"]].append((t["name"], t["type"]))
        elif m.get("kind") == "fused":
            fused[m["target"]].append((t["name"], t["type"]))

    # fused weights: every partner must agree on the type, else the container cannot mirror both
    problems = []
    for target, members in fused.items():
        types = {ty for _, ty in members}
        if len(types) > 1:
            problems.append(dict(target=target, members=members, why="fused tensors disagree on type"))

    ws, names = (None, [])
    if args.declared:
        ws, names = _declared_index(args.declared)
    name_set = set(names)
    for r in rows:
        r["servable"] = TYPES.get(r["type"], UNKNOWN_TYPE)[2]
        tgt = r["target"]
        r["declared_found"] = None
        if tgt and name_set:
            if "*" in tgt:
                rx = "^" + re.escape(tgt).replace(r"\*", r"\d+") + "$"
                r["declared_found"] = sum(1 for n in names if re.match(rx, n))
            else:
                r["declared_found"] = 1 if tgt in name_set else 0
        if r["kind"] == "unknown":
            r["servable"] = False

    # declared weights the map does not cover: the MTP head and the vision tower (the UD text GGUF
    # leaves them out), plus i64 metadata. They are kept from the bf16 source.
    kept = []
    if ws is not None:
        targets = [r["target"] for r in rows if r["target"]]
        for w in ws:
            if _covered(w["name"], targets):
                continue
            kept.append(dict(name=w["name"], encoding=w["encoding"], bytes=w["bytes"],
                             shape=w["shape"]))
    unmapped = [r["name"] for r in rows if r["kind"] == "unknown"]

    map_bytes = sum(r["bytes"] for r in rows)
    map_bits = sum(r["bytes"] * 8 for r in rows)
    map_elems = sum((r["shape"][0] if r["shape"] else 0) * (r["shape"][1] if len(r["shape"]) > 1 else 1)
                    * (r["shape"][2] if len(r["shape"]) > 2 else 1) for r in rows)
    kept_bytes = sum(k["bytes"] for k in kept)
    total = map_bytes + kept_bytes

    def norm(t):
        return re.sub(r"blk\.\d+\.", "blk.*.", t) if t else t

    by = collections.OrderedDict()
    for r in rows:
        k = (r["role"], r["type"], norm(r["target"]), r["servable"])
        a = by.setdefault(k, [0, 0])
        a[0] += 1
        a[1] += r["bytes"]
    # map coverage: for every distinct target, does the declared plan carry it (and how many)?
    cov = {}
    for r in rows:
        if not r["target"]:
            continue
        cov.setdefault(r["target"], r.get("declared_found") or 0)
    missing = {t: n for t, n in cov.items() if n == 0}
    print(f"{arch}: {len(rows)} mapped tensors, {map_bytes/2**30:.3f} GiB in Unsloth's formats, "
          f"{kept_bytes/2**30:.3f} GiB kept from the bf16 source (not in the map), "
          f"total {total/2**30:.3f} GiB")
    print(f'{"role":24s}{"ggml":9s}{"target":30s}{"servable":9s}{"n":>4s}{"GiB":>9s}')
    for (role, ty, tgt, ok), (n, b) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f'{(role or "?"):24s}{ty:9s}{(tgt or "-"):30s}{"yes" if ok else "NO":9s}'
              f'{n:4d}{b/2**30:9.3f}')
    print(f'\nmap coverage: {len(cov)-len(missing)}/{len(cov)} distinct declared targets present')
    if missing:
        groups = collections.Counter(re.sub(r'^blk\.\d+\.', 'blk.*.', t) for t in missing)
        print(f'  no declared weight for: ' + ', '.join(f'{k} x{n}' for k, n in groups.most_common(6)))
    ok_bytes = sum(r["bytes"] for r in rows if r["servable"])
    print(f'servable by the current build: {ok_bytes/2**30:.3f} GiB ({100*ok_bytes/total:.1f}% of the total); '
          f'blocked on kernels: {(total-ok_bytes)/2**30:.3f} GiB')
    tsum = sorted(((ty, sum(r["bytes"] for r in rows if r["type"] == ty))
                   for ty in {r["type"] for r in rows}), key=lambda x: -x[1])
    print('  by type (GiB): ' + ', '.join(f'{ty} {b/2**30:.2f}' for ty, b in tsum))
    if kept:
        print(f'\nkept from bf16 (no map entry): {kept_bytes/2**30:.3f} GiB in {len(kept)} weights')
        def pre(name):
            m = re.match(r'(blk\.\d+|v|mtp|[a-z_]+)', name)
            return m.group(1) if m else name
        kb = collections.Counter((k["encoding"], pre(k["name"])) for k in kept)
        for (enc, pre_), n in kb.most_common(10):
            gb = sum(k["bytes"] for k in kept if pre(k["name"]) == pre_) / 2**30
            print(f'  {pre_:12s} {enc:10s} {n:4d} weight(s) {gb:7.3f} GiB')
    if problems:
        print("\nFUSION PROBLEMS (the container cannot mirror both halves):")
        for p in problems:
            print(" ", p)
    if unmapped:
        print(f"\nunmapped GGUF tensors ({len(unmapped)}): {unmapped[:12]}")

    if args.recipe_out:
        lines = ["# Generated by tools/ud_map/ud_map.py from %s" % os.path.basename(args.gguf[-1]),
                 "# The GGUF is the MAP: `ggml type=X` reproduces Unsloth's format from the bf16 source.",
                 "# A weight with no rule keeps the checkpoint (bf16). Pass --imatrix for IQ quality.",
                 ""]
        written = set()
        # per-layer expert rules (the map's mix differs by layer)
        for r in rows:
            if "exps" not in (r["target"] or "") or "*" not in (r["target"] or ""):
                continue
            key = (r["target"], r["type"])
            if key in written:
                continue
            written.add(key)
            spell = TYPES.get(r["type"], UNKNOWN_TYPE)[1]
            if spell is None:
                lines.append(f'# {r["target"]}: {r["type"]} has no libquant type (unknown/keep)')
                continue
            lines.append(f'{r["target"]:38s} ggml type={spell}   # {r["type"]}')
        # one rule per (target, type) for the rest, exact targets only
        for r in rows:
            tgt = r["target"]
            if not tgt or "*" in tgt or (tgt, r["type"]) in written:
                continue
            spell = TYPES.get(r["type"], UNKNOWN_TYPE)[1]
            if spell is None:
                continue  # F32/BF16: keep; unknown types: no rule
            written.add((tgt, r["type"]))
            lines.append(f'{tgt:38s} ggml type={spell}   # {r["type"]}')
        open(args.recipe_out, "w").write("\n".join(lines) + "\n")

    if args.json:
        json.dump(dict(tool="ud_map 0.1", arch=arch, policy="mirror", map=list(args.gguf),
                       declared_plan=args.declared, source=args.source,
                       totals=dict(map_bytes=map_bytes, kept_bytes=kept_bytes, total_bytes=total,
                                   map_bpw=(map_bits / map_elems if map_elems else 0)),
                       tensors=rows, kept=kept, unmapped=unmapped, fusion_problems=problems),
                  open(args.json, "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("gguf", nargs="+", help="the GGUF, all shards")
    p.add_argument("--declared", help="rad-convert --plan-only --plan-json output for the bf16 source")
    p.add_argument("--source", help="the bf16 source directory (recorded in the plan)")
    p.add_argument("--recipe-out")
    p.add_argument("--json")
    p.set_defaults(fn=plan)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
