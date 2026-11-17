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
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ud_repack"))
from ud_repack import load_all, role_of  # noqa: E402

# GGUF type -> (bits per weight, libquant `ggml type=` spelling, servable by the current build)
# servable False means: libquant can write the format, but no kernel/arch path in the deployed
# build reads it. The MoE lane (base 2a3200f) serves IQ4_NL (iq4nl32a16 + the PLE gather), Q5_0
# (q50a16) and Q2_0 (q20a16). Q6_K/Q8_0 have dense kernels but the qwen4exp arch does not wire
# them yet (lane L3); IQ3_XXS/IQ2_S/IQ4_XS still lack kernels.
TYPES = {
    "Q8_0":    (8.5,      "q8_0",   False),
    "Q6_K":    (6.5625,   "q6_k",   False),
    "Q5_K":    (5.5,      "q5_k",   False),
    "Q4_K":    (4.5,      "q4_k",   False),
    "Q5_0":    (5.5,      "q5_0",   True),
    "IQ4_NL":  (4.5,      "iq4_nl", True),
    "IQ4_XS":  (4.25,     "iq4_xs", False),
    "IQ3_XXS": (3.0625,   "iq3_xxs", False),
    "IQ2_S":   (2.5625,   "iq2_s",  False),
    # llama.cpp type 42 is Q2_0: 64 values, f16 scale, 2-bit codes (-1,0,+1,+2), 2.25 bpw.
    # libquant's `q2_0` kind and the q20a16 expert GEMM exist; the GGUF names it TYPE42.
    "TYPE42":  (2.25,     "q2_0",   True),
    "Q2_0":    (2.25,     "q2_0",   True),
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


def qwen35moe(name):
    """Qwen3.5/3.6 MoE (35B-A3B): the GGUF tensor -> the name arch/qwen35moe_fp8 declares.

    The fp8 container fuses what llama.cpp keeps apart, exactly as the hand-built q36-w4w5
    recipes show: `attn_qkv` + `attn_gate` -> `ssm_inz`, `ssm_alpha` + `ssm_beta` -> `ssm_ab`,
    the routed `gate` + `up` -> `ffn_gate_up_exps.E`, the shared `gate` + `up` ->
    `ffn_gate_up_shexp`, and llama.cpp's `ffn_gate_inp_shexp` -> the shared gate
    `ffn_gate_shexp`. The MTP layer is GGUF `blk.40.*`; its four `nextn` tensors are the fp8
    container's `mtp.*` head.
    """
    if name in ("token_embd.weight", "output.weight", "output_norm.weight"):
        return dict(role="output" if name == "output.weight" else
                    "embedding" if name == "token_embd.weight" else "norm",
                    kind="one", target=name)
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return dict(role="other", kind="unknown", target=None, note="no map rule")
    l, r = m.group(1), m.group(2)
    b = f"blk.{l}."
    if l == "40" and r.startswith("nextn."):
        mtp = {"nextn.eh_proj.weight": "mtp.fc.weight",
               "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
               "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
               "nextn.shared_head_norm.weight": "mtp.norm.weight"}
        if r in mtp:
            return dict(role="mtp", kind="one", target=mtp[r])
    one = {
        "attn_q.weight": "attn_qg.weight", "attn_k.weight": "attn_k.weight",
        "attn_v.weight": "attn_v.weight", "attn_output.weight": "attn_output.weight",
        "attn_q_norm.weight": "attn_q_norm.weight", "attn_k_norm.weight": "attn_k_norm.weight",
        "attn_norm.weight": "attn_norm.weight", "post_attention_norm.weight": "ffn_norm.weight",
        "ssm_a": "ssm_a_log", "ssm_conv1d.weight": "ssm_conv1d.weight",
        "ssm_dt.bias": "ssm_dt.bias", "ssm_norm.weight": "ssm_norm.weight",
        "ssm_out.weight": "ssm_out.weight", "ffn_gate_inp.weight": "ffn_gate_inp.weight",
        "ffn_gate_inp_shexp.weight": "ffn_gate_shexp.weight",
        "ffn_down_shexp.weight": "ffn_down_shexp.weight",
        "ffn_down_exps.weight": "ffn_down_exps.*.weight",
    }
    if r in one:
        return dict(role=r.split(".")[0], kind="one", target=b + one[r])
    fused = {
        "attn_qkv.weight": ("ssm_inz.weight", "attn_gate.weight"),
        "attn_gate.weight": ("ssm_inz.weight", "attn_qkv.weight"),
        "ssm_alpha.weight": ("ssm_ab.weight", "ssm_beta.weight"),
        "ssm_beta.weight": ("ssm_ab.weight", "ssm_alpha.weight"),
        "ffn_gate_exps.weight": ("ffn_gate_up_exps.*.weight", "ffn_up_exps.weight"),
        "ffn_up_exps.weight": ("ffn_gate_up_exps.*.weight", "ffn_gate_exps.weight"),
        "ffn_gate_shexp.weight": ("ffn_gate_up_shexp.weight", "ffn_up_shexp.weight"),
        "ffn_up_shexp.weight": ("ffn_gate_up_shexp.weight", "ffn_gate_shexp.weight"),
    }
    if r in fused:
        return dict(role=r.split(".")[0], kind="fused", target=b + fused[r][0],
                    partner=b + fused[r][1])
    return dict(role=r.split(".")[0], kind="unknown", target=None, note="no map rule")


ARCHES = {"qwen4exp": qwen4exp, "qwen35moe": qwen35moe}

# Route B's fast formats, for an architecture that serves the existing kernels rather than the
# GGUF's own planes. The GGUF still decides the bits: a Q4_K expert becomes w4nl64a8h, a
# Q5_K/Q6_K one w5nl64a8h, and the dense Q8_0 trunk the served block FP8. (The codebooks are not
# the GGUF's K-quants; they are the nearest form libr4d has a kernel for, which is what the
# hand-built q36-w4w5 recipes use.)
#
# The Flash-Next (qwen4exp) map adds the UD v3.1 types: IQ4_NL/IQ4_XS -> the rotated w4nl
# codebook (4.125 bpw, smaller than IQ4_NL's 4.5 and the production Flash-Next format), Q5_0 ->
# w5nl, and IQ3_XXS/IQ2_S/Q2_0 -> w3nl (the existing 3-bit codebook). Q2_0's native q2_0 plane is
# smaller (2.25 bpw, served as q20a16), but its only tensors are blk.8/blk.10 down projections
# whose gate/up are IQ2_S; MoeFP8 refuses an unrotated q20a16 table beside a rotated codebook in
# one layer ("one table of experts takes one encoding"), so Q2_0 takes w3nl too (+0.17 GiB).
_FAST_W4NL = ("rtn table=w4nl group=64 scale=fp8_e4m3 scale2=f32 block2=*x* "
              "scale2_value=0.0001220703125 transform=fwht128 rule=search")
_FAST_W5NL = ("rtn table=w5nl group=64 scale=fp8_e4m3 scale2=f32 block2=*x* "
              "scale2_value=0.0001220703125 transform=fwht128 rule=search")
_FAST_W3NL = ("rtn table=w3nl group=64 scale=fp8_e4m3 scale2=f32 block2=*x* "
              "scale2_value=0.0001220703125 transform=fwht128 rule=search")
FAST_EXPERTS = {
    "Q4_K":    ("w4nl64", _FAST_W4NL),
    "Q5_K":    ("w5nl64", _FAST_W5NL),
    "Q6_K":    ("w5nl64", _FAST_W5NL),
    "Q8_0":    ("w5nl64", _FAST_W5NL),
    "IQ4_NL":  ("w4nl64", _FAST_W4NL),
    "IQ4_XS":  ("w4nl64", _FAST_W4NL),
    "Q5_0":    ("w5nl64", _FAST_W5NL),
    "IQ3_XXS": ("w3nl64", _FAST_W3NL),
    "IQ2_S":   ("w3nl64", _FAST_W3NL),
    "Q2_0":    ("w3nl64", _FAST_W3NL),
    "TYPE42":  ("w3nl64", _FAST_W3NL),
}
# Bits per weight of each fast form, including its scale planes (the fp8/f32 scale overhead is
# folded in: 4.125, 5.125 and 3.125 are 4/5/3 code bits plus one E4M3 byte per 64 of K; fp8 is
# block fp8 at bf16 128x128 scales; q2_0 is 2 bits plus f16 per 64).
FAST_BPW = {
    "w4nl64": 4.125,
    "w5nl64": 5.125,
    "w3nl64": 3.125,
    "q20a16": 2.25,
    "q6_k":   6.5625,
    "q8_0":   8.5,
    "iq4_nl": 4.5,
    "fp8":    8.0 + 16.0 / (128 * 128),
    "f32":    32.0,
    "f16":    16.0,
    "bf16":   16.0,
}
# qwen35moe's fast recipe only ever remaps the K-quants; the qwen4exp arch additionally serves
# the IQ/4-bit codebooks and Q2_0. Keep the old arch's set narrow so its behaviour is unchanged.
_FAST_EXPERT_QWEN35 = ("Q4_K", "Q5_K", "Q6_K", "Q8_0")
_Y_KEEP = ("F32", "F16", "BF16")
_FAST_TRUNK = "rtn codes=fp8_e4m3 block=128x128 scale=bf16"
FAST_TRUNK = _FAST_TRUNK
FAST_DRAFT = "rtn codes=u2 group=128 scale=f16 zero=u8"
BF16_TO_FAST_QUANT = "fp8"

# Formats a *kept* (not-in-the-map) weight can be quantised to, in bits/weight. `keep` leaves it as
# the checkpoint holds it. Used only for the MTP layer and the vision tower, whose formats the UD
# text GGUF does not decide.
KEPT_FORMATS = {"keep": None, "bf16": 16.0, "fp8": 8.0, "q8_0": 8.5,
                "w4nl64": 4.125, "w5nl64": 5.125}


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


def _kept_group(name):
    if name.startswith("blk.48.") or name.startswith("mtp."):
        return "mtp"
    if name.startswith("v."):
        return "vision"
    return "other"


def _kept_bytes(w, fmt):
    """Bytes of a kept weight under a target format. 1-D weights (norms, biases), integer data and
    small tensors stay as stored: the arch does not quantise them."""
    bpw = KEPT_FORMATS.get(fmt)
    shp = w.get("shape") or []
    quantisable = (bpw is not None and w["encoding"] == "bf16" and len(shp) >= 2 and
                   min(shp) >= 16)
    if not quantisable:
        return w["bytes"]
    elems = w["bytes"] / 2          # bf16 source
    return int(elems * bpw / 8)


# The two mirrors the arch serves through a *different* encoding. Q6_K output.weight has no
# logits_gemm row (the op carries one weight operand; Q6_K needs three planes), so the head goes to
# the served int8 form: 497 MiB -> 615 MiB (+118 MiB, +0.115 GiB in the 89 GiB container) and the
# int8 logits kernel is bit-for-bit its own plane values. The connection mixing matrices are read
# by hc_read, which serves bf16 or E4M3 rows, not Q8_0; E4M3 + f32 per row group is the served
# form and is essentially the same size (the Q8_0 and E4M3 codes are both 8 bits, the f32 scales
# add ~0.1 MiB per matrix). Both are lossy where the GGUF was exact; keep the GGUF planes when a
# container is only measured, not served.
Q6K_LM_HEAD_RULE = "rtn  codes=i8 group=128 scale=bf16 clamp=sym rule=search"
HC_E4M3_DOWN_RULE = "rtn  codes=fp8_e4m3 group=128 scale=f32"
HC_E4M3_UP_RULE = "rtn  codes=fp8_e4m3 group=80 scale=f32"


def _mirror_override(target, ty):
    """(pattern, rule, comment) for a map tensor whose format the arch serves another way."""
    if target == "output.weight" and ty == "Q6_K":
        return ("output.weight", Q6K_LM_HEAD_RULE,
                "# Q6_K output.weight -> i8: no logits_gemm q6k row (three planes, one operand); "
                "497 MiB -> 615 MiB")
    if ty == "Q8_0" and target and target.endswith("_hc_down.weight"):
        return ("*_hc_down.weight", HC_E4M3_DOWN_RULE,
                "# Q8_0 connection mixing -> E4M3 rows (hc_read serves bf16/e4m3, not Q8_0)")
    if ty == "Q8_0" and target and target.endswith("_hc_up.weight"):
        return ("*_hc_up.weight", HC_E4M3_UP_RULE,
                "# Q8_0 connection mixing -> E4M3 rows (hc_read serves bf16/e4m3, not Q8_0)")
    return None


def _mirror_recipe(rows, gguf, mtp=None, mtp_layer=None):
    lines = ["# Generated by tools/ud_map/ud_map.py from %s" % os.path.basename(gguf),
             "# The GGUF is the MAP: `ggml type=X` reproduces Unsloth's format from the bf16 source.",
             "# A weight with no rule keeps the checkpoint (bf16). Pass --imatrix for IQ quality.",
             ""]
    written = set()
    skipped = set()
    for r in rows:
        ov = _mirror_override(r.get("target"), r["type"])
        if not ov or ov[0] in skipped:
            continue
        skipped.add(ov[0])
        lines.append(ov[2])
        lines.append(f"{ov[0]:38s} {ov[1]}")
        written.add((ov[0], r["type"]))
    if skipped:
        lines.append("")
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
        ov = _mirror_override(tgt, r["type"])
        if ov and (ov[0] == tgt or (ov[0].startswith("*") and tgt.endswith(ov[0][1:]))):
            continue
        spell = TYPES.get(r["type"], UNKNOWN_TYPE)[1]
        if spell is None:
            continue  # F32/BF16: keep; unknown types: no rule
        written.add((tgt, r["type"]))
        lines.append(f'{tgt:38s} ggml type={spell}   # {r["type"]}')
    if mtp in ("w4nl64", "w5nl64") and mtp_layer is not None:
        rule = _expert_rule("Q4_K" if mtp == "w4nl64" else "Q5_K", False)
        for pattern in ("ffn_gate_up_exps", "ffn_down_exps"):
            lines.append(f"blk.{mtp_layer}.{pattern}.*.weight".ljust(40) + " " + rule +
                         f"   # kept MTP layer -> {mtp}")
    return "\n".join(lines) + "\n"


# The dense trunk the qwen35moe_fp8 container serves as block FP8, and the draft head it derives
# from output.weight. The source is the bf16 release, so the recipe is what q36-w4w5-rtn.recipe
# states by hand; only the expert rules follow the map's per-layer numbers.
_FAST_DENSE = ("attn_qg.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
               "ssm_inz.weight", "ssm_out.weight", "ffn_gate_up_shexp.weight",
               "ffn_down_shexp.weight")


def _expert_rule(fmt, gptq):
    entry = FAST_EXPERTS.get(fmt)
    if entry is None:
        return None
    rule = entry[1]
    if gptq and rule.startswith("rtn "):
        rule = rule.replace("rtn ", "gptq ", 1) + " cd=3 calib=$CALIB"
    return rule


def _fast_dtype(arch, row):
    """The fast-path form of one map row: (dtype label, bits/weight, servable now, dependency).

    None means no served form: the caller reports it rather than guessing. `servable` is the
    current build (base 2a3200f): the MoE codebooks w4nl64a8h/w5nl64a8h/q20a16 and the native
    IQ4_NL PLE gather are wired; w3nl64a8h has a kernel but not yet a qwen4exp arch format, and
    the native Q6_K/Q8_0 dense forms wait on lane L3.
    """
    ty = row.get("type")
    tgt = row.get("target") or ""
    role = row.get("role") or ""
    if arch == "qwen4exp":
        # The 26.8 GiB PLE n-gram table is a gather, not a GEMM. Keep the GGUF's own IQ4_NL:
        # that is the kernel the PLE lane wired (there is no w4nl64 gather), and it is smaller
        # than a block-FP8 table.
        if role == "ple-table" or tgt.endswith("ple_ngram.weight"):
            return ("iq4_nl", FAST_BPW["iq4_nl"], True, None)
        if "exps." in tgt:
            entry = FAST_EXPERTS.get(ty)
            if entry is None:
                return None
            label = entry[0]
            if label == "w3nl64":
                return (label, FAST_BPW[label], False,
                        "arch: qwen4exp format_of/kServed/probe for w3nl64a8h (L1 kernel exists)")
            return (label, FAST_BPW[label], True, None)
        if role == "embedding" or tgt == "token_embd.weight":
            # The early path keeps the 4.25-bpw token embedding as the checkpoint's bf16
            # (1.186 GiB). A q8_0 form is 0.630 GiB, but it needs L3's embedding wiring.
            return ("bf16", FAST_BPW["bf16"], True, None)
        if ty == "Q6_K":
            return ("q6_k", FAST_BPW["q6_k"], False, "L3: qwen4exp dense Q6_K wiring")
        if ty == "Q8_0":
            return ("q8_0", FAST_BPW["q8_0"], False, "L3: qwen4exp dense Q8_0 wiring")
        if ty in _Y_KEEP:
            return (ty.lower(), FAST_BPW[ty.lower()], True, None)
        return None
    # qwen35moe: the Route B forms the hand-built q36-w4w5 recipes use.
    if "*" in tgt:
        if ty not in _FAST_EXPERT_QWEN35:
            return None
        label = FAST_EXPERTS[ty][0]
        return (label, FAST_BPW[label], True, None)
    if ty == "Q8_0":
        return ("fp8", FAST_BPW["fp8"], True, None)
    if ty in _Y_KEEP:
        return (ty.lower(), FAST_BPW[ty.lower()], True, None)
    return None


def _fast_bytes(row, label, bpw):
    """The row's size in a fast form. A form the GGUF already stores is exactly its stored bytes;
    a remap is elements * bpw / 8 (the scale planes are in FAST_BPW)."""
    ty = row.get("type")
    native = {"iq4_nl": ("IQ4_NL",), "q2_0": ("Q2_0", "TYPE42"),
              "q6_k": ("Q6_K",), "q8_0": ("Q8_0",),
              "f32": ("F32",), "f16": ("F16",), "bf16": ("BF16",)}.get(label, ())
    if ty in native:
        return row["bytes"]
    elems = 1
    for x in row.get("shape") or []:
        elems *= x
    if elems <= 0:
        return row["bytes"]
    return int(elems * bpw / 8)


def _fast_recipe(rows, gguf, gptq=False, mtp=None, arch="qwen35moe", mtp_layer=None):
    """The Route B recipe for arch/qwen35moe_fp8 or arch/qwen4exp: the GGUF decides the bits, the
    recipe picks the nearest format the architecture serves."""
    if arch == "qwen4exp":
        return _fast_recipe_4exp(rows, gguf, gptq=gptq, mtp=mtp, mtp_layer=mtp_layer)
    lines = ["# Generated by tools/ud_map/ud_map.py from %s" % os.path.basename(gguf),
             "# The GGUF is the MAP. The routed experts take the nearest fast codebook format to",
             "# Unsloth's K-quant (Q4_K -> w4nl64a8h, Q5_K/Q6_K/Q8_0 -> w5nl64a8h); the dense Q8_0",
             "# trunk takes the served block FP8. The source is the bf16 release, so convert with",
             "# --set quantization_config.quant_method=fp8. Everything not named keeps the",
             "# checkpoint, as the FP8 container does (token_embd, norms, router, ssm_*, mtp.fc).",
             ""]
    lines.append("# routed experts: one rule per layer, because the map's mix differs by layer")
    expert = collections.defaultdict(dict)
    for r in rows:
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", r["name"])
        if not m:
            continue
        layer, kind = int(m.group(1)), "down" if m.group(2) == "down" else "gate_up"
        expert[layer].setdefault(kind, []).append(r["type"])
    for layer in sorted(expert):
        e = expert[layer]
        for kind, pattern in (("gate_up", "ffn_gate_up_exps"), ("down", "ffn_down_exps")):
            if kind not in e:
                continue
            ty = max(e[kind], key=lambda x: TYPES.get(x, UNKNOWN_TYPE)[0] or 0)
            if layer == 40 and mtp not in (None, "", "keep"):
                ty = {"bf16": "Q8_0", "fp8": "Q8_0", "q8_0": "Q8_0",
                      "w4nl64": "Q4_K", "w5nl64": "Q5_K"}.get(mtp, ty)
            rule = _expert_rule(ty, gptq)
            if rule is None:
                lines.append(f"# blk.{layer}.{pattern}: {ty} has no fast format")
                continue
            lines.append(f"blk.{layer}.{pattern}.*.weight".ljust(40) + " " + rule +
                         f"   # {ty}")
    lines.append("")
    lines.append("# dense trunk, exactly as the FP8 container holds it (block FP8)")
    written = set()
    for r in rows:
        tgt = r.get("target")
        if not tgt or "*" in tgt:
            continue
        base = re.sub(r"^blk\.\d+\.", "", tgt)
        if base not in _FAST_DENSE or base in written:
            continue
        if r["type"] != "Q8_0" or (r.get("shape") and len(r["shape"]) < 2):
            continue
        written.add(base)
        lines.append(f"blk.*.{base}".ljust(40) + " " + FAST_TRUNK + f"   # {r['type']}")
    if "output.weight" not in written:
        lines.append("output.weight".ljust(40) + " " + FAST_TRUNK + "   # Q8_0")
    lines.append("")
    lines.append("# MTP draft head, derived from the (quantised) output.weight")
    lines.append("mtp.draft_head.weight".ljust(40) + " " + FAST_DRAFT)
    return "\n".join(lines) + "\n"


def _expert_choices(rows):
    """Per-layer choice between the GGUF-native family and the rotated codebook family.

    MoeFP8 keeps one encoding family per layer, so the choice is made for the layer's gate/up and
    down projections together. The native family is exactly the UD map; the rotated family is the
    FAST_EXPERTS mapping. The measured .rad sizes show native IQ3_XXS carries ~5% extra plane
    overhead and w3nl is smaller there, while IQ2_S/Q2_0 are genuinely smaller native; the
    per-layer comparison still picks the family with the smaller total.
    """
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", r["name"])
        if not m:
            continue
        layer, kind = int(m.group(1)), "down" if m.group(2) == "down" else "gate_up"
        groups[layer][kind].append(r)
    out = {}
    for layer, kinds in groups.items():
        native = sum(r["bytes"] for rs in kinds.values() for r in rs)
        fast = 0
        for rs in kinds.values():
            for r in rs:
                entry = FAST_EXPERTS.get(r["type"])
                if entry is None:
                    fast += r["bytes"]
                else:
                    label = entry[0]
                    fast += _fast_bytes(r, label, FAST_BPW[label])
        out[layer] = dict(kinds=kinds, native=native, fast=fast, use_fast=fast < native)
    return out


def _size_map_bytes(rows):
    """The map's size after the per-layer family choice."""
    saved = sum(c["native"] - c["fast"] for c in _expert_choices(rows).values()
                if c["use_fast"])
    return sum(r["bytes"] for r in rows) - saved


def _size_recipe_4exp(rows, gguf, mtp=None, mtp_layer=None):
    """The size policy: one rule per layer, choosing the smaller encoding family.

    The mirror recipe is reused for every non-expert rule; its expert rules are removed first so
    the chosen layer rule is the only one for that target. Mixing native and rotated projections
    inside one layer is not used: MoeFP8 refuses it.
    """
    mirror = _mirror_recipe(rows, gguf, mtp=mtp, mtp_layer=mtp_layer).splitlines()
    expert_rule = re.compile(r"^blk\.\d+\.ffn_(?:gate_up|down)_exps\.\*\.weight\s")
    lines = [line for line in mirror if not expert_rule.match(line)]
    lines.append("")
    lines.append("# SIZE path: one rule per expert layer, the smaller of native and rotated")
    for layer, c in sorted(_expert_choices(rows).items()):
        for kind, pattern in (("gate_up", "ffn_gate_up_exps"), ("down", "ffn_down_exps")):
            if kind not in c["kinds"]:
                continue
            ty = max({r["type"] for r in c["kinds"][kind]},
                     key=lambda x: TYPES.get(x, UNKNOWN_TYPE)[0] or 0)
            if c["use_fast"]:
                rule = _expert_rule(ty, False)
                if rule is None:
                    continue
                lines.append(f"blk.{layer}.{pattern}.*.weight".ljust(40) + " " + rule +
                             f"   # {ty} -> {FAST_EXPERTS[ty][0]} (size)")
            else:
                spell = TYPES.get(ty, UNKNOWN_TYPE)[1]
                if spell is None:
                    continue
                lines.append(f"blk.{layer}.{pattern}.*.weight".ljust(40) +
                             f" ggml type={spell}   # {ty} (native, size)")
    if mtp in ("w4nl64", "w5nl64") and mtp_layer is not None:
        rule = _expert_rule("Q4_K" if mtp == "w4nl64" else "Q5_K", False)
        for pattern in ("ffn_gate_up_exps", "ffn_down_exps"):
            lines.append(f"blk.{mtp_layer}.{pattern}.*.weight".ljust(40) + " " + rule +
                         f"   # kept MTP layer -> {mtp}")
    return "\n".join(lines) + "\n"

def _fast_recipe_4exp(rows, gguf, gptq=False, mtp=None, mtp_layer=None):
    """The Flash-Next Route B recipe: UD formats -> the kernels the build already serves.

    Routed experts take the nearest rotated codebook form (IQ4_NL/IQ4_XS -> w4nl64a8h, Q5_0 ->
    w5nl64a8h, IQ3_XXS/IQ2_S/Q2_0 -> w3nl64a8h). Q2_0's native q20a16 is smaller but unrotated,
    and its layers' gate/up are IQ2_S, so MoeFP8's one-encoding-per-layer rule keeps it in the
    codebook family. The dense Q6_K/Q8_0 trunk keeps its own dense formats (the kernels exist;
    the qwen4exp arch wiring is lane L3). The PLE n-gram table stays IQ4_NL (the wired gather);
    the token embedding stays bf16 for the early path. The kept MTP layer (blk.48) experts take
    w4nl64 by default, because a whole layer left bf16 is refused ("the sort places at most 8").
    """
    lines = ["# Generated by tools/ud_map/ud_map.py from %s" % os.path.basename(gguf),
             "# Flash-Next FAST path: the UD map's formats mapped onto the kernels that exist.",
             "# Routed experts: IQ4_NL/IQ4_XS -> w4nl64a8h, Q5_0 -> w5nl64a8h,",
             "# IQ3_XXS/IQ2_S/Q2_0 -> w3nl64a8h (kernel present; the qwen4exp arch must",
             "# declare it; Q2_0 stays in the rotated family, see _fast_dtype). All are rotated",
             "# codebooks, so one layer keeps one encoding family. This is the early, not",
             "# format-faithful, path. Dense Q6_K/Q8_0 keep their own dense formats; the",
             "# qwen4exp dense wiring is L3, so a conversion stops on those until it lands",
             "# (3.52 GiB). The source is bf16: convert with",
             "# --set quantization_config.quant_method=fp8.",
             ""]
    lines.append("# routed experts: one rule per layer, because the map's mix differs by layer")
    expert = collections.defaultdict(dict)
    for r in rows:
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", r["name"])
        if not m:
            continue
        layer, kind = int(m.group(1)), "down" if m.group(2) == "down" else "gate_up"
        expert[layer].setdefault(kind, []).append(r["type"])
    for layer in sorted(expert):
        e = expert[layer]
        for kind, pattern in (("gate_up", "ffn_gate_up_exps"), ("down", "ffn_down_exps")):
            if kind not in e:
                continue
            ty = max(e[kind], key=lambda x: TYPES.get(x, UNKNOWN_TYPE)[0] or 0)
            rule = _expert_rule(ty, gptq)
            if rule is None:
                lines.append(f"# blk.{layer}.{pattern}: {ty} has no fast format")
                continue
            lines.append(f"blk.{layer}.{pattern}.*.weight".ljust(40) + " " + rule +
                         f"   # {ty} -> {FAST_EXPERTS[ty][0]}")
    if mtp in (None, "") and mtp_layer is not None:
        mtp = "w4nl64"          # a whole bf16 expert layer is refused by the arch
    if mtp in ("w4nl64", "w5nl64") and mtp_layer is not None:
        rule = _expert_rule("Q4_K" if mtp == "w4nl64" else "Q5_K", gptq)
        for pattern in ("ffn_gate_up_exps", "ffn_down_exps"):
            lines.append(f"blk.{mtp_layer}.{pattern}.*.weight".ljust(40) + " " + rule +
                         f"   # MTP layer -> {mtp}")
    lines.append("")
    ple = next((r.get("target") for r in rows
                if r.get("target") and (r.get("role") == "ple-table" or
                                        r["target"].endswith("ple_ngram.weight"))), None)
    if ple:
        lines.append("# the PLE n-gram table: the native IQ4_NL the gather kernel reads")
        lines.append(ple.ljust(40) + " ggml type=iq4_nl   # IQ4_NL (native, PLE gather)")
        lines.append("")
    lines.append("# dense Q6_K/Q8_0: their own dense formats (qwen4exp wiring: L3)")
    lines.append("# Q6_K output.weight has no logits_gemm q6k row; the int8 head is served.")
    lines.append("output.weight".ljust(40) + " " + Q6K_LM_HEAD_RULE + "   # Q6_K -> i8 (L3)")
    lines.append("# Q8_0 connection mixing matrices are read by hc_read (bf16/E4M3), not Q8_0.")
    lines.append("*_hc_down.weight".ljust(40) + " " + HC_E4M3_DOWN_RULE + "   # Q8_0 -> E4M3")
    lines.append("*_hc_up.weight".ljust(40) + " " + HC_E4M3_UP_RULE + "   # Q8_0 -> E4M3")
    written = set()
    for r in rows:
        tgt, ty = r.get("target"), r.get("type")
        if not tgt or "*" in tgt or ty not in ("Q6_K", "Q8_0"):
            continue
        if r.get("role") == "ple-table" or tgt.endswith("ple_ngram.weight"):
            continue
        if tgt == "token_embd.weight" or tgt == "output.weight":
            continue
        if tgt.endswith("_hc_down.weight") or tgt.endswith("_hc_up.weight"):
            continue
        key = (tgt, ty)
        if key in written:
            continue
        written.add(key)
        lines.append(f"{tgt:40s} ggml type={TYPES[ty][1]}   # {ty} -> {ty.lower()} (L3)")
    lines.append("")
    lines.append("# token_embd.weight (IQ4_XS) is kept bf16 for the early path (1.186 GiB);")
    lines.append("# q8_0 would be 0.630 GiB once L3 wires the embedding.")
    return "\n".join(lines) + "\n"


def compute_plan(args):
    kv, ts = load_all(args.gguf)
    arch = kv.get("general.architecture", "?")
    mapper = ARCHES.get(arch)
    if mapper is None:
        sys.exit(f"ud_map: no name map for architecture '{arch}' (have: {', '.join(ARCHES)})")
    policy = getattr(args, "policy", "mirror")
    if policy == "auto":
        policy = "fast" if arch == "qwen35moe" else "mirror"

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
            problems.append(dict(target=target, members=members,
                                 why="fused tensors disagree on type"))

    ws, names = (None, [])
    if getattr(args, "declared", None):
        ws, names = _declared_index(args.declared)
    name_set = set(names)
    for r in rows:
        if policy == "fast":
            fd = _fast_dtype(arch, r)
            if fd is None:
                r["fast_dtype"] = r["fast_bpw"] = r["fast_dep"] = None
                r["fast_servable"] = False
                r["fast_bytes"] = r["bytes"]
                r["servable"] = False
            else:
                r["fast_dtype"], r["fast_bpw"], r["fast_servable"], r["fast_dep"] = fd
                r["fast_bytes"] = _fast_bytes(r, r["fast_dtype"], r["fast_bpw"])
                r["servable"] = r["fast_servable"]
        else:
            ty = r["type"]
            if ty in ("Q6_K", "Q8_0"):
                # The dense forms are served (LinearFP8); output.weight goes to i8 and the
                # connection rows to E4M3 in the recipe. Grouped experts, the token embedding
                # gather and the PLE table have separate kernels and are not.
                tgt = r["target"] or ""
                r["servable"] = not ("_exps" in tgt or "token_embd" in tgt or
                                     "ple_ngram" in tgt)
            else:
                r["servable"] = TYPES.get(ty, UNKNOWN_TYPE)[2]
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
            if policy == "fast":
                r["fast_servable"] = False
                r["fast_dtype"] = None

    # declared weights the map does not cover: the MTP head and the vision tower (the UD text GGUF
    # leaves them out), plus i64 metadata. They are kept from the bf16 source.
    kept, derived = [], []
    if ws is not None:
        targets = [r["target"] for r in rows if r["target"]]
        for w in ws:
            if _covered(w["name"], targets):
                continue
            rec = dict(name=w["name"], encoding=w["encoding"], bytes=w["bytes"],
                       shape=w["shape"])
            if w["name"] == "mtp.draft_head.weight":
                derived.append(rec)          # made from output.weight by the recipe
            else:
                kept.append(rec)
    unmapped = [r["name"] for r in rows if r["kind"] == "unknown"]

    map_bytes = sum(r["bytes"] for r in rows)
    map_bits = sum(r["bytes"] * 8 for r in rows)
    map_elems = sum((r["shape"][0] if r["shape"] else 0) *
                    (r["shape"][1] if len(r["shape"]) > 1 else 1) *
                    (r["shape"][2] if len(r["shape"]) > 2 else 1) for r in rows)
    kept_bytes = sum(k["bytes"] for k in kept)
    derived_bytes = sum(k["bytes"] for k in derived)
    total = map_bytes + kept_bytes + derived_bytes

    mtp_opts = [x.strip() for x in getattr(args, "mtp", "keep").split(",") if x.strip()]
    vis_opts = [x.strip() for x in getattr(args, "vision", "keep").split(",") if x.strip()]
    variations = []
    for m in mtp_opts:
        for v in vis_opts:
            if m not in KEPT_FORMATS or v not in KEPT_FORMATS:
                sys.exit(f"ud_map: unknown kept format (have: {', '.join(KEPT_FORMATS)})")
            other = sum(k["bytes"] for k in kept if _kept_group(k["name"]) == "other")
            mb = sum(_kept_bytes(k, m) for k in kept if _kept_group(k["name"]) == "mtp")
            vb = sum(_kept_bytes(k, v) for k in kept if _kept_group(k["name"]) == "vision")
            variations.append(dict(mtp=m, vision=v,
                                   total_bytes=map_bytes + other + mb + vb + derived_bytes,
                                   mtp_bytes=mb, vision_bytes=vb, other_bytes=other,
                                   derived_bytes=derived_bytes))

    # The MTP layer is the last expert-bearing block. Its sizes are the lever the variations report
    # is about; on Qwen3.6 it is blk.40 (the Unsloth GGUF carries it, the fp8 container serves it).
    mtp_layer = None
    for r in rows:
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", r["name"])
        if m:
            mtp_layer = max(mtp_layer or 0, int(m.group(1)))
    mtp_variations = []
    if mtp_layer is not None:
        lrows = [r for r in rows
                 if r["name"].startswith(f"blk.{mtp_layer}.") and "_exps." in r["name"]]
        cur = sum(r["bytes"] for r in lrows)
        elems = sum((r["shape"][0] if r["shape"] else 0) *
                    (r["shape"][1] if len(r["shape"]) > 1 else 1) *
                    (r["shape"][2] if len(r["shape"]) > 2 else 1) for r in lrows)
        for fmt, bpw in KEPT_FORMATS.items():
            new = cur if bpw is None else int(elems * bpw / 8)
            mtp_variations.append(dict(mtp=fmt, layer=mtp_layer, mtp_bytes=new,
                                       total_bytes=total - cur + new,
                                       delta_bytes=new - cur))

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
    tsum = sorted(((ty, sum(r["bytes"] for r in rows if r["type"] == ty))
                   for ty in {r["type"] for r in rows}), key=lambda x: -x[1])
    ok_bytes = sum(r["bytes"] for r in rows if r["servable"])

    fast = None
    size = None
    mtp_kept_layer = None
    n_layers = 0
    for r in rows:
        m = re.match(r"blk\.(\d+)\.ffn_", r["name"])
        if m:
            n_layers = max(n_layers, int(m.group(1)) + 1)
    for k in kept:
        m = re.match(r"blk\.(\d+)\.", k["name"])
        if m and _kept_group(k["name"]) == "mtp":
            mtp_kept_layer = max(mtp_kept_layer or 0, int(m.group(1)))
        m2 = re.match(r"mtp\.layers\.(\d+)\.", k["name"])
        if m2 and n_layers:
            mtp_kept_layer = max(mtp_kept_layer or 0, n_layers + int(m2.group(1)))
    if policy == "fast":
        fast_map = sum(r["fast_bytes"] for r in rows)
        fast_total = fast_map + kept_bytes + derived_bytes
        fast_ok = sum(r["fast_bytes"] for r in rows if r["fast_servable"])
        mtp_variant = getattr(args, "mtp_variant", None)
        if arch == "qwen4exp" and mtp_variant in (None, ""):
            # The fast recipe always writes the kept MTP layer (a whole bf16 expert layer is
            # refused by the arch); w4nl64 is the measured production choice.
            mtp_variant = "w4nl64"
        mtp_delta = 0
        if mtp_variant in ("w4nl64", "w5nl64") and mtp_kept_layer is not None:
            for k in kept:
                if (k["name"].startswith(f"blk.{mtp_kept_layer}.") and
                        re.search(r"ffn_(?:gate_up|gate|up|down)_exps\.", k["name"])):
                    mtp_delta += _kept_bytes(k, mtp_variant) - k["bytes"]
            fast_ok += mtp_delta
        # Kept/derived weights are served as bf16 (or the chosen MTP form), so they count here.
        fast_ok += kept_bytes + derived_bytes
        ok_bytes = fast_ok
        fb = collections.OrderedDict()
        for r in rows:
            k = (r["fast_dtype"], r["type"], r["fast_dep"] or "", r["fast_servable"])
            a = fb.setdefault(k, [0, 0, 0])
            a[0] += 1
            a[1] += r["bytes"]
            a[2] += r["fast_bytes"]
        fast_by = [dict(dtype=d, ud_type=ty, dep=dep, servable=ok, n=n, ud_bytes=b,
                        bytes=fb_) for (d, ty, dep, ok), (n, b, fb_) in
                   sorted(fb.items(), key=lambda kv: -kv[1][2])]
        fast_cov = {}
        for r in rows:
            if r["target"]:
                fast_cov.setdefault(r["target"], r["fast_dtype"])
        fast_missing = [t for t, d in fast_cov.items() if d is None]
        fast_largest = [dict(name=r["name"], target=r["target"], type=r["type"],
                             dtype=r["fast_dtype"], bytes=r["fast_bytes"],
                             dep=r["fast_dep"]) for r in
                        sorted(rows, key=lambda r: -r["fast_bytes"])[:15]]
        fast = dict(map_bytes=fast_map, total_bytes=fast_total,
                    build_bytes=fast_total + mtp_delta, kept_bytes=kept_bytes,
                    derived_bytes=derived_bytes, servable_bytes=fast_ok,
                    mtp_variant=mtp_variant, mtp_variant_delta=mtp_delta,
                    by=fast_by, largest=fast_largest,
                    coverage=dict(targets=len(fast_cov),
                                  covered=len(fast_cov) - len(fast_missing),
                                  missing=fast_missing))
        recipe = _fast_recipe(rows, args.gguf[-1], gptq=getattr(args, "gptq", False),
                              mtp=mtp_variant, arch=arch,
                              mtp_layer=mtp_kept_layer)
    else:
        mirror_variant = getattr(args, "mtp_variant", None)
        if arch == "qwen4exp" and mirror_variant in (None, ""):
            mirror_variant = "w4nl64"
        recipe = _mirror_recipe(rows, args.gguf[-1], mtp=mirror_variant,
                                mtp_layer=mtp_kept_layer)
        if policy == "size":
            choices = _expert_choices(rows)
            size_map = _size_map_bytes(rows)
            size = dict(map_bytes=size_map, saved_bytes=map_bytes - size_map,
                        by_layer=[dict(layer=l, native=c["native"], fast=c["fast"],
                                       use_fast=c["use_fast"])
                                  for l, c in sorted(choices.items())])
            map_bytes = size_map
            total = map_bytes + kept_bytes + derived_bytes
            ok_bytes = total
            recipe = _size_recipe_4exp(rows, args.gguf[-1], mtp=mirror_variant,
                                       mtp_layer=mtp_kept_layer)
    return dict(tool="ud_map 0.2", arch=arch, policy=policy, map=list(args.gguf),
                declared_plan=getattr(args, "declared", None),
                source=getattr(args, "source", None),
                totals=dict(map_bytes=map_bytes, kept_bytes=kept_bytes,
                            derived_bytes=derived_bytes, total_bytes=total,
                            map_bpw=(map_bits / map_elems if map_elems else 0),
                            servable_bytes=ok_bytes),
                variations=variations, mtp_variations=mtp_variations,
                mtp_layer=mtp_layer, mtp_kept_layer=mtp_kept_layer,
                tensors=rows, kept=kept, derived=derived,
                unmapped=unmapped, fast=fast, size=size,
                fusion_problems=problems, coverage=dict(declared=len(cov),
                                                        missing=missing),
                recipe=recipe), by, missing, tsum


def plan(args):
    data, by, missing, tsum = compute_plan(args)
    arch = data["arch"]
    total = data["totals"]["total_bytes"]
    map_bytes = data["totals"]["map_bytes"]
    kept_bytes = data["totals"]["kept_bytes"]
    rows, kept = data["tensors"], data["kept"]
    label = "chosen formats" if data["policy"] == "size" else "Unsloth's formats"
    print(f"{arch}: {len(rows)} mapped tensors, {map_bytes/2**30:.3f} GiB in {label}, "
          f"{kept_bytes/2**30:.3f} GiB kept from the bf16 source (not in the map), "
          f"total {total/2**30:.3f} GiB")
    if data.get("size"):
        print(f'size policy: saved {data["size"]["saved_bytes"]/2**30:.3f} GiB by choosing the '
              f'smaller per-layer family')
    print(f'{"role":24s}{"ggml":9s}{"target":30s}{"servable":9s}{"n":>4s}{"GiB":>9s}')
    for (role, ty, tgt, ok), (n, b) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f'{(role or "?"):24s}{ty:9s}{(tgt or "-"):30s}{"yes" if ok else "NO":9s}'
              f'{n:4d}{b/2**30:9.3f}')
    cov = data["coverage"]
    print(f'\nmap coverage: {cov["declared"]-len(missing)}/{cov["declared"]} distinct declared targets present')
    if missing:
        groups = collections.Counter(re.sub(r'^blk\.\d+\.', 'blk.*.', t) for t in missing)
        print('  no declared weight for: ' + ', '.join(f'{k} x{n}' for k, n in groups.most_common(6)))
    ok_bytes = data["totals"]["servable_bytes"]
    fast = data.get("fast")
    servable_base = fast["build_bytes"] if fast else total
    print(f'servable by the current build: {ok_bytes/2**30:.3f} GiB '
          f'({100*ok_bytes/servable_base:.1f}% of the {"fast build" if fast else "total"}); '
          f'blocked on kernels: {(servable_base-ok_bytes)/2**30:.3f} GiB')
    print('  by type (GiB): ' + ', '.join(f'{ty} {b/2**30:.2f}' for ty, b in tsum))
    if fast:
        print('\nfast mapping (UD type -> served kernel):')
        print(f'  {"fast dtype":12s}{"UD type":10s}{"servable":9s}{"depends":38s}'
              f'{"n":>4s}{"UD GiB":>9s}{"fast GiB":>10s}')
        for x in fast["by"]:
            print(f'  {(x["dtype"] or "-"):12s}{(x["ud_type"] or "-"):10s}'
                  f'{"yes" if x["servable"] else "NO":9s}{(x["dep"] or "-"):38s}'
                  f'{x["n"]:4d}{x["ud_bytes"]/2**30:9.3f}{x["bytes"]/2**30:10.3f}')
        delta = (fast["total_bytes"] - total) / 2**30
        print(f'fast total (MTP kept bf16, like the UD plan): {fast["total_bytes"]/2**30:.3f} GiB '
              f'(map {fast["map_bytes"]/2**30:.3f} + kept {fast["kept_bytes"]/2**30:.3f} '
              f'+ derived {fast["derived_bytes"]/2**30:.3f}); '
              f'delta vs the UD plan {total/2**30:.3f} GiB: {delta:+.3f} GiB')
        if fast.get("mtp_variant") in ("w4nl64", "w5nl64"):
            print(f'  MTP layer {data.get("mtp_kept_layer")} at {fast["mtp_variant"]}: '
                  f'{fast["mtp_variant_delta"]/2**30:+.3f} GiB vs kept; '
                  f'what the fast recipe writes: {fast["build_bytes"]/2**30:.3f} GiB')
        cov_f = fast["coverage"]
        print(f'fast coverage: {cov_f["covered"]}/{cov_f["targets"]} distinct declared targets '
              f'have a selected fast dtype')
        if cov_f["missing"]:
            groups = collections.Counter(re.sub(r'^blk\.\d+\.', 'blk.*.', t)
                                         for t in cov_f["missing"])
            print('  no fast dtype for: ' + ', '.join(f'{k} x{n}' for k, n in
                                                      groups.most_common(6)))
        print(f'fast servable by the current build: {fast["servable_bytes"]/2**30:.3f} GiB '
              f'({100*fast["servable_bytes"]/fast["build_bytes"]:.1f}% of the fast build); '
              f'blocked: {(fast["build_bytes"]-fast["servable_bytes"])/2**30:.3f} GiB '
              f'(w3nl64 arch + L3 dense)')
        print('\nlargest targets by fast size:')
        print(f'  {"target":40s}{"UD":9s}{"fast":10s}{"GiB":>9s}  depends')
        for x in fast["largest"]:
            print(f'  {(x["target"] or x["name"]):40s}{x["type"]:9s}{(x["dtype"] or "-"):10s}'
                  f'{x["bytes"]/2**30:9.3f}  {x["dep"] or "-"}')
    variations = data["variations"]
    if variations:
        base = next((x for x in variations if x["mtp"] == "keep" and x["vision"] == "keep"), None)
        _mn = data.get("mtp_kept_layer") or data.get("mtp_layer")
        print(f'\nkept-weight variations (MTP layer blk.{_mn} + mtp.*, vision v.*): total container')
        print(f'  {"MTP":8s}{"vision":8s}{"total GiB":>11s}{"delta GiB":>11s}')
        for x in variations:
            delta = (x["total_bytes"] - base["total_bytes"]) / 2**30 if base else 0.0
            print(f'  {x["mtp"]:8s}{x["vision"]:8s}{x["total_bytes"]/2**30:11.3f}{delta:11.3f}')
    mtp_variations = data.get("mtp_variations") or []
    if mtp_variations:
        print(f'\nlast expert layer {data.get("mtp_layer")} experts (map size; the draft-only lever)')
        print(f'  {"format":8s}{"layer GiB":>11s}{"total GiB":>11s}{"delta GiB":>11s}')
        for x in mtp_variations:
            print(f'  {x["mtp"]:8s}{x["mtp_bytes"]/2**30:11.3f}'
                  f'{x["total_bytes"]/2**30:11.3f}{x["delta_bytes"]/2**30:11.3f}')
    if data.get("derived"):
        db = sum(k["bytes"] for k in data["derived"])
        print(f'\nderived from the map (not read from bf16): {db/2**30:.3f} GiB in '
              f'{len(data["derived"])} weight(s)')
        for k in data["derived"]:
            print(f'  {k["name"]:40s} {k["encoding"]:24s} {k["bytes"]/2**30:7.3f} GiB')
    if kept:
        print(f'\nkept from bf16 (no map entry): {kept_bytes/2**30:.3f} GiB in {len(kept)} weights')
        def pre(name):
            m = re.match(r'(blk\.\d+|v|mtp|[a-z_]+)', name)
            return m.group(1) if m else name
        kb = collections.Counter((k["encoding"], pre(k["name"])) for k in kept)
        for (enc, pre_), n in kb.most_common(10):
            gb = sum(k["bytes"] for k in kept if pre(k["name"]) == pre_) / 2**30
            print(f'  {pre_:12s} {enc:10s} {n:4d} weight(s) {gb:7.3f} GiB')
    if data["fusion_problems"]:
        print("\nFUSION PROBLEMS (the container cannot mirror both halves):")
        for p in data["fusion_problems"]:
            print(" ", p)
    if data["unmapped"]:
        print(f"\nunmapped GGUF tensors ({len(data['unmapped'])}): {data['unmapped'][:12]}")

    if getattr(args, "recipe_out", None):
        open(args.recipe_out, "w").write(data["recipe"])
    if getattr(args, "json", None):
        with open(args.json, "w") as f:
            json.dump(data, f, indent=1)
    return 0



# ==================================================================== shared pipeline helpers

def _env_pairs(pairs):
    env = dict(os.environ)
    for kv in pairs or []:
        k, eq, v = kv.partition("=")
        if not eq or not k:
            sys.exit(f"ud_map: --env wants KEY=VALUE, got '{kv}'")
        env[k] = v
    return env


def _child_env(args, base=None):
    env = dict(base) if base is not None else _env_pairs(getattr(args, "env", []))
    if getattr(args, "home", None):
        env["RADIANCE_HOME"] = args.home
    if getattr(args, "kernels", None):
        env.setdefault("RADIANCE_KERNELS", args.kernels)
    return env


def _tool(args, name, attr, env_word):
    v = getattr(args, attr, None)
    if v:
        return v
    v = os.environ.get(env_word)
    if v:
        return v
    found = shutil.which(name)
    if not found:
        sys.exit(f"ud_map: cannot find {name}; pass --{attr.replace('_', '-')} or set {env_word}")
    return found


def _run(cmd, log, env=None, timeout=None, dry=False):
    """Run a command, tee its output to `log`, and return (rc, text). A dry run only prints."""
    text = shlex.join(cmd)
    if dry:
        print("+ " + text)
        return 0, ""
    print("+ " + text, flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        f.write("$ " + text + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                           timeout=timeout, text=True, errors="replace")
        f.write(p.stdout or "")
    dt = time.time() - t0
    print(f"  exit {p.returncode} in {dt:.1f}s; log {log}", flush=True)
    if p.returncode != 0:
        tail = "\n".join((p.stdout or "").splitlines()[-25:])
        print(tail, file=sys.stderr)
    return p.returncode, p.stdout or ""


def source_quant(path):
    """'bf16', 'gguf' or 'quantised' from a checkpoint directory/GGUF, without loading weights."""
    if path.endswith(".gguf"):
        return "gguf"
    if os.path.isfile(path) and path.endswith(".rad"):
        return "rad"
    if not os.path.isdir(path):
        return "unknown"
    cfg = os.path.join(path, "config.json")
    if os.path.exists(cfg):
        try:
            d = json.load(open(cfg))
        except Exception:
            d = {}
        qc = d.get("quantization_config") or {}
        if qc:
            return qc.get("quant_method") or qc.get("fmt") or "quantised"
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        try:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hd = json.loads(fh.read(n))
        except Exception:
            continue
        dts = {v.get("dtype") for k, v in hd.items() if k != "__metadata__"}
        if dts & {"BF16", "F16", "F32"}:
            return "bf16"
    return "unknown"


def _plan_namespace(args, declared, policy, recipe_out=None, json_out=None):
    return argparse.Namespace(
        gguf=args.gguf, declared=declared, source=getattr(args, "source", None),
        recipe_out=recipe_out, json=json_out,
        mtp=getattr(args, "mtp", "keep"), vision=getattr(args, "vision", "keep"),
        policy=policy, gptq=getattr(args, "gptq", False),
        mtp_variant=getattr(args, "mtp_variant", None))


def _convert_command(args, recipe, out, declared_json=None, plan_only=False):
    cmd = [_tool(args, "rad-convert", "rad_convert", "RADIANCE_RAD_CONVERT"), args.source]
    cmd += ["--recipe", recipe, "-o", out]
    if plan_only:
        cmd.append("--plan-only")
    if declared_json:
        cmd += ["--plan-json", declared_json]
    if getattr(args, "imatrix", None):
        cmd += ["--imatrix", args.imatrix]
    if getattr(args, "reuse", None):
        cmd += ["--reuse", args.reuse]
    for s in getattr(args, "set", []) or []:
        cmd += ["--set", s]
    if getattr(args, "auto_fp8", True):
        if getattr(args, "_source_quant", None) == "bf16" and not any(
                s.startswith("quantization_config.") for s in (getattr(args, "set", []) or [])):
            cmd += ["--set", "quantization_config.quant_method=" + BF16_TO_FAST_QUANT]
    if getattr(args, "home", None):
        cmd += ["--home", args.home]
    if getattr(args, "kernels", None):
        cmd += ["--kernels", args.kernels]
    cmd += list(getattr(args, "extra_arg", []) or [])
    return cmd


def _choose_policy(args, arch):
    p = getattr(args, "policy", "auto")
    if p in ("auto", None):
        return "fast" if arch == "qwen35moe" else "mirror"
    return p


def build(args):
    """plan -> recipe -> rad-convert, with a dry run and a machine-readable plan."""
    q = getattr(args, "source_quant", "auto")
    sq = source_quant(args.source) if q == "auto" else q
    args._source_quant = sq
    gguf = args.gguf if isinstance(args.gguf, list) else [args.gguf]
    arch = load_all(gguf)[0].get("general.architecture", "?")
    policy = _choose_policy(args, arch)
    out = args.out
    recipe_path = args.recipe_out or (out + ".recipe")
    plan_json = args.json or (out + ".plan.json")
    declared_json = args.declared or (out + ".declared.json")
    log_path = args.log or (out + ".convert.log")

    ns = _plan_namespace(args, args.declared, policy, recipe_out=recipe_path)
    data, by, missing, tsum = compute_plan(ns)
    data["source_quant"] = sq
    data["recipe_path"] = recipe_path
    data["plan_json"] = plan_json
    data["declared_plan"] = args.declared
    data["recipe_policy"] = policy

    if getattr(args, "dry_run", False):
        # Emit the recipe and the plan, print the exact command, run nothing.
        cmd = _convert_command(args, recipe_path, out, declared_json, args.plan_only)
        data["command"] = cmd
        open(recipe_path, "w").write(data["recipe"])
        with open(plan_json, "w") as f:
            json.dump(data, f, indent=1)
        print("dry run: recipe " + recipe_path + ", plan " + plan_json)
        print("+ " + shlex.join(cmd))
        return 0

    # The declared plan lets the plan check coverage and gives verify the source names. Generate it
    # first, from the recipe we are about to use; this also proves the recipe is accepted.
    open(recipe_path, "w").write(data["recipe"])
    cmd0 = _convert_command(args, recipe_path, out, declared_json, plan_only=True)
    rc, text = _run(cmd0, log_path + ".plan", env=_env_pairs(args.env),
                    timeout=args.timeout, dry=False)
    if rc != 0:
        print("ud_map: the plan-only conversion refused the recipe; see " + log_path + ".plan",
              file=sys.stderr)
        return rc
    if args.declared:
        declared_json = args.declared
    ns = _plan_namespace(args, declared_json, policy, recipe_out=recipe_path)
    data, by, missing, tsum = compute_plan(ns)
    data["source_quant"] = sq
    data["recipe_path"] = recipe_path
    data["plan_json"] = plan_json
    data["declared_plan"] = declared_json
    data["recipe_policy"] = policy
    open(recipe_path, "w").write(data["recipe"])
    with open(plan_json, "w") as f:
        json.dump(data, f, indent=1)
    if args.plan_only:
        print("plan-only: " + plan_json + "; declared " + declared_json)
        return 0

    cmd = _convert_command(args, recipe_path, out, None, plan_only=False)
    data["command"] = cmd
    rc, text = _run(cmd, log_path, env=_env_pairs(args.env), timeout=args.timeout)
    data["convert_rc"] = rc
    data["convert_log"] = log_path
    if os.path.exists(out):
        data["container_bytes"] = os.path.getsize(out)
        h = hashlib.sha256()
        with open(out, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        data["container_sha256"] = h.hexdigest()
    report = args.report_out or (out + ".build.json")
    with open(report, "w") as f:
        json.dump(data, f, indent=1)
    print(f"build: {out} ({data.get('container_bytes', 0)/2**30:.3f} GiB), report {report}")
    if rc == 0 and getattr(args, "verify", False):
        vargs = argparse.Namespace(gguf=gguf, rad=out, source=args.source,
                                   declared=declared_json, plan_json=plan_json,
                                   out=args.report_out or (out + ".verify.json"),
                                   only=None, workers=getattr(args, "workers", 8),
                                   tol=getattr(args, "tol", 0.20),
                                   draft_tol=getattr(args, "draft_tol", 0.75),
                                   skip_gguf=getattr(args, "skip_gguf", False),
                                   max_experts=getattr(args, "max_experts", 0),
                                   policy="auto")
        return rc if verify(vargs) else 1
    return rc


def _with_flock(args, cmd):
    lock = getattr(args, "flock", None)
    if getattr(args, "no_flock", False):
        return cmd
    if not lock:
        default = os.path.expanduser("~/radiance-test/gpu.lock")
        if os.path.exists(default):
            lock = default
    if not lock:
        return cmd
    flock = shutil.which("flock")
    if not flock:
        sys.exit("ud_map: flock not found; pass --no-flock only for a non-GPU run")
    out = [flock]
    w = getattr(args, "flock_wait", None)
    if w:
        out += ["-w", str(w)]
    return out + [lock] + cmd


_CALIB_RE = re.compile(r"calibration wrote .*after (\d+) tokens")


def calibrate(args):
    """Drive a calibration batch run (the ENV tap, not just --calib-dir), then a GPTQ convert
    with --reuse the RTN container."""
    if not args.convert_only and not args.corpus:
        sys.exit("ud_map calibrate: give --corpus (or --convert-only)")
    os.makedirs(args.calib_dir, exist_ok=True)
    args._source_quant = source_quant(args.source) if getattr(args, "source", None) else None
    report = dict(tool="ud_map calibrate", calib_dir=args.calib_dir, tokens=args.tokens,
                  corpus=args.corpus, model=args.model, phases=[])
    env = _child_env(args)
    env["RADIANCE_CALIB_DIR"] = args.calib_dir
    env["RADIANCE_CALIB_TOKENS"] = str(args.tokens)
    if not args.convert_only:
        radiance = _tool(args, "radiance", "radiance", "RADIANCE_RADIANCE")
        cmd = [radiance, "--model", args.model, "--calib-corpus", args.corpus,
               "--calib-dir", args.calib_dir,
               "--max-model-len", str(args.max_model_len), "--tp", str(args.tp),
               "--kv-cache-dtype", args.kv_cache_dtype,
               "--max-num-batched-tokens", str(args.max_num_batched_tokens),
               "--max-num-seqs", str(args.max_num_seqs)]
        if args.num_speculative_tokens is not None:
            cmd += ["--num-speculative-tokens", str(args.num_speculative_tokens)]
        if args.kernels:
            cmd += ["--kernels", args.kernels]
        cmd += list(args.extra_arg or [])
        cmd = _with_flock(args, cmd)
        log = args.log or os.path.join(args.calib_dir, "calibrate.log")
        rc, text = _run(cmd, log, env=env, timeout=args.timeout)
        found = _CALIB_RE.findall(text or "")
        nfiles = len(glob.glob(os.path.join(args.calib_dir, "gram.*.bin")))
        report["phases"].append(dict(name="calibrate", command=cmd, rc=rc, log=log,
                                     tokens_reported=int(found[-1]) if found else None,
                                     gram_files=nfiles))
        if rc != 0:
            with open(args.report_out or (args.calib_dir + "/calibrate.report.json"), "w") as f:
                json.dump(report, f, indent=1)
            return rc
    if not args.calibrate_only:
        if not args.source or not args.out:
            sys.exit("ud_map calibrate: the GPTQ phase needs --source and --out")
        if args.recipe:
            recipe = args.recipe
        else:
            if not args.gguf:
                sys.exit("ud_map calibrate: pass --gguf (the UD map) or --recipe for the GPTQ recipe")
            gguf = args.gguf if isinstance(args.gguf, list) else [args.gguf]
            arch = load_all(gguf)[0].get("general.architecture", "?")
            policy = _choose_policy(args, arch)
            args.gptq = True
            ns = _plan_namespace(args, None, policy)
            data, _, _, _ = compute_plan(ns)
            recipe = args.recipe_out or (args.out + ".gptq.recipe")
            open(recipe, "w").write(data["recipe"])
        # The hand-built recipe reads $CALIB; the env carries it as well as the recipe's directory.
        if args.imatrix is None and getattr(args, "auto_imatrix", None):
            args.imatrix = args.auto_imatrix
        cmd = _convert_command(args, recipe, args.out, None, plan_only=False)
        if args.reuse is None:
            cmd += ["--reuse", args.model]
        env2 = _child_env(args, env)
        env2["CALIB"] = args.calib_dir
        log = args.log or (args.out + ".convert.log")
        rc, text = _run(cmd, log, env=env2, timeout=args.timeout)
        phase = dict(name="convert", command=cmd, rc=rc, log=log, recipe=recipe)
        if os.path.exists(args.out):
            phase["container_bytes"] = os.path.getsize(args.out)
            h = hashlib.sha256()
            with open(args.out, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            phase["container_sha256"] = h.hexdigest()
        report["phases"].append(phase)
        if rc != 0:
            with open(args.report_out or (args.out + ".calibrate.json"), "w") as f:
                json.dump(report, f, indent=1)
            return rc
    out = args.report_out or ((args.out + ".calibrate.json") if args.out else
                              os.path.join(args.calib_dir, "calibrate.report.json"))
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("calibrate: report " + out)
    return 0


# ==================================================================== verify

def _e4m3_table():
    t = [0.0] * 256
    for i in range(256):
        s = -1.0 if i & 0x80 else 1.0
        e, m = (i >> 3) & 15, i & 7
        t[i] = float("nan") if (e == 15 and m == 7) else s * (
            (m / 8 * 2 ** -6) if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return t


E4M3 = None


def _plane_float(rf, e, role, rows, cols, row0=0, nrows=None):
    """A named plane as f32 in logical (rows, cols), decoding E4M3 and expanding its block.
    With row0/nrows only those logical rows are expanded, so a 500M-element draft head never
    becomes three 4 GB arrays."""
    import numpy as np
    from radfile import plane_values
    global E4M3
    roles = [p.role for p in e.enc.planes]
    if role not in roles:
        return None
    k = roles.index(role)
    p = e.enc.planes[k]
    if nrows is None:
        nrows = rows - row0
    if p.kind == 1:                         # table: every slice carries the whole table
        v = plane_values(rf, e, k)
        return np.asarray(v, np.float32).reshape(-1) if p.dtype != 4 else \
            (E4M3 if E4M3 is not None else np.array(_e4m3_table(), np.float32))[
                np.asarray(v).astype(np.int64)].reshape(-1)
    br = p.block[0] if p.block[0] > 0 else rows
    bc = p.block[1] if p.block[1] > 0 else cols
    pr, pc = p.dims(rows, cols)
    if br >= rows and pc == 1 and p.block[0] == 0:
        # A whole-tensor level (scale2's fixed value): one plane element covers every row.
        v = plane_values(rf, e, k, 0, 1)
        if p.dtype == 4:
            if E4M3 is None:
                E4M3 = np.array(_e4m3_table(), np.float32)
            v = E4M3[np.asarray(v).astype(np.int64)]
        v = np.asarray(v, np.float32).reshape(1, -1)
        row = np.broadcast_to(v[0], (nrows, cols)) if v.shape[1] == 1 else \
            np.broadcast_to(np.repeat(v, bc, axis=1)[0], (nrows, cols))
        return np.ascontiguousarray(row)
    p0 = row0 // br if br < rows else 0
    p1 = (row0 + nrows - 1) // br + 1 if br < rows else 1
    v = plane_values(rf, e, k, p0, p1 - p0)
    if p.dtype == 4:
        if E4M3 is None:
            E4M3 = np.array(_e4m3_table(), np.float32)
        v = E4M3[np.asarray(v).astype(np.int64)]
    v = np.asarray(v, np.float32).reshape(p1 - p0, pc)
    v = np.repeat(v, bc, axis=1)[:, :cols]
    if br == 1:
        return np.ascontiguousarray(v[:nrows])
    offset = row0 - p0 * br
    need = offset + nrows
    v = np.repeat(v, br, axis=0)[offset:need]
    return np.ascontiguousarray(v)


def decode_entry(rf, e, row0=0, nrows=None):
    """Decode one .rad entry (a chunk of rows) to f32. Handles plain, affine with a scalar table,
    block FP8 and the fwht128 transform the w4nl/w5nl experts use.

    A plugin-private plane (abi/rad_types.h >= RAD_DT_PLUGIN_BASE) is not guessed at: raises
    radfile.KernelPrivateDtype so verify() can report it as kernel-private rather than FAIL or
    silently mis-decode it."""
    import numpy as np
    from radfile import plane_values, KernelPrivateDtype
    enc = e.enc
    if enc.kernel_private_planes():
        raise KernelPrivateDtype([p.dtype for p in enc.kernel_private_planes()])
    rows = int(np.prod(e.shape[:-1])) if len(e.shape) > 1 else 1
    cols = e.shape[-1] if e.shape else 1
    if nrows is None:
        nrows = rows - row0
    if enc.scheme == "plain":
        return plane_values(rf, e, 0, row0, nrows).astype(np.float32)
    if enc.scheme != "affine":
        raise NotImplementedError(enc.name())
    roles = [p.role for p in enc.planes]
    table = _plane_float(rf, e, "table", rows, cols) if "table" in roles else None
    q = _plane_float(rf, e, "codes", rows, cols, row0, nrows)
    if q is None:
        raise NotImplementedError(enc.name())
    if "grid" in roles:
        # A vector codebook: one u-code a V-element grid entry, then a sign bit a value
        # (abi/rad_encoding.h: q = grid[code][k], negated when signs says so). The IQ3_XXS and
        # IQ2_S experts use this; a scalar table does not.
        gi = roles.index("grid")
        gp = enc.planes[gi]
        V = gp.extent[1]
        grid = np.asarray(plane_values(rf, e, gi)).reshape(gp.extent)
        code = np.ascontiguousarray(q[:, ::V])
        if code.size and (int(code.min()) < 0 or int(code.max()) >= grid.shape[0]):
            raise ValueError(f"{enc.name()}: a code is outside a {grid.shape[0]}-entry grid")
        q = grid[code.astype(np.int64)].reshape(nrows, cols)
        table = None
    elif table is not None:
        q = table[q.astype(np.int64)]
    if "signs" in roles:
        sg = _plane_float(rf, e, "signs", rows, cols, row0, nrows)
        q = np.where(sg != 0, -q, q)
    if "zero" in roles:
        q = q - _plane_float(rf, e, "zero", rows, cols, row0, nrows)
    lv = 0
    while True:
        r = "scale" if lv == 0 else f"scale.{lv}"
        if r not in roles:
            break
        q = q * _plane_float(rf, e, r, rows, cols, row0, nrows)
        lv += 1
    lv = 0
    sub = None
    while True:
        r = "min" if lv == 0 else f"min.{lv}"
        if r not in roles:
            break
        m = _plane_float(rf, e, r, rows, cols, row0, nrows)
        sub = m if sub is None else sub * m
        lv += 1
    if sub is not None:
        q = q - sub
    if enc.transform == "fwht128":
        # The container stores the orthonormal fwht128 of the weight (fit_experiments' form); the
        # core's inverse is the *unnormalised* Hadamard, i.e. fwht128(q) * sqrt(128).
        from fit_experiments import fwht128
        q = fwht128(q) * np.sqrt(128.0)
    elif enc.transform:
        raise NotImplementedError(enc.name())
    return q.astype(np.float32)


def _sha(a):
    import numpy as np
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a, dtype="<f4").tobytes())
    return h.hexdigest()


def _rel(a, b):
    import numpy as np
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    den = float((a * a).sum())
    if den == 0:
        return 0.0 if float(((a - b) ** 2).sum()) == 0 else float("inf")
    return float(np.sqrt(((a - b) ** 2).sum() / den))


class Refs:
    """The source and GGUF references for one .rad weight name."""

    def __init__(self, rf, src, gg, L):
        self.rf, self.src, self.gg, self.L = rf, src, gg, L
        # The draft/MTP layer is the last expert-bearing block (blk.40 on Qwen3.6, blk.48 on
        # Flash-Next). Its source tensors live under mtp.layers.0.*, not model.language_model.*.
        self.mtp_layer = None
        for e in rf.entries if rf is not None else []:
            m = re.match(r"blk\.(\d+)\.ffn_(?:gate_up|gate|up|down)_exps\.", e.name)
            if m:
                self.mtp_layer = max(self.mtp_layer or 0, int(m.group(1)))

    def _src(self, name, expert=None, row_limit=None):
        if self.src is None:
            return None
        try:
            a = self.src.get(name, slice(0, row_limit) if row_limit else None)
        except KeyError:
            return None
        if expert is not None:
            a = a[expert]
        return a

    def _gg(self, name, expert=None, row_limit=None):
        if self.gg is None or name not in self.gg.T:
            return None
        t = self.gg.T[name]
        rows, cols = self.gg.logical(t)
        E = list(reversed(t["shape"]))[0] if expert is not None else 1
        if expert is not None:
            rows //= E
        if row_limit:
            rows = min(rows, row_limit)
        return self.gg.rows_f32(name, 0, rows, expert)

    def _ssm(self, key):
        """An SSM hyperparameter, spelled under the GGUF's own architecture prefix (qwen4exp on
        Flash-Next, qwen35moe on Qwen3.6)."""
        arch = (self.gg.kv.get("general.architecture") if self.gg else None)
        for pre in ([arch] if arch else []) + ["qwen4exp", "qwen35moe"]:
            k = f"{pre}.ssm.{key}"
            if k in self.gg.kv:
                return self.gg.kv[k]
        raise KeyError(f"{key}: no <arch>.ssm.{key} in the GGUF")

    @staticmethod
    def _untile(x, axis, start, span, a, inner):
        import numpy as np
        b = span // (a * inner)
        idx = np.arange(span).reshape(a, b, inner).transpose(1, 0, 2).reshape(-1)
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(start, start + span)
        y = x.copy()
        y[tuple(sl)] = x[tuple(sl)].take(idx, axis=axis)
        return y

    def _mtp_source(self, name, row_limit=None):
        """The draft head's source names: the MTP layer and head live under mtp.* in the checkpoint."""
        import numpy as np
        hc = re.fullmatch(r"mtp\.(attn|ffn|output)_hc_(down|up|inject|norm)\.weight", name)
        if hc:
            part = {"down": "input_mix_weight_down.weight",
                    "up": "input_mix_weight_up.weight",
                    "inject": "block_inject_weight.weight",
                    "norm": "hc_norm.weight"}[hc.group(2)]
            grp = {"attn": "attn_hyper_connection", "ffn": "mlp_hyper_connection",
                   "output": "hyper_connection_mixer"}[hc.group(1)]
            pre = "mtp" if hc.group(1) == "output" else "mtp.layers.0"
            return self._src(f"{pre}.{grp}.{part}")
        direct = {"mtp.fc_embedding.weight", "mtp.fc_hidden.weight",
                  "mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_embedding.weight"}
        if name in direct:
            return self._src(name, row_limit=row_limit)
        return None

    def _ple_ngram(self, pre, row_limit=None):
        """The PLE n-gram table is 128 checkpoint shards, concatenated in shard order."""
        import numpy as np
        if self.src is None:
            return None
        stem = pre + "ple.ple_embedding.ngram_embedding.shard_"
        names = [n for n in getattr(self.src, "idx", {}) if n.startswith(stem)]
        if not names:
            return None
        names.sort(key=lambda n: int(re.fullmatch(re.escape(stem) + r"(\d+)\.weight", n).group(1)))
        parts, got = [], 0
        for n in names:
            rows = self.src.idx[n][4][0]
            take = rows if row_limit is None else max(0, min(rows, row_limit - got))
            if take:
                parts.append(self._src(n, row_limit=take))
                got += take
            if row_limit is not None and got >= row_limit:
                break
        return np.concatenate(parts, axis=0) if parts else None

    def source(self, name, row_limit=None):
        """(source array or None, note)."""
        import numpy as np
        if name.startswith("v."):
            m = re.fullmatch(r"v\.blk\.(\d+)\.(qkv|proj|fc1|fc2|norm1|norm2)\.(weight|bias)", name)
            if m:
                blk, part, kind = m.group(1), m.group(2), m.group(3)
                p = {"qkv": f"attn.qkv.{kind}", "proj": f"attn.proj.{kind}",
                     "fc1": f"mlp.linear_fc1.{kind}", "fc2": f"mlp.linear_fc2.{kind}",
                     "norm1": f"norm1.{kind}", "norm2": f"norm2.{kind}"}[part]
                return self._src(f"model.visual.blocks.{blk}.{p}", row_limit=row_limit), ""
            m = re.fullmatch(r"v\.(patch|pos|merger\.fc1|merger\.fc2|merger\.norm)\.(weight|bias)", name)
            if m:
                p = {"patch": "patch_embed.proj", "pos": "pos_embed",
                     "merger.fc1": "merger.linear_fc1", "merger.fc2": "merger.linear_fc2",
                     "merger.norm": "merger.norm"}[m.group(1)]
                return self._src(f"model.visual.{p}.{m.group(2)}", row_limit=row_limit), ""
            return None, "no source rule"
        if name == "token_embd.weight":
            return self._src("model.language_model.embed_tokens.weight", row_limit=row_limit), ""
        if name == "output.weight":
            return self._src("lm_head.weight", row_limit=row_limit), ""
        if name == "output_norm.weight":
            return self._src("model.language_model.norm.weight"), ""
        if name in ("output_hc_down.weight", "output_hc_up.weight", "output_hc_norm.weight"):
            part = {"output_hc_down.weight": "input_mix_weight_down.weight",
                    "output_hc_up.weight": "input_mix_weight_up.weight",
                    "output_hc_norm.weight": "hc_norm.weight"}[name]
            return self._src("model.language_model.hyper_connection_mixer." + part), ""
        if name == "mtp.draft_head.weight":
            return self._src("lm_head.weight", row_limit=row_limit), "derived from output.weight"
        mtp_head = {"mtp.fc.weight": "mtp.fc.weight",
                    "mtp.pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden.weight",
                    "mtp.pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
                    "mtp.norm.weight": "mtp.norm.weight"}
        if name in mtp_head:
            return self._src(mtp_head[name]), ""
        if name.startswith("mtp."):
            return self._mtp_source(name, row_limit), ""
        m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if not m:
            return None, "no source rule"
        l, rest = int(m.group(1)), m.group(2)
        pre = ("mtp.layers.0." if self.mtp_layer is not None and l == self.mtp_layer
               else f"model.language_model.layers.{l}.")
        hc = re.fullmatch(r"(attn|ffn)_hc_(down|up|inject|norm)\.weight", rest)
        if hc:
            part = {"down": "input_mix_weight_down.weight",
                    "up": "input_mix_weight_up.weight",
                    "inject": "block_inject_weight.weight",
                    "norm": "hc_norm.weight"}[hc.group(2)]
            grp = {"attn": "attn_hyper_connection", "ffn": "mlp_hyper_connection"}[hc.group(1)]
            return self._src(pre + f"{grp}.{part}"), ""
        ple = {"ple_conv.weight": "ple.conv1d.weight",
               "ple_key.weight": "ple.key_proj.weight",
               "ple_value.weight": "ple.value_proj.weight",
               "ple_norm_conv.weight": "ple.norm_conv.weight",
               "ple_norm_key.weight": "ple.norm_key.weight",
               "ple_norm_query.weight": "ple.norm_query.weight"}
        if rest in ple:
            return self._src(pre + ple[rest], row_limit=row_limit), ""
        if rest == "ple_ngram.weight":
            return self._ple_ngram(pre, row_limit), ""
        qsa = {"qsa_q_norm.weight": "self_attn.indexer.q_layernorm.weight",
               "qsa_k_norm.weight": "self_attn.indexer.k_layernorm.weight"}
        if rest in qsa:
            return self._src(pre + qsa[rest], row_limit=row_limit), ""
        if rest == "qsa_qk.weight":
            return self._src(pre + "self_attn.indexer.index_qk_proj.weight",
                             row_limit=row_limit), ""
        # Integer metadata the arch declares beside the weights; llama.cpp's UD GGUF does not
        # carry it, so it is compared bit-exactly with the checkpoint and never with the GGUF.
        meta = {"ple_mult": "ple.ple_embedding.layer_multipliers",
                "ple_offsets": "ple.ple_embedding.ngram_heads_offsets",
                "ple_vocab_sizes": "ple.ple_embedding.ngram_heads_vocab_sizes"}
        if rest in meta:
            return self._src(pre + meta[rest]), "integer metadata"
        e = re.fullmatch(r"ffn_(gate_up|down)_exps\.(\d+)\.weight", rest)
        if e:
            kind, ex = e.group(1), int(e.group(2))
            p = pre + "mlp.experts." + ("gate_up_proj" if kind == "gate_up" else "down_proj")
            a = self._src(p, ex)
            return (a.reshape(a.shape[-2], a.shape[-1]) if a is not None and a.ndim > 2 else a), ""
        one = {
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "attn_qg.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
            "attn_q_norm.weight": "self_attn.q_norm.weight",
            "attn_k_norm.weight": "self_attn.k_norm.weight",
            "ssm_inz.weight": None, "ssm_out.weight": "linear_attn.out_proj.weight",
            "ssm_ab.weight": None, "ssm_a_log": "linear_attn.A_log",
            "ssm_dt.bias": "linear_attn.dt_bias", "ssm_conv1d.weight": "linear_attn.conv1d.weight",
            "ssm_norm.weight": "linear_attn.norm.weight",
            "ffn_gate_inp.weight": "mlp.gate.weight",
            "ffn_gate_shexp.weight": "mlp.shared_expert_gate.weight",
            "ffn_gate_up_shexp.weight": None,
            "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
        }
        if rest in one:
            p = one[rest]
            if p is None:
                if rest == "ssm_inz.weight":
                    a = self._src(pre + "linear_attn.in_proj_qkv.weight")
                    b = self._src(pre + "linear_attn.in_proj_z.weight")
                    return (np.concatenate([a, b], axis=0) if a is not None and b is not None
                            else None), ""
                if rest == "ssm_ab.weight":
                    a = self._src(pre + "linear_attn.in_proj_a.weight")
                    b = self._src(pre + "linear_attn.in_proj_b.weight")
                    return (np.concatenate([a, b], axis=0) if a is not None and b is not None
                            else None), ""
                if rest == "ffn_gate_up_shexp.weight":
                    a = self._src(pre + "mlp.shared_expert.gate_proj.weight")
                    b = self._src(pre + "mlp.shared_expert.up_proj.weight")
                    return (np.concatenate([a, b], axis=0) if a is not None and b is not None
                            else None), ""
            a = self._src(pre + p)
            if a is not None and rest == "ssm_conv1d.weight":
                a = a.reshape(a.shape[0], -1)
            return a, ""
        return None, "no source rule"

    def gguf(self, name, row_limit=None):
        """(GGUF array or None, note). Value-head order and the -exp(A_log) convention are undone."""
        import numpy as np
        if name.startswith("v."):
            return None, "not in the text GGUF"
        if name in ("token_embd.weight", "output.weight", "output_norm.weight"):
            return self._gg(name, row_limit=row_limit), ""
        if name == "mtp.draft_head.weight":
            return self._gg("output.weight", row_limit=row_limit), "derived from output.weight"
        mtp_head = {"mtp.fc.weight": "blk.40.nextn.eh_proj.weight",
                    "mtp.pre_fc_norm_hidden.weight": "blk.40.nextn.hnorm.weight",
                    "mtp.pre_fc_norm_embedding.weight": "blk.40.nextn.enorm.weight",
                    "mtp.norm.weight": "blk.40.nextn.shared_head_norm.weight"}
        if name in mtp_head:
            return self._gg(mtp_head[name]), ""
        if name in ("output_hc_down.weight", "output_hc_up.weight", "output_hc_norm.weight"):
            return self._gg(name), ""
        if name == "blk.1.ple_ngram.weight":
            return self._gg("per_layer_token_embd.weight", row_limit=row_limit), ""
        m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if not m:
            return None, "no GGUF rule"
        l, rest = int(m.group(1)), m.group(2)
        b = f"blk.{l}."
        e = re.fullmatch(r"ffn_(gate_up|down)_exps\.(\d+)\.weight", rest)
        if e:
            kind, ex = e.group(1), int(e.group(2))
            if kind == "down":
                return self._gg(b + "ffn_down_exps.weight", ex), ""
            g = self._gg(b + "ffn_gate_exps.weight", ex)
            u = self._gg(b + "ffn_up_exps.weight", ex)
            return (np.concatenate([g, u], axis=0) if g is not None and u is not None else None), ""
        if rest == "qsa_qk.weight":
            q = self._gg(b + "indexer.q_proj.weight")
            k = self._gg(b + "indexer.k_proj.weight")
            return (np.concatenate([q, k], axis=0) if q is not None and k is not None else None), ""
        hc = re.fullmatch(r"(attn|ffn)_hc_(down|up|inject|norm)\.weight", rest)
        if hc:
            return self._gg(b + f"hc_{hc.group(1)}_{hc.group(2)}.weight",
                            row_limit=row_limit), ""
        one = {"attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "post_attention_norm.weight",
               "attn_qg.weight": "attn_q.weight", "attn_k.weight": "attn_k.weight",
               "attn_v.weight": "attn_v.weight", "attn_output.weight": "attn_output.weight",
               "attn_q_norm.weight": "attn_q_norm.weight", "attn_k_norm.weight": "attn_k_norm.weight",
               "ssm_norm.weight": "ssm_norm.weight",
               "ffn_gate_inp.weight": "ffn_gate_inp.weight",
               "ffn_gate_shexp.weight": "ffn_gate_inp_shexp.weight",
               "ffn_down_shexp.weight": "ffn_down_shexp.weight",
               "ple_conv.weight": "ple_conv1d.weight",
               "ple_key.weight": "ple_key.weight", "ple_value.weight": "ple_value.weight",
               "ple_norm_conv.weight": "ple_norm_conv.weight",
               "ple_norm_key.weight": "ple_norm_key.weight",
               "ple_norm_query.weight": "ple_norm_query.weight",
               "qsa_q_norm.weight": "indexer.q_norm.weight",
               "qsa_k_norm.weight": "indexer.k_norm.weight"}
        if rest in one:
            a = self._gg(b + one[rest], row_limit=row_limit)
            return a, ""
        if rest == "ssm_dt.bias":
            nk = int(self._ssm("group_count"))
            nv = int(self._ssm("time_step_rank"))
            a = self._gg(b + "ssm_dt.bias")
            if a is None:
                return None, ""
            return self._untile(a.reshape(1, -1), 1, 0, nv, nv // nk, 1), "value heads regrouped"
        if rest == "ssm_a_log":
            a = self._gg(b + "ssm_a")
            if a is None:
                return None, ""
            nk = int(self._ssm("group_count"))
            nv = int(self._ssm("time_step_rank"))
            a = self._untile(a.reshape(-1), 0, 0, nv, nv // nk, 1) if a.size >= nv else a
            return np.log(-a.astype(np.float64)).astype(np.float32), "log(-ssm_a)"
        if rest == "ssm_inz.weight":
            nk = int(self._ssm("group_count"))
            nv = int(self._ssm("time_step_rank"))
            hk = int(self._ssm("state_size"))
            inner = self._ssm("inner_size")
            hv = int(inner) // nv
            a = int(nv // nk)
            qk, vdim = 2 * nk * hk, nv * hv
            g = self._gg(b + "attn_qkv.weight")
            z = self._gg(b + "attn_gate.weight")
            if g is None or z is None:
                return None, ""
            g = self._untile(g, 0, qk, vdim, a, hv)
            z = self._untile(z, 0, 0, vdim, a, hv)
            return np.concatenate([g, z], axis=0), "value heads regrouped"
        if rest == "ssm_ab.weight":
            nk = int(self._ssm("group_count"))
            nv = int(self._ssm("time_step_rank"))
            a = int(nv // nk)
            al = self._gg(b + "ssm_alpha.weight")
            be = self._gg(b + "ssm_beta.weight")
            if al is None or be is None:
                return None, ""
            return np.concatenate([self._untile(al, 0, 0, nv, a, 1),
                                   self._untile(be, 0, 0, nv, a, 1)], axis=0), "value heads regrouped"
        if rest == "ffn_gate_up_shexp.weight":
            g = self._gg(b + "ffn_gate_shexp.weight")
            u = self._gg(b + "ffn_up_shexp.weight")
            return (np.concatenate([g, u], axis=0) if g is not None and u is not None else None), ""
        if rest in ("ssm_out.weight", "ssm_conv1d.weight"):
            x = self._gg(b + rest)
            if x is None:
                return None, ""
            nk = int(self._ssm("group_count"))
            nv = int(self._ssm("time_step_rank"))
            hk = int(self._ssm("state_size"))
            hv = int(self._ssm("inner_size")) // nv
            a = int(nv // nk)
            qk, vdim = 2 * nk * hk, nv * hv
            start = 0 if rest == "ssm_out.weight" else qk
            return (self._untile(x, 1 if rest == "ssm_out.weight" else 0, start, vdim, a, hv),
                    "value heads regrouped")
        return None, "no GGUF rule"



def _rad_role(name):
    if "ffn_gate_up_exps" in name:
        return "routed-expert gate/up"
    if name.startswith("v."):
        return "vision"
    return role_of(name)


def _load_rad_verify():
    import rad_verify
    return rad_verify


def _load_ref_classes():
    import route_b
    import rad_verify
    return route_b.Safetensors, rad_verify.Gguf


def tensor_class(hash_equal_gguf, hash_equal_source, has_source, has_gguf, kept=False, derived=False):
    """What kind of evidence a weight can have, so each class is judged by the right rule.
    EXACT: the planes decode bit-identically to the UD GGUF (or, for a plain copy, to the source).
    DERIVED / KEPT: the plan says the GGUF does not carry it; judged against the source only.
    UNVERIFIABLE: nothing to compare with (reported, never silently passed).
    REQUANT: everything else, judged numerically against whichever references exist."""
    if hash_equal_gguf or (hash_equal_source and not has_gguf):
        return "EXACT"
    if derived:
        return "DERIVED"
    if kept:
        return "KEPT"
    if not has_source and not has_gguf:
        return "UNVERIFIABLE"
    return "REQUANT"


def error_bound(tol, ud_rmse, slack=1.5):
    """The relative-RMSE a requantised weight may show. A fixed cap alone fails tensors whose
    *reference* GGUF is itself far from bf16 (Unsloth's IQ3_XXS sit near 0.28), so the bound
    follows the UD file's own error against the source: max(cap, slack * that error)."""
    return tol if ud_rmse is None else max(tol, slack * ud_rmse)


UNVERIFIABLE_MAX_FRACTION = 0.01  # of container bytes; more than this and "unverifiable" hides a bad --source


def _policy_verdict(passed, kp_weights=0, unverifiable=0):
    """PASS / PARTIAL / FAIL. PARTIAL means every weight that could be checked passed, but some
    carry plugin-private planes the Python decoder honestly cannot read (checked by the engine) or
    have no reference at all (UNVERIFIABLE; listed in the report)."""
    if not passed:
        return "FAIL"
    return "PARTIAL" if (kp_weights or unverifiable) else "PASS"


def _policy_line(policy, verdict, n_decoded, kp_weights, kp_bytes, kp_dtypes, out=None):
    base = f"policy {policy}: {verdict}"
    if verdict == "PARTIAL":
        names = ', '.join(kp_dtypes) or "plugin-private"
        base += (f" - kernel-private planes not Python-decoded "
                 f"({kp_weights} weight(s), {kp_bytes/2**30:.3f} GiB: {names}); "
                 f"PASS on {n_decoded} Python-decoded weight(s)")
    if out:
        base += f"; report {out}"
    return base


def verify(args):
    """rad_verify-style hashes, per-tensor relative RMSE against the bf16 source and the GGUF's
    own dequantisation, a size table by role/type, and the exactness/refusal policy verdict."""
    import numpy as np
    from radfile import RadFile, KernelPrivateDtype, dt_name, DT, plane_values
    Safetensors, Gguf = _load_ref_classes()
    rf = RadFile(args.rad)
    src = Safetensors(args.source) if args.source else None
    pos = list(getattr(args, "gguf_pos", None) or [])
    given = args.gguf if isinstance(args.gguf, list) else ([args.gguf] if args.gguf else [])
    args.gguf = (given + pos) or None
    if args.gguf:
        paths = args.gguf if isinstance(args.gguf, list) else [args.gguf]
        ggs = [Gguf(p) for p in paths]
        class Multi:
            pass
        gg = Multi()
        gg.parts = ggs
        gg.T, gg.kv = {}, {}
        for g in ggs:
            gg.T.update(g.T)
            gg.kv.update(g.kv)
        def rows_f32(name, r0, nr, expert=None):
            for g in ggs:
                if name in g.T:
                    return g.rows_f32(name, r0, nr, expert)
            raise KeyError(name)
        def logical(t):
            for g in ggs:
                if any(v is t for v in g.T.values()):
                    return g.logical(t)
            raise KeyError(t.get("name", "?"))
        gg.rows_f32, gg.logical = rows_f32, logical
    else:
        gg = None
    refs = Refs(rf, src, gg, 0)
    # The plan says which weights the UD GGUF does not carry (kept from bf16) or derives; those
    # legitimately have no GGUF reference and are compared with the source only.
    planned, kept_set, derived_set = None, set(), set()
    if args.plan_json and os.path.exists(args.plan_json):
        try:
            pj = json.load(open(args.plan_json))
            planned = [w["target"] for w in pj.get("tensors", []) if w.get("target")]
            kept_set = {w["name"] for w in pj.get("kept", [])}
            derived_set = {w["name"] for w in pj.get("derived", [])}
        except Exception:
            planned, kept_set, derived_set = None, set(), set()
    names = [e.name for e in rf.entries]
    if args.only:
        names = [n for n in names if re.search(args.only, n)]
    expert_sample = getattr(args, "max_experts", 0) or 0
    if expert_sample:
        keep, seen = [], collections.Counter()
        for n in names:
            m = re.fullmatch(r"blk\.(\d+)\.(ffn_gate_up_exps|ffn_down_exps)\.(\d+)\.weight", n)
            if not m:
                keep.append(n)
                continue
            key = (m.group(1), m.group(2))
            seen[key] += 1
            if seen[key] <= expert_sample:
                keep.append(n)
        names = keep
    rows = []
    tot = 0
    by_role = collections.defaultdict(lambda: [0, 0])
    by_enc = collections.defaultdict(lambda: [0, 0])
    fails = []
    missing_src = missing_gg = 0
    hash_ok = hash_n = 0
    kp_weights = kp_bytes = 0
    kp_dtypes = collections.Counter()
    meta_weights = meta_equal = meta_bytes = 0
    src_num = src_den = gg_num = gg_den = 0.0
    unverifiable, unverifiable_bytes = [], 0
    t0 = time.time()
    for i, name in enumerate(names):
        e = rf.by_name[name]
        role = _rad_role(name)
        enc = e.enc.name()
        tot += e.bytes
        by_role[role][0] += 1; by_role[role][1] += e.bytes
        by_enc[enc][0] += 1; by_enc[enc][1] += e.bytes
        kp_planes = e.enc.kernel_private_planes()
        if kp_planes:
            # Not a failure and not a pass: the core cannot interpret these planes, so neither
            # does this reader. Record the weight and keep the verdict honest.
            kp_weights += 1
            kp_bytes += e.bytes
            for p in kp_planes:
                kp_dtypes[dt_name(p.dtype)] += 1
            rec = dict(name=name, role=role, encoding=enc, bytes=e.bytes, shape=list(e.shape),
                       sampled_rows=None, rmse_source=None, rmse_gguf=None, sha256_rad=None,
                       sha256_source=None, sha256_gguf=None, hash_equal_source=None,
                       hash_equal_gguf=None, kernel_private=True,
                       kernel_private_dtypes=[dt_name(p.dtype) for p in kp_planes],
                       note="kernel-private plane(s): not Python-decoded",
                       status="KERNEL-PRIVATE",
                       why=["kernel-private planes not Python-decoded: " +
                            ", ".join(dt_name(p.dtype) for p in kp_planes)])
            rows.append(rec)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(names)}  {time.time()-t0:.0f}s", flush=True)
            continue
        dt0 = e.enc.planes[0].dtype
        if e.enc.scheme == "plain" and dt0 in DT and not dt_name(dt0).startswith(("f", "bf")):
            # Integer metadata (the PLE layer multipliers, n-gram head offsets and vocab sizes):
            # bit-exact, not a float weight and not in llama.cpp's UD GGUF.
            sa, snote = refs.source(name) if src is not None else (None, "no --source")
            v = plane_values(rf, e, 0).reshape(-1).astype(np.int64)
            meta_weights += 1
            meta_bytes += e.bytes
            rh = hashlib.sha256(np.ascontiguousarray(v, "<i8").tobytes()).hexdigest()
            sh = None
            if sa is None:
                status, why, eq = "METADATA", ["integer metadata; no source reference"], None
            else:
                a = np.asarray(sa).reshape(-1)
                eq = bool(a.size == v.size and np.array_equal(v, a))
                sh = hashlib.sha256(np.ascontiguousarray(a, "<i8").tobytes()).hexdigest()
                status = "ok" if eq else "FAIL"
                why = [] if eq else ["integer metadata differs from the source"]
                meta_equal += 1 if eq else 0
            rec = dict(name=name, role=role, encoding=enc, bytes=e.bytes, shape=list(e.shape),
                       sampled_rows=None, rmse_source=None, rmse_gguf=None, sha256_rad=rh,
                       sha256_source=sh, sha256_gguf=None, hash_equal_source=eq,
                       hash_equal_gguf=None, kind="metadata", note=snote,
                       status=status, why=why)
            if status == "FAIL":
                fails.append(rec)
            rows.append(rec)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(names)}  {time.time()-t0:.0f}s", flush=True)
            continue
        nrows_full = int(np.prod(e.shape[:-1])) if len(e.shape) > 1 else 1
        ncols = e.shape[-1] if e.shape else 1
        # A 248K x 2048 head as f32 is 2 GB on each side; compare the first 4096 rows of such
        # a weight and say so. The table experts, which are where the quantisation happens, are
        # small enough to check whole.
        big = nrows_full * ncols > (40 << 20)
        check_rows = min(nrows_full, 4096) if big else nrows_full
        nrows = check_rows
        chunk = max(1, (8 << 20) // max(1, ncols))
        sa, snote = refs.source(name, row_limit=check_rows if big else None) \
            if src is not None else (None, "no --source")
        ga, gnote = refs.gguf(name, row_limit=check_rows if big else None) \
            if gg is not None else (None, "no --gguf")
        def _fit(a, what):
            if a is None:
                return None, ""
            if a.size == nrows * ncols:
                return a.reshape(nrows, ncols), ""
            if a.squeeze().size == nrows * ncols:
                return a.reshape(nrows, ncols), ""
            return None, f"{what} shape {a.shape} != {e.shape}"
        sa, sa_note = _fit(sa, "source")
        ga, ga_note = _fit(ga, "GGUF")
        if (sa is not None and ga is not None and name != "ssm_a_log" and
                sa.size == ga.size and (nrows == 1 or ncols == 1)):
            # llama.cpp stores this family's RMSNorm gains as w + 1; the bf16 source holds w.
            # Detect the convention per tensor instead of trusting a name list.
            import numpy as _np
            d = float(_np.median(ga.reshape(-1).astype(_np.float64) -
                                 sa.reshape(-1).astype(_np.float64)))
            if abs(d - 1.0) < 0.1:
                ga = (ga.astype(_np.float64) - 1.0).astype(_np.float32)
        if sa is None:
            missing_src += 1
        if ga is None and gg is not None:
            missing_gg += 1
        rnum = rden = gnum = gden = unum = 0.0
        rh = hashlib.sha256()
        sh = hashlib.sha256()
        gh = hashlib.sha256()
        rad_all = None
        if nrows * ncols <= (40 << 20):
            rad_all = decode_entry(rf, e, 0, nrows)
            rh.update(np.ascontiguousarray(rad_all, "<f4").tobytes())
            if sa is not None:
                sh.update(np.ascontiguousarray(sa, "<f4").tobytes())
            if ga is not None:
                gh.update(np.ascontiguousarray(ga, "<f4").tobytes())
            rnum = float(((rad_all.astype(np.float64) - sa.astype(np.float64)) ** 2).sum()) if sa is not None else 0
            rden = float((sa.astype(np.float64) ** 2).sum()) if sa is not None else 0
            gnum = float(((rad_all.astype(np.float64) - ga.astype(np.float64)) ** 2).sum()) if ga is not None else 0
            gden = float((ga.astype(np.float64) ** 2).sum()) if ga is not None else 0
            unum = float(((ga.astype(np.float64) - sa.astype(np.float64)) ** 2).sum()) \
                if (ga is not None and sa is not None) else 0
        else:
            for r0 in range(0, nrows, chunk):
                nr = min(chunk, nrows - r0)
                d = decode_entry(rf, e, r0, nr)
                rh.update(np.ascontiguousarray(d, "<f4").tobytes())
                if sa is not None:
                    s = sa[r0:r0 + nr]
                    sh.update(np.ascontiguousarray(s, "<f4").tobytes())
                    sd = s.astype(np.float64)
                    rnum += float(((d.astype(np.float64) - sd) ** 2).sum()); rden += float((sd ** 2).sum())
                if ga is not None:
                    g = ga[r0:r0 + nr]
                    gh.update(np.ascontiguousarray(g, "<f4").tobytes())
                    gd = g.astype(np.float64)
                    gnum += float(((d.astype(np.float64) - gd) ** 2).sum()); gden += float((gd ** 2).sum())
                    if sa is not None:
                        unum += float(((gd - sa[r0:r0 + nr].astype(np.float64)) ** 2).sum())
        e_rmse = (np.sqrt(rnum / rden) if rden > 0 else (0.0 if sa is not None else None))
        g_rmse = (np.sqrt(gnum / gden) if gden > 0 else (0.0 if ga is not None else None))
        if sa is not None:
            src_num += rnum; src_den += rden
        if ga is not None:
            gg_num += gnum; gg_den += gden
        eq_src = (rh.hexdigest() == sh.hexdigest()) if sa is not None else None
        eq_gg = (rh.hexdigest() == gh.hexdigest()) if ga is not None else None
        if sa is not None:
            hash_n += 1; hash_ok += 1 if eq_src else 0
        draft = name == "mtp.draft_head.weight"
        tol = args.draft_tol if draft else args.tol
        ud_rmse = (float(np.sqrt(unum / rden)) if (ga is not None and sa is not None and rden > 0) else None)
        klass = tensor_class(eq_gg, eq_src, sa is not None, ga is not None,
                             kept=name in kept_set, derived=name in derived_set)
        bound = error_bound(tol, ud_rmse, getattr(args, "ud_slack", 1.5))
        bad = []
        if klass == "EXACT":
            pass  # bit-identical to its reference: the error against bf16 is the reference's own
        elif klass == "UNVERIFIABLE":
            unverifiable.append(name); unverifiable_bytes += e.bytes
        else:
            if e_rmse is not None and e_rmse > bound:
                bad.append(f"source RMSE {e_rmse:.3f} > {bound:.3f}")
            if (ga is None and gg is not None and not draft and not name.startswith("v.")
                    and name not in kept_set and name not in derived_set
                    and not getattr(args, "skip_gguf", False)):
                bad.append("no GGUF reference")
            elif g_rmse is not None and g_rmse > bound:
                bad.append(f"GGUF RMSE {g_rmse:.3f} > {bound:.3f}")
        rec = dict(name=name, role=role, encoding=enc, bytes=e.bytes, shape=list(e.shape),
                   sampled_rows=(check_rows if big else None),
                   cls=klass, rmse_ud_vs_source=ud_rmse, bound=bound,
                   rmse_source=e_rmse, rmse_gguf=g_rmse, sha256_rad=rh.hexdigest(),
                   sha256_source=sh.hexdigest() if sa is not None else None,
                   sha256_gguf=gh.hexdigest() if ga is not None else None,
                   hash_equal_source=eq_src, hash_equal_gguf=eq_gg,
                   note="; ".join(x for x in (snote, gnote, sa_note, ga_note) if x) or None,
                   status="FAIL" if bad else ("UNVERIFIABLE" if klass == "UNVERIFIABLE" else "ok"),
                   why=bad)
        if bad:
            fails.append(rec)
        rows.append(rec)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(names)}  {time.time()-t0:.0f}s", flush=True)

    # coverage against the plan, and the finality policy
    missing_plan = []
    if planned:
        have = set(names)
        for t in planned:
            if "*" in t:
                rx = "^" + re.escape(t).replace(r"\*", r"\d+") + "$"
                if not any(re.match(rx, n) for n in have):
                    missing_plan.append(t)
            elif t not in have:
                missing_plan.append(t)
    src_rmse = np.sqrt(src_num / src_den) if src_den else None
    gg_rmse = np.sqrt(gg_num / gg_den) if gg_den else None
    policy = args.policy
    if policy == "auto":
        policy = "lossless" if (rf.quant or "").startswith("gguf_ftype_") else "quantised"
    if policy == "lossless":
        bad = [r for r in rows if r["hash_equal_gguf"] is False
               and r["name"] not in ("ssm_a_log", "mtp.draft_head.weight")]
        bad += [r for r in rows if r["hash_equal_source"] is False and r["hash_equal_gguf"] is False]
        passed = not bad and not missing_plan
    else:
        passed = not fails and not missing_plan
    if getattr(args, "skip_gguf", False):
        passed = not any(r["status"] == "FAIL" for r in rows) and not missing_plan
    if unverifiable and unverifiable_bytes > UNVERIFIABLE_MAX_FRACTION * max(1, tot):
        passed = False
        print(f"policy: {len(unverifiable)} unverifiable weight(s) are {unverifiable_bytes/max(1,tot):.1%} "
              f"of the container (limit {UNVERIFIABLE_MAX_FRACTION:.0%}); is --source/--gguf the right pair?")
    verdict = _policy_verdict(passed, kp_weights, len(unverifiable))
    by_class = collections.Counter(r.get("cls", "?") for r in rows)
    kp_names = [d for d, _ in kp_dtypes.most_common()]
    report = dict(tool="ud_map verify", rad=args.rad, source=args.source,
                  gguf=(args.gguf if isinstance(args.gguf, list) else [args.gguf]) if args.gguf else None,
                  policy=policy, tol=args.tol, draft_tol=args.draft_tol,
                  weights=len(rows), bytes=tot,
                  container_bytes=os.path.getsize(args.rad),
                  expert_sample_per_layer=(expert_sample or None),
                  rmse_source=src_rmse, rmse_gguf=gg_rmse,
                  hash_equal_source=f"{hash_ok}/{hash_n}",
                  python_decoded_weights=len(rows) - kp_weights,
                  metadata=dict(weights=meta_weights, equal=meta_equal, bytes=meta_bytes,
                                note=("integer metadata, compared bit-exactly with the source"
                                      if meta_weights else None)),
                  kernel_private=dict(weights=kp_weights, bytes=kp_bytes,
                                      dtypes=kp_names,
                                      note=("kernel-private planes not Python-decoded; "
                                            "checked by the engine" if kp_weights else None)),
                  missing_source=missing_src, missing_gguf=missing_gg,
                  missing_plan=missing_plan,
                  failures=[r["name"] for r in fails], pass_=passed, verdict=verdict,
                  by_class=dict(by_class), ud_slack=getattr(args, "ud_slack", 1.5),
                  unverifiable=dict(weights=len(unverifiable), bytes=unverifiable_bytes,
                                    names=unverifiable[:64]),
                  by_role={k: dict(n=v[0], bytes=v[1]) for k, v in sorted(by_role.items())},
                  by_encoding={k: dict(n=v[0], bytes=v[1]) for k, v in sorted(by_enc.items())},
                  tensors=rows, seconds=round(time.time() - t0, 1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    print(f"verify: {len(rows)} weights, {tot/2**30:.3f} GiB; overall RMSE vs source "
          f"{src_rmse if src_rmse is None else round(src_rmse,4)}, vs GGUF "
          f"{gg_rmse if gg_rmse is None else round(gg_rmse,4)}; hash-equal source {hash_ok}/{hash_n}")
    print(f'{"role":28s}{"n":>6s}{"GiB":>10s}')
    for k, v in sorted(by_role.items(), key=lambda kv: -kv[1][1]):
        print(f'{k:28s}{v[0]:6d}{v[1]/2**30:10.3f}')
    print("by encoding:")
    for k, v in sorted(by_enc.items(), key=lambda kv: -kv[1][1]):
        print(f'  {k:70s} {v[0]:6d} {v[1]/2**30:9.3f} GiB')
    if missing_plan:
        print(f"policy: {len(missing_plan)} planned target(s) are not in the container: "
              f"{missing_plan[:8]}")
    if fails:
        print(f"policy: {len(fails)} weight(s) outside the exactness/refusal tolerance")
        for r in fails[:12]:
            print("  FAIL", r["name"], "; ".join(r["why"]),
                  f"src={r['rmse_source']} gguf={r['rmse_gguf']}")
    print("by class: " + ", ".join(f"{k} {v}" for k, v in sorted(by_class.items())))
    if unverifiable:
        print(f"unverifiable: {len(unverifiable)} weight(s), {unverifiable_bytes} B, no source and no "
              f"GGUF reference: {', '.join(unverifiable[:8])}{' ...' if len(unverifiable) > 8 else ''}")
    if kp_weights:
        print(f"kernel-private: {kp_weights} weight(s), {kp_bytes/2**30:.3f} GiB, "
              f"dtypes {', '.join(kp_names)}; not Python-decoded, checked by the engine")
    if meta_weights:
        print(f"integer metadata: {meta_weights} tensor(s), {meta_bytes} B, "
              f"bit-equal to the source {meta_equal}/{meta_weights} (not in the GGUF)")
    print(_policy_line(policy, verdict, len(rows) - kp_weights, kp_weights, kp_bytes,
                       kp_names, args.out))
    return passed


# ==================================================================== score

def _kld_run(args, model, mode, out_json=None, ref_dir=None):
    radiance = _tool(args, "radiance", "radiance", "RADIANCE_RADIANCE")
    cmd = [radiance, "--model", model, "--tp", str(args.tp),
           "--max-model-len", str(args.max_model_len),
           "--kv-cache-dtype", args.kv_cache_dtype]
    if args.num_speculative_tokens is not None:
        cmd += ["--num-speculative-tokens", str(args.num_speculative_tokens)]
    if args.kernels:
        cmd += ["--kernels", args.kernels]
    if mode == "record":
        cmd += ["--kld-record", ref_dir, "--kld-corpus", args.corpus]
        if args.top:
            cmd += ["--kld-top", str(args.top)]
    else:
        cmd += ["--kld-ref", ref_dir]
        if out_json:
            cmd += ["--kld-out", out_json]
    cmd += list(args.extra_arg or [])
    cmd = _with_flock(args, cmd)
    return cmd


def score_kld(args):
    if not args.reference or not args.corpus:
        sys.exit("ud_map score kld: give --reference and --corpus")
    os.makedirs(args.out_dir, exist_ok=True)
    ref_dir = args.ref_dir or os.path.join(args.out_dir, "kld-ref")
    report = dict(tool="ud_map score kld", reference=args.reference, corpus=args.corpus,
                  ref_dir=ref_dir, candidates=[], phases=[])
    if args.record and not os.path.exists(os.path.join(ref_dir, "kld.json")):
        cmd = _kld_run(args, args.reference, "record", ref_dir=ref_dir)
        rc, text = _run(cmd, os.path.join(args.out_dir, "kld-record.log"),
                        env=_child_env(args), timeout=args.timeout)
        report["phases"].append(dict(name="record", command=cmd, rc=rc,
                                     log=os.path.join(args.out_dir, "kld-record.log")))
        if rc != 0:
            with open(os.path.join(args.out_dir, "score.json"), "w") as f:
                json.dump(report, f, indent=1)
            return rc
    for spec in args.candidate:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            sys.exit("ud_map score kld: --candidate wants NAME=PATH")
        out_json = os.path.join(args.out_dir, f"{name}.kld.json")
        cmd = _kld_run(args, path, "score", out_json=out_json, ref_dir=ref_dir)
        rc, text = _run(cmd, os.path.join(args.out_dir, f"{name}.log"),
                        env=_child_env(args), timeout=args.timeout)
        rec = dict(name=name, model=path, rc=rc, kld_json=out_json if rc == 0 else None)
        if rc == 0 and os.path.exists(out_json):
            d = json.load(open(out_json))
            a = d.get("all", {})
            rec.update(docs=d.get("docs"), positions=a.get("positions"),
                       mean_kl=a.get("kld", {}).get("mean"),
                       median_kl=a.get("kld", {}).get("median"),
                       p99_kl=a.get("kld", {}).get("p99"),
                       top1=a.get("top1_agreement"),
                       ppl=a.get("ppl", {}).get("candidate"),
                       ppl_ratio=a.get("ppl", {}).get("ratio"))
        report["candidates"].append(rec)
    with open(os.path.join(args.out_dir, "score.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("score kld: " + os.path.join(args.out_dir, "score.json"))
    for c in report["candidates"]:
        print(f"  {c['name']:12s} mean KL {c.get('mean_kl')}  top-1 {c.get('top1')}  "
              f"PPL ratio {c.get('ppl_ratio')}")
    return 0 if all(c["rc"] == 0 for c in report["candidates"]) else 1


def _free_port(base=8210):
    import socket
    for p in range(base, base + 40):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()
    raise RuntimeError("no free port")


def _server_cmd(args, model, port, max_model_len=None, spec=None):
    radiance = _tool(args, "radiance", "radiance", "RADIANCE_RADIANCE")
    cmd = [radiance, "--model", model, "--host", "127.0.0.1", "--port", str(port),
           "--tp", str(args.tp),
           "--max-model-len", str(max_model_len or args.max_model_len),
           "--kv-cache-dtype", args.kv_cache_dtype,
           "--max-num-seqs", str(getattr(args, "max_num_seqs", 4))]
    if spec is None:
        spec = args.num_speculative_tokens
    if spec is not None:
        cmd += ["--num-speculative-tokens", str(spec)]
    if args.kernels:
        cmd += ["--kernels", args.kernels]
    if getattr(args, "api_key", None):
        cmd += ["--api-key-env", "RADIANCE_UD_MAP_KEY"]
    cmd += list(args.extra_arg or [])
    return _with_flock(args, cmd)


def _wait_server(port, proc, key=None, timeout=900):
    import urllib.request
    h = {"Authorization": "Bearer " + key} if key else {}
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/models", headers=h), timeout=5).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def _stop_server(proc):
    """Terminate the server's whole process group: a flock wrapper is a parent, and terminating
    it alone would leave the engine holding the card and the GPU lock."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)


def _metrics(port, key=None):
    import urllib.request
    h = {"Authorization": "Bearer " + key} if key else {}
    try:
        txt = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/metrics", headers=h), timeout=20).read().decode()
    except Exception:
        return {}
    out = {}
    for ln in txt.splitlines():
        if ln.startswith("radiance"):
            parts = ln.split()
            if len(parts) >= 2:
                try:
                    # Keep the labelled key and the bare metric name, so a caller can ask for
                    # radiance:draft_acceptance_ratio without knowing the engine's labels.
                    out[parts[0]] = float(parts[-1])
                    out[parts[0].split("{")[0]] = float(parts[-1])
                except ValueError:
                    pass
    return out


def _chat(port, key, prompt, max_tokens, tools=None, think=False, messages=None):
    import urllib.request
    msgs = messages if messages is not None else [{"role": "user", "content": prompt}]
    body = {"model": "radiance", "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    rec = dict(ttft=None, n=None, chunks=[], text="", error=None, tool_calls=None)
    t0 = time.time()
    try:
        if tools:
            body["stream"] = False
            with urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json.dumps(body).encode(), h), timeout=1800) as r:
                j = json.loads(r.read())
            msg = (j.get("choices") or [{}])[0].get("message", {})
            rec["text"] = msg.get("content") or ""
            rec["tool_calls"] = msg.get("tool_calls")
            rec["n"] = (j.get("usage") or {}).get("completion_tokens")
            rec["ttft"] = time.time() - t0
        else:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            with urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json.dumps(body).encode(), h), timeout=1800) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    d = line[5:].strip()
                    if d == "[DONE]":
                        break
                    j = json.loads(d)
                    if j.get("usage"):
                        rec["n"] = j["usage"].get("completion_tokens")
                    for ch in j.get("choices", []):
                        dl = ch.get("delta", {})
                        c = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                        if c:
                            now = time.time()
                            if rec["ttft"] is None:
                                rec["ttft"] = now - t0
                            rec["chunks"].append(now - t0)
                            rec["text"] += c
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    rec["total"] = time.time() - t0
    n = rec.get("n") or len(rec["chunks"])
    rec["decode_tok_s"] = ((n - 1) / (rec["total"] - rec["ttft"])
                          if rec["ttft"] and n > 1 and rec["total"] > rec["ttft"] else None)
    return rec


def _grade(item, rec):
    import ast
    if rec.get("error"):
        return False
    kind, exp, text = item.get("kind"), item.get("expected"), rec.get("text") or ""
    if kind == "mc":
        m = re.findall(r"\b([A-D])\b", text)
        return bool(m) and m[-1] == exp
    if kind == "gsm8k":
        m = re.findall(r"(-?[0-9][0-9,\.]*)", text.replace(",", ""))
        if not m:
            return False
        got = m[-1].rstrip(".")
        try:
            return abs(float(got) - float(exp)) < 1e-6
        except ValueError:
            return got == str(exp)
    if kind == "tool":
        try:
            want = ast.literal_eval(exp) if isinstance(exp, str) else exp
        except Exception:
            return False
        calls = rec.get("tool_calls") or []
        if not calls:
            return False
        fn = calls[0].get("function", {})
        if fn.get("name") != want.get("name"):
            return False
        try:
            got = json.loads(fn.get("arguments") or "{}")
        except Exception:
            return False
        return all(got.get(k) == v for k, v in (want.get("args") or {}).items())
    return None


def score_quality(args):
    if not args.model or not args.evalset:
        sys.exit("ud_map score quality: give --model and --evalset")
    os.makedirs(args.out_dir, exist_ok=True)
    key = args.api_key
    port = args.port or _free_port()
    env = _child_env(args)
    if key:
        env["RADIANCE_UD_MAP_KEY"] = key
    cmd = _server_cmd(args, args.model, port)
    log = open(os.path.join(args.out_dir, "server.log"), "w")
    print("+ " + shlex.join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    try:
        if not _wait_server(port, proc, key, args.timeout):
            print("ud_map score quality: the server did not come up; see " + log.name,
                  file=sys.stderr)
            return 1
        items = [json.loads(l) for l in open(args.evalset) if l.strip()]
        out_path = os.path.join(args.out_dir, "quality.jsonl")
        done = set()
        if os.path.exists(out_path) and not args.restart:
            done = {json.loads(l)["id"] for l in open(out_path)}
        results = []
        with open(out_path, "a") as fo:
            for it in items:
                if it["id"] in done:
                    continue
                rec = _chat(port, key, None, it.get("max_tokens", 512),
                            tools=it.get("tools"), think=bool(it.get("think")),
                            messages=it["messages"])
                rec.update(id=it["id"], kind=it.get("kind"), expected=it.get("expected"))
                rec["ok"] = _grade(it, rec)
                fo.write(json.dumps({k: v for k, v in rec.items() if k != "chunks"},
                                    ensure_ascii=False) + "\n")
                fo.flush()
                results.append(rec)
        # include earlier results for the summary
        allr = [json.loads(l) for l in open(out_path)]
        by = collections.defaultdict(lambda: [0, 0])
        for r in allr:
            by[r.get("kind")][0] += 1
            by[r.get("kind")][1] += int(bool(r.get("ok")))
        summary = dict(tool="ud_map score quality", model=args.model, evalset=args.evalset,
                       out=out_path, items=len(allr),
                       correct=sum(int(bool(r.get("ok"))) for r in allr),
                       by_kind={k: dict(n=v[0], correct=v[1]) for k, v in by.items()},
                       errors=sum(1 for r in allr if r.get("error")))
        with open(os.path.join(args.out_dir, "quality.summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(json.dumps(summary, indent=1))
        return 0
    finally:
        _stop_server(proc)
        log.close()


def score(args):
    return score_kld(args) if args.mode == "kld" else score_quality(args)


# ==================================================================== variations

def _measure_one(args, name, path):
    import threading
    key = args.api_key
    port = args.port or _free_port()
    env = _child_env(args)
    if key:
        env["RADIANCE_UD_MAP_KEY"] = key
    spec = args.num_speculative_tokens if args.num_speculative_tokens is not None else 2
    cmd = _server_cmd(args, path, port, spec=spec)
    log = open(os.path.join(args.out_dir, f"server-{name}.log"), "w")
    print("+ " + shlex.join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    try:
        if not _wait_server(port, proc, key, args.timeout):
            return dict(name=name, model=path, error="server did not start")
        before = _metrics(port, key)
        # A: two streams, 200 tokens. B: one ~8K-token prompt, 20 tokens.
        words = ("river stone lantern copper orbit velvet harbor quartz meadow signal "
                 "ember glacier thistle anchor mosaic canyon willow cipher falcon").split()
        import random
        r = random.Random(7)
        prefill = " ".join(r.choice(words) for _ in range(4000))
        recs = []
        th = [threading.Thread(target=lambda i=i: recs.append(
            _chat(port, key, f"Variant {name}-{i}: Write a detailed story.", 200)))
            for i in range(2)]
        t0 = time.time()
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.time() - t0
        b = _chat(port, key, prefill + "\n\nSummarize the passage in one sentence.", 20)
        after = _metrics(port, key)
        decs = [x["decode_tok_s"] for x in recs if x["decode_tok_s"]]
        ttfts = [x["ttft"] for x in recs if x["ttft"]]
        acc = after.get("radiance:draft_acceptance_ratio")
        steps = after.get("radiance:engine_steps_decode_total")
        toks = after.get("radiance:decode_tokens_total")
        tps = (toks / steps) if toks and steps else None
        return dict(name=name, model=path, file_bytes=os.path.getsize(path),
                    spec=spec,
                    A_decode_tok_s_1stream=decs[0] if decs else None,
                    A_decode_tok_s_agg=(sum(x.get("n") or 0 for x in recs) / wall
                                        if wall > 0 else None),
                    A_ttft_p50=(sorted(ttfts)[len(ttfts) // 2] if ttfts else None),
                    A_tokens_per_step=tps,
                    acceptance=acc,
                    B_prefill_ttft=b["ttft"], B_decode_tok_s=b["decode_tok_s"],
                    metrics_delta={k: after[k] - before.get(k, 0) for k in after
                                   if isinstance(after[k], float) and
                                   after[k] != before.get(k, 0)},
                    errors=[x["error"] for x in recs + [b] if x.get("error")])
    finally:
        _stop_server(proc)
        log.close()


def variations(args):
    os.makedirs(args.out_dir, exist_ok=True) if getattr(args, "out_dir", None) else None
    gguf = args.gguf if isinstance(args.gguf, list) else [args.gguf]
    arch = load_all(gguf)[0].get("general.architecture", "?")
    policy = _choose_policy(args, arch)
    ns = _plan_namespace(args, args.declared, policy,
                         json_out=args.plan_json)
    data, by, missing, tsum = compute_plan(ns)
    report = dict(tool="ud_map variations", policy=policy, plan=args.plan_json,
                  variations=data["variations"], kept=data["kept"],
                  containers=[], measurements=[])
    if data.get("mtp_variations"):
        print(f'MTP layer {data["mtp_layer"]} experts (map size)')
        print(f'  {"format":8s}{"layer GiB":>11s}{"total GiB":>11s}{"delta GiB":>11s}')
        for x in data["mtp_variations"]:
            print(f'  {x["mtp"]:8s}{x["mtp_bytes"]/2**30:11.3f}'
                  f'{x["total_bytes"]/2**30:11.3f}{x["delta_bytes"]/2**30:11.3f}')
    if data["variations"]:
        print(f'{"MTP":10s}{"vision":10s}{"total GiB":>11s}{"delta GiB":>11s}')
        base = next((x for x in data["variations"]
                     if x["mtp"] == "keep" and x["vision"] == "keep"), None)
        for x in data["variations"]:
            delta = (x["total_bytes"] - base["total_bytes"]) / 2**30 if base else 0.0
            print(f'{x["mtp"]:10s}{x["vision"]:10s}{x["total_bytes"]/2**30:11.3f}{delta:11.3f}')
    for spec in args.container or []:
        name, sep, path = spec.partition("=")
        if not sep:
            sys.exit("ud_map variations: --container wants NAME=PATH")
        rec = dict(name=name, model=path, file_bytes=os.path.getsize(path))
        report["containers"].append(rec)
        print(f'container {name:12s} {rec["file_bytes"]/2**30:8.3f} GiB  {path}')
    if args.measure:
        for spec in args.container or []:
            name, _, path = spec.partition("=")
            for rep in range(max(1, args.reps)):
                rec = _measure_one(args, name if args.reps == 1 else f"{name}-r{rep}", path)
                rec["rep"] = rep
                report["measurements"].append(rec)
                print(f'measure {name} r{rep}: ' + json.dumps(
                    {k: v for k, v in rec.items() if k not in ("metrics_delta", "model")}))
    if args.out_dir:
        if args.plan_json:
            with open(args.plan_json, "w") as f:
                json.dump(data, f, indent=1)
        out = os.path.join(args.out_dir, "variations.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
        print("variations: " + out)
    return 0

def _common(p, server=False):
    p.add_argument("--home", help="RADIANCE_HOME")
    p.add_argument("--kernels", help="RADIANCE_KERNELS")
    p.add_argument("--radiance", help="the radiance binary (default: $RADIANCE_RADIANCE or PATH)")
    p.add_argument("--env", action="append", default=[], metavar="K=V",
                   help="extra environment for the child")
    p.add_argument("--timeout", type=int, default=24 * 3600, help="child timeout, seconds")
    p.add_argument("--flock", help="lock file for GPU runs (default ~/radiance-test/gpu.lock if it exists)")
    p.add_argument("--flock-wait", type=int, help="flock -w seconds")
    p.add_argument("--no-flock", action="store_true", help="do not wrap a GPU run in flock")
    p.add_argument("--extra-arg", action="append", default=[], help="one more child argument")
    if server:
        p.add_argument("--tp", type=int, default=1)
        p.add_argument("--max-model-len", type=int, default=8192)
        p.add_argument("--kv-cache-dtype", default="fp8")
        p.add_argument("--num-speculative-tokens", type=int)
        p.add_argument("--max-num-seqs", type=int, default=4)
        p.add_argument("--api-key")
        p.add_argument("--port", type=int)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("plan", help="map a GGUF onto the bf16 source's declared weights")
    p.add_argument("gguf", nargs="+", help="the GGUF, all shards")
    p.add_argument("--declared", help="rad-convert --plan-only --plan-json output for the bf16 source")
    p.add_argument("--source", help="the bf16 source directory (recorded in the plan)")
    p.add_argument("--recipe-out")
    p.add_argument("--json")
    p.add_argument("--mtp", default="keep", help="comma list of keep,bf16,fp8,q8_0,w4nl64,w5nl64")
    p.add_argument("--vision", default="keep", help="comma list of keep,bf16,fp8,q8_0,w4nl64,w5nl64")
    p.add_argument("--policy", choices=["mirror", "fast", "size", "auto"], default="mirror")
    p.add_argument("--gptq", action="store_true", help="with --policy fast: the calibrated recipe")
    p.add_argument("--mtp-variant", choices=["keep", "bf16", "fp8", "q8_0", "w4nl64", "w5nl64"],
                   help="with --policy fast: the MTP layer's expert format")
    p.set_defaults(fn=plan)

    b = sp.add_parser("build", help="plan -> recipe -> rad-convert")
    b.add_argument("gguf", nargs="+")
    b.add_argument("--source", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--declared")
    b.add_argument("--policy", choices=["mirror", "fast", "size", "auto"], default="auto")
    b.add_argument("--source-quant", choices=["auto", "bf16", "quantised", "gguf", "unknown"],
                   default="auto")
    b.add_argument("--imatrix")
    b.add_argument("--reuse")
    b.add_argument("--set", action="append", default=[], metavar="K=V")
    b.add_argument("--auto-fp8", action=argparse.BooleanOptionalAction, default=True,
                   help="with a bf16 source, add --set quantization_config.quant_method=fp8")
    b.add_argument("--gptq", action="store_true")
    b.add_argument("--mtp", default="keep")
    b.add_argument("--vision", default="keep")
    b.add_argument("--mtp-variant", choices=["keep", "bf16", "fp8", "q8_0", "w4nl64", "w5nl64"])
    b.add_argument("--recipe-out")
    b.add_argument("--json", help="plan JSON path")
    b.add_argument("--report-out")
    b.add_argument("--log")
    b.add_argument("--plan-only", action="store_true",
                   help="run rad-convert --plan-only and write the declared plan; write nothing")
    b.add_argument("--dry-run", action="store_true", help="emit recipe and plan, print the command")
    b.add_argument("--verify", action="store_true", help="verify the container after the build")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--tol", type=float, default=0.20)
    b.add_argument("--draft-tol", type=float, default=0.75)
    b.add_argument("--skip-gguf", action="store_true")
    b.add_argument("--max-experts", type=int, default=0)
    b.add_argument("--rad-convert")
    _common(b)
    b.set_defaults(fn=build)

    c = sp.add_parser("calibrate", help="a calibration batch run, then the GPTQ conversion")
    c.add_argument("--model", required=True, help="the RTN container the tap serves")
    c.add_argument("--corpus")
    c.add_argument("--calib-dir", required=True)
    c.add_argument("--tokens", type=int, default=1500000)
    c.add_argument("--source", help="the bf16 source (GPTQ phase)")
    c.add_argument("--out", help="the GPTQ container (GPTQ phase)")
    c.add_argument("--gguf", nargs="+", help="the UD map for the GPTQ recipe")
    c.add_argument("--recipe")
    c.add_argument("--recipe-out")
    c.add_argument("--reuse")
    c.add_argument("--imatrix")
    c.add_argument("--set", action="append", default=[])
    c.add_argument("--auto-fp8", action=argparse.BooleanOptionalAction, default=True)
    c.add_argument("--policy", choices=["mirror", "fast", "size", "auto"], default="auto")
    c.add_argument("--mtp-variant", choices=["keep", "bf16", "fp8", "q8_0", "w4nl64", "w5nl64"])
    c.add_argument("--calibrate-only", action="store_true")
    c.add_argument("--convert-only", action="store_true")
    c.add_argument("--max-num-batched-tokens", type=int, default=8192)
    c.add_argument("--log")
    c.add_argument("--report-out")
    c.add_argument("--rad-convert")
    _common(c, server=True)
    c.set_defaults(fn=calibrate)

    v = sp.add_parser("verify", help="hashes, per-tensor RMSE, size table and policy verdict")
    v.add_argument("gguf_pos", nargs="*",
                   help="the GGUF shards (also accepted as --gguf, as the runbook writes it)")
    v.add_argument("--gguf", nargs="+")
    v.add_argument("--rad", required=True)
    v.add_argument("--source")
    v.add_argument("--declared")
    v.add_argument("--plan-json", dest="plan_json")
    v.add_argument("--out")
    v.add_argument("--only")
    v.add_argument("--workers", type=int, default=8)
    v.add_argument("--home")
    v.add_argument("--kernels")
    v.add_argument("--tol", type=float, default=0.20)
    v.add_argument("--draft-tol", type=float, default=0.75)
    v.add_argument("--ud-slack", dest="ud_slack", type=float, default=1.5,
                   help="a requantised weight may show up to max(--tol, slack x the UD file's own "
                        "relative error against the source)")
    v.add_argument("--policy", choices=["auto", "lossless", "quantised"], default="auto")
    v.add_argument("--skip-gguf", action="store_true")
    v.add_argument("--allow-missing", action="store_true")
    v.add_argument("--max-experts", type=int, default=0,
                   help="check at most N experts per (layer, projection); 0 checks every one")
    v.set_defaults(fn=verify)

    sc = sp.add_parser("score", help="the KL mode or the existing quality sets")
    sc.add_argument("--mode", choices=["kld", "quality"], default="kld")
    sc.add_argument("--reference", help="the reference .rad (kld)")
    sc.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    sc.add_argument("--corpus", help="the JSONL corpus (kld)")
    sc.add_argument("--ref-dir")
    sc.add_argument("--top", type=int, default=20)
    sc.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    sc.add_argument("--model", help="the .rad to serve (quality)")
    sc.add_argument("--evalset", help="the quality evalset JSONL (quality)")
    sc.add_argument("--out-dir", required=True)
    sc.add_argument("--restart", action="store_true")
    _common(sc, server=True)
    sc.set_defaults(fn=score)

    va = sp.add_parser("variations", help="the size table and, per container, decode/acceptance/TTFT")
    va.add_argument("gguf", nargs="+")
    va.add_argument("--declared")
    va.add_argument("--policy", choices=["mirror", "fast", "size", "auto"], default="auto")
    va.add_argument("--mtp", default="keep")
    va.add_argument("--vision", default="keep")
    va.add_argument("--mtp-variant", choices=["keep", "bf16", "fp8", "q8_0", "w4nl64", "w5nl64"])
    va.add_argument("--gptq", action="store_true")
    va.add_argument("--container", action="append", default=[], metavar="NAME=PATH")
    va.add_argument("--measure", action="store_true")
    va.add_argument("--reps", type=int, default=1)
    va.add_argument("--out-dir")
    va.add_argument("--plan-json")
    _common(va, server=True)
    va.set_defaults(fn=variations)

    a = ap.parse_args()
    rc = a.fn(a)
    sys.exit(0 if rc is None or rc is True else (1 if rc is False else rc))

if __name__ == "__main__":
    main()
