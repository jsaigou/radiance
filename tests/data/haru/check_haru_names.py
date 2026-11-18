#!/usr/bin/env python3
"""Check the Haru blocks' declared weights against the tiny and real checkpoints.

The four blocks in arch/haru_common/ declare their weights from generated names
(nm.f / nm.ckpt / plain_weight / declare_weight).  This script parses those
declarations from the sources, expands the per-layer and per-stage loops with
each checkpoint's config, and compares the resulting checkpoint-side names,
shapes and dtypes with a checkpoint header.  It reports declarations that are
absent from the checkpoint and checkpoint tensors that no block declares.

Only the Python standard library is used; tiny is read as a safetensors header
and the real model as a header.json.  Nothing here needs torch, safetensors or
a C++ build.

    python3 tests/data/haru/check_haru_names.py
    python3 tests/data/haru/check_haru_names.py --self-test

Exit status is 0 only when both manifests match and every parser count is
non-zero and agrees with the expected count.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCH = ROOT / "arch" / "haru_common"


class ParseError(Exception):
    pass


# ------------------------------------------------------------------ small C++ text helpers

def _match_delim(text: str, start: int) -> int:
    """Return the index of the delimiter matching text[start] ('(', '{' or '[')."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    opener = text[start]
    if opener not in pairs:
        raise ParseError(f"not a delimiter at {start}: {opener!r}")
    closer = pairs[opener]
    depth = 0
    i = start
    state = "code"  # code, line comment, block comment, string, char
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if c == '"':
                state = "string"
                i += 1
                continue
            if c == "'":
                state = "char"
                i += 1
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return i
        elif state == "line":
            if c == "\n":
                state = "code"
        elif state == "block":
            if c == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        elif state == "string":
            if c == "\\":
                i += 2
                continue
            if c == '"':
                state = "code"
        elif state == "char":
            if c == "\\":
                i += 2
                continue
            if c == "'":
                state = "code"
        i += 1
    raise ParseError("unbalanced delimiter")


def function_body(text: str, signature: str) -> str:
    """Extract the body of the first function whose signature matches a regex."""
    m = re.search(signature, text)
    if not m:
        raise ParseError(f"function not found: {signature}")
    open_brace = text.find("{", m.end())
    if open_brace < 0:
        raise ParseError(f"no body for {signature}")
    close_brace = _match_delim(text, open_brace)
    return text[open_brace + 1 : close_brace]


def iter_calls(text: str, name: str):
    """Yield every call text for `name(...)`; the text starts at '('."""
    pat = re.compile(r"(?<![\w])" + re.escape(name) + r"\s*\(")
    for m in pat.finditer(text):
        open_paren = text.find("(", m.start())
        close_paren = _match_delim(text, open_paren)
        yield text[open_paren : close_paren + 1]


def iter_named(text: str, name: str):
    """Yield every `name(...)` expression, function name included."""
    pat = re.compile(r"(?<![\w])" + re.escape(name) + r"\s*\(")
    for m in pat.finditer(text):
        open_paren = text.find("(", m.start())
        close_paren = _match_delim(text, open_paren)
        yield text[m.start() : close_paren + 1]


def split_args(call: str) -> list[str]:
    """Split one call's top-level comma-separated arguments."""
    if not call.startswith("(") or not call.endswith(")"):
        raise ParseError(f"not a call: {call[:80]}")
    inner = call[1:-1]
    args: list[str] = []
    start = 0
    depth_paren = depth_brace = depth_bracket = 0
    state = "code"
    i = 0
    while i < len(inner):
        c = inner[i]
        nxt = inner[i + 1] if i + 1 < len(inner) else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if c == '"':
                state = "string"
                i += 1
                continue
            if c == "'":
                state = "char"
                i += 1
                continue
            if c == "(":
                depth_paren += 1
            elif c == ")":
                depth_paren -= 1
            elif c == "{":
                depth_brace += 1
            elif c == "}":
                depth_brace -= 1
            elif c == "[":
                depth_bracket += 1
            elif c == "]":
                depth_bracket -= 1
            elif c == "," and depth_paren == depth_brace == depth_bracket == 0:
                args.append(inner[start:i].strip())
                start = i + 1
        elif state == "line":
            if c == "\n":
                state = "code"
        elif state == "block":
            if c == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        elif state == "string":
            if c == "\\":
                i += 2
                continue
            if c == '"':
                state = "code"
        elif state == "char":
            if c == "\\":
                i += 2
                continue
            if c == "'":
                state = "code"
        i += 1
    tail = inner[start:].strip()
    if tail:
        args.append(tail)
    return args


def string_literal(expr: str) -> str:
    m = re.search(r'"((?:\\.|[^"\\])*)"', expr)
    if not m:
        raise ParseError(f"not a string literal: {expr[:80]}")
    return m.group(1)


def parse_nm(expr: str) -> tuple[str, str, list[str]]:
    """Parse an nm.f(...) or nm.ckpt(...) expression into (kind, format, args)."""
    m = re.search(r"\bnm\.(f|ckpt)\s*\(", expr)
    if not m:
        raise ParseError(f"not an nm.f/nm.ckpt expression: {expr[:80]}")
    open_paren = expr.find("(", m.start())
    call = expr[open_paren : _match_delim(expr, open_paren) + 1]
    args = split_args(call)
    return m.group(1), string_literal(args[0]), args[1:]


def _replace_ints(fmt: str, value: int) -> str:
    return fmt.replace("%lld", str(value)).replace("%d", str(value))


def expand_f(fmt: str, layer: int | None) -> str:
    if "%" not in fmt:
        return fmt
    if layer is None:
        raise ParseError(f"layer format without a layer: {fmt!r}")
    out = _replace_ints(fmt, layer)
    if "%" in out:
        raise ParseError(f"unexpanded format specifier: {out!r}")
    return out


def expand_ckpt(fmt: str, layer: int | None, pre: str | None) -> str:
    out = fmt
    if "%s" in out:
        if pre is None:
            raise ParseError(f"checkpoint format needs a prefix: {fmt!r}")
        out = out.replace("%s", pre)
    if "%" in out:
        if layer is None:
            raise ParseError(f"layer format without a layer: {fmt!r}")
        out = _replace_ints(out, layer)
    if "%" in out:
        raise ParseError(f"unexpanded format specifier: {out!r}")
    return out


def expand_stage(fmt: str, stage: int, block: int) -> str:
    out = fmt.replace("%d", str(stage)).replace("%lld", str(block))
    if "%" in out:
        raise ParseError(f"unexpanded format specifier: {out!r}")
    return out


# ------------------------------------------------------------------ config and manifests

def config_dtype(cfg: dict) -> str:
    text = cfg.get("dtype", "float32")
    if text in ("bfloat16", "bf16"):
        return "bf16"
    if text in ("float32", "f32", "float"):
        return "f32"
    if text in ("float16", "f16"):
        return "f16"
    raise ParseError(f"unsupported config dtype {text!r}")


def header_dtype(text: str) -> str:
    text = text.upper()
    if text in ("BF16", "BFLOAT16"):
        return "bf16"
    if text in ("F32", "FLOAT32"):
        return "f32"
    if text in ("F16", "FLOAT16"):
        return "f16"
    raise ParseError(f"unsupported header dtype {text!r}")


def prod(shape) -> int:
    n = 1
    for v in shape:
        n *= int(v)
    return n


class Entry:
    __slots__ = ("shape", "dtype", "block", "note")

    def __init__(self, shape, dtype: str, block: str, note: str = ""):
        self.shape = tuple(int(v) for v in shape)
        self.dtype = dtype
        self.block = block
        self.note = note


class Manifest:
    def __init__(self):
        self.sources: dict[str, Entry] = {}
        self.logical: dict[str, set[str]] = {}
        self.source_names: dict[str, set[str]] = {}
        self.virtual: list[tuple[str, str, tuple[int, ...], str]] = []
        self.notes: list[str] = []

    def add_logical(self, block: str, name: str):
        names = self.logical.setdefault(block, set())
        if name in names:
            raise ParseError(f"logical weight declared twice: {name}")
        names.add(name)

    def add_source(self, block: str, name: str, shape, dtype: str, note: str = ""):
        entry = Entry(shape, dtype, block, note)
        if name in self.sources:
            old = self.sources[name]
            if old.shape != entry.shape or old.dtype != entry.dtype:
                raise ParseError(
                    f"source tensor {name} declared twice with different shape/dtype: "
                    f"{old.shape}/{old.dtype} vs {entry.shape}/{entry.dtype}")
            return
        self.sources[name] = entry
        self.source_names.setdefault(block, set()).add(name)

    def identity(self, block: str, name: str, shape, dtype: str, note: str = ""):
        self.add_logical(block, name)
        self.add_source(block, name, shape, dtype, note)


def lm_dims(cfg: dict) -> dict:
    lm = cfg["lm"]
    head_dim = int(lm["head_dim"])
    n_head = int(lm["num_attention_heads"])
    n_kv = int(lm["num_key_value_heads"])
    n_kh = int(lm["linear_num_key_heads"])
    n_vh = int(lm["linear_num_value_heads"])
    head_k = int(lm["linear_key_head_dim"])
    head_v = int(lm["linear_value_head_dim"])
    k_dim = n_kh * head_k
    v_dim = n_vh * head_v
    conv_dim = 2 * k_dim + v_dim
    return {
        "V": int(lm["vocab_size"]),
        "H": int(lm["hidden_size"]),
        "I": int(lm["intermediate_size"]),
        "n_layers": int(lm["num_hidden_layers"]),
        "head_dim": head_dim,
        "q_dim": n_head * head_dim,
        "kv_dim": n_kv * head_dim,
        "n_vh": n_vh,
        "head_v": head_v,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "conv_dim": conv_dim,
        "in_dim": conv_dim + 2 * n_vh,
        "conv_k": int(lm["linear_conv_kernel_dim"]),
    }


def lm_source_shape(name: str, d: dict) -> tuple[int, ...]:
    if name == "lm.embed_tokens.weight":
        return (d["V"], d["H"])
    if name == "lm.norm.weight":
        return (d["H"],)
    if re.fullmatch(r"lm\.layers\.\d+\.input_layernorm\.weight", name) or \
       re.fullmatch(r"lm\.layers\.\d+\.post_attention_layernorm\.weight", name):
        return (d["H"],)
    if re.fullmatch(r"lm\.layers\.\d+\.self_attn\.q_proj\.weight", name):
        return (2 * d["q_dim"], d["H"])
    if re.fullmatch(r"lm\.layers\.\d+\.self_attn\.[kv]_proj\.weight", name):
        return (d["kv_dim"], d["H"])
    if re.fullmatch(r"lm\.layers\.\d+\.self_attn\.o_proj\.weight", name):
        return (d["H"], d["q_dim"])
    if re.fullmatch(r"lm\.layers\.\d+\.self_attn\.[qk]_norm\.weight", name):
        return (d["head_dim"],)
    if re.fullmatch(r"lm\.layers\.\d+\.linear_attn\.in_proj_qkv\.weight", name):
        return (d["conv_dim"], d["H"])
    if re.fullmatch(r"lm\.layers\.\d+\.linear_attn\.in_proj_[ab]\.weight", name):
        return (d["n_vh"], d["H"])
    if re.fullmatch(r"lm\.layers\.\d+\.linear_attn\.in_proj_z\.weight", name):
        return (d["v_dim"], d["H"])
    if re.fullmatch(r"lm\.layers\.\d+\.linear_attn\.conv1d\.weight", name):
        return (d["conv_dim"], 1, d["conv_k"])
    if re.fullmatch(r"lm\.layers\.\d+\.linear_attn\.(A_log|dt_bias)", name):
        return (d["n_vh"],)
    if re.fullmatch(r"lm\.layers\.\d+\.linear_attn\.norm\.weight", name):
        return (d["head_v"],)
    if re.fullmatch(r"lm\.layers\.\d+\.linear_attn\.out_proj\.weight", name):
        return (d["H"], d["v_dim"])
    if re.fullmatch(r"lm\.layers\.\d+\.mlp\.(gate_proj|up_proj)\.weight", name):
        return (d["I"], d["H"])
    if re.fullmatch(r"lm\.layers\.\d+\.mlp\.down_proj\.weight", name):
        return (d["H"], d["I"])
    raise ParseError(f"no LM shape formula for {name}")


def lm_layer_types(cfg: dict) -> list[str]:
    lm = cfg["lm"]
    n = int(lm["num_hidden_layers"])
    types = lm.get("layer_types")
    if types is None:
        interval = int(lm.get("full_attention_interval", 0))
        if interval <= 0:
            raise ParseError("lm.layer_types missing and no full_attention_interval")
        types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                 for i in range(n)]
    if len(types) != n:
        raise ParseError(f"layer_types has {len(types)} entries, num_hidden_layers is {n}")
    for t in types:
        if t not in ("full_attention", "linear_attention"):
            raise ParseError(f"unknown layer type {t!r}")
    return list(types)


def resolve_lm_dtype(token: str, cfg: dict) -> str:
    token = token.strip()
    if token == "wd":
        return config_dtype(cfg)
    if token == "RAD_F32":
        return "f32"
    if token == "RAD_BF16":
        return "bf16"
    raise ParseError(f"unknown LM declaration dtype {token!r}")


def parse_lm_section(body: str, layers: list[int], pre: str | None,
                     cfg: dict, manifest: Manifest, block: str, log: list[str]):
    """Parse one LM section into maps/logical names and add its source tensors."""
    maps: dict[str, list[str]] = {}
    virtual: dict[str, str] = {}

    def add_map(lname: str, sources: list[str], is_virtual: bool = False):
        if lname in maps:
            raise ParseError(f"{block}: {lname} mapped twice")
        maps[lname] = sources
        if is_virtual:
            virtual[lname] = sources[0]

    for call in iter_calls(body, "map_copy"):
        args = split_args(call)
        _, ffmt, _ = parse_nm(args[1])
        _, cfmt, _ = parse_nm(args[2])
        for l in layers:
            add_map(expand_f(ffmt, l), [expand_ckpt(cfmt, l, pre)])

    for call in iter_calls(body, "map_concat"):
        args = split_args(call)
        _, ffmt, _ = parse_nm(args[1])
        comps: list[str] = []
        for comp in iter_named(args[2], "nm.ckpt"):
            _, cfmt, _ = parse_nm(comp)
            comps.append(cfmt)
        if not comps:
            raise ParseError(f"{block}: map_concat with no nm.ckpt components")
        for l in layers:
            add_map(expand_f(ffmt, l), [expand_ckpt(c, l, pre) for c in comps])

    for call in iter_calls(body, "map_rows"):
        args = split_args(call)
        _, ffmt, _ = parse_nm(args[1])
        comps = []
        for comp in iter_named(args[2], "nm.ckpt"):
            _, cfmt, _ = parse_nm(comp)
            comps.append(cfmt)
        if not comps:
            # The source may be a local alias such as
            # `const char* emb = nm.ckpt("lm.embed_tokens.weight");`.
            aliases: dict[str, str] = {}
            for m in re.finditer(r"const\s+char\s*\*\s*(\w+)\s*=\s*", body):
                tail = body[m.end():]
                if tail.lstrip().startswith("nm.ckpt"):
                    _, cfmt, _ = parse_nm(tail)
                    aliases[m.group(1)] = cfmt
            for ident in re.findall(r"\b[A-Za-z_]\w*\b", args[2]):
                if ident in aliases and aliases[ident] not in comps:
                    comps.append(aliases[ident])
        if len(comps) != 1:
            raise ParseError(f"{block}: map_rows source is not one tensor ({comps})")
        for l in layers:
            add_map(expand_f(ffmt, l), [expand_ckpt(comps[0], l, pre)], is_virtual=True)

    decls: dict[str, str] = {}
    for call in iter_calls(body, "decl_w"):
        args = split_args(call)
        _, ffmt, _ = parse_nm(args[1])
        dtype = resolve_lm_dtype(args[2], cfg)
        for l in layers:
            lname = expand_f(ffmt, l)
            if lname in decls:
                raise ParseError(f"{block}: {lname} declared twice")
            decls[lname] = dtype

    if set(maps) != set(decls):
        missing = sorted(set(maps) - set(decls))
        extra = sorted(set(decls) - set(maps))
        raise ParseError(f"{block}: map/declaration mismatch; mapped-only {missing}, "
                         f"declared-only {extra}")
    if len(decls) == 0:
        raise ParseError(f"{block}: the parser found zero declarations")
    log.append(f"{block}: {len(decls)} declarations from {len(layers)} layer(s)")

    d = lm_dims(cfg)
    for lname in sorted(maps):
        manifest.add_logical("lm", lname)
        if lname not in virtual:
            dtype = decls[lname]
            # A map_concat declaration is one logical weight with several source tensors.
            for src in maps[lname]:
                manifest.add_source("lm", src, lm_source_shape(src, d), dtype)
    for lname in sorted(virtual):
        src = virtual[lname]
        if src not in manifest.sources:
            raise ParseError(f"{lname}: virtual source {src} has no declaration")
        manifest.virtual.append((lname, src, (2, d["H"]), decls[lname]))


def build_lm(cfg: dict, src: str, manifest: Manifest, log: list[str]):
    main_body = function_body(src, r"int\s+HaruLm::declare\s*\(")
    types = lm_layer_types(cfg)
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    linear = [i for i, t in enumerate(types) if t == "linear_attention"]

    parse_lm_section(main_body, [None], None, cfg, manifest, "lm:main", log)

    for signature, layers, label in (
        (r"int\s+HaruLm::declare_attn\s*\(", full, "lm:attn"),
        (r"int\s+HaruLm::declare_gdn\s*\(", linear, "lm:gdn"),
        (r"int\s+HaruLm::declare_mlp\s*\(", list(range(len(types))), "lm:mlp"),
    ):
        body = function_body(src, signature)
        m = re.search(r'const\s+char\s*\*\s*pre\s*=\s*"([^"]*)"', body)
        pre = m.group(1) if m else None
        if not layers:
            raise ParseError(f"{label}: no layers match the config's schedule")
        parse_lm_section(body, layers, pre, cfg, manifest, label, log)

    d = lm_dims(cfg)
    expected = 3 + 4 * len(types) + 6 * len(full) + 7 * len(linear)
    got = len(manifest.logical["lm"])
    if got != expected:
        raise ParseError(f"lm: {got} logical declarations, expected {expected}")
    log.append(f"lm: {got} logical declarations, {len(manifest.source_names['lm'])} source "
               f"tensors; {d['n_layers']} layers ({len(full)} full, {len(linear)} linear)")


def build_head(cfg: dict, src: str, manifest: Manifest, log: list[str]):
    body = function_body(src, r"int\s+HaruHead::declare\s*\(")
    h = cfg["head"]
    H = int(h["hidden_size"])
    D = int(h["latent_dim"])
    F = int(H * float(h["ffn_ratio"]))
    layers = int(h["num_layers"])
    freq = 256  # kHaruScalarFreqs in haru_head.h

    names: list[str] = []
    n_loop_calls = 0
    for call in iter_calls(body, "plain_weight"):
        args = split_args(call)
        if re.fullmatch(r'"[^"]+"', args[1].strip()):
            names.append(string_literal(args[1]))
        else:
            n_loop_calls += 1

    templates = [m.group(1) for m in re.finditer(
        r'snprintf\s*\(\s*n\s*,\s*sizeof\s+n\s*,\s*"((?:\\.|[^"\\])*)"', body)]
    if len(templates) != n_loop_calls:
        raise ParseError(f"head: {len(templates)} snprintf templates for {n_loop_calls} "
                         "loop plain_weight calls")
    for fmt in templates:
        for l in range(layers):
            names.append(expand_f(fmt, l))

    def shape(name: str) -> tuple[int, ...]:
        if name == "head.cond_proj.weight":
            return (H, H)
        if name == "head.latent_proj.weight":
            return (H, D)
        if re.fullmatch(r"head\.(t|dt|h|omega)_embed\.fc1\.weight", name):
            return (H, freq)
        if re.fullmatch(r"head\.(t|dt|h|omega)_embed\.fc2\.weight", name):
            return (H, H)
        if re.fullmatch(r"head\.layers\.\d+\.norm\.weight", name):
            return (H,)
        if re.fullmatch(r"head\.layers\.\d+\.ada\.weight", name):
            return (3 * H, H)
        if re.fullmatch(r"head\.layers\.\d+\.ffn\.(gate_proj|up_proj)\.weight", name):
            return (F, H)
        if re.fullmatch(r"head\.layers\.\d+\.ffn\.down_proj\.weight", name):
            return (H, F)
        if name == "head.final.ada.weight":
            return (2 * H, H)
        if name == "head.final.proj.weight":
            return (D, H)
        raise ParseError(f"no head shape formula for {name}")

    if len(names) != len(set(names)):
        raise ParseError("head: a name is declared twice")
    expected = 12 + 5 * layers
    if len(names) != expected:
        raise ParseError(f"head: {len(names)} declarations, expected {expected}")
    for name in names:
        manifest.identity("head", name, shape(name), "f32")
    log.append(f"head: {len(names)} declarations ({len(names) - 5 * layers} literals + "
               f"5 x {layers} layers)")


def build_proj(cfg: dict, src: str, manifest: Manifest, log: list[str]):
    body = function_body(src, r"int\s+HaruProj::declare\s*\(")
    lm = cfg["lm"]
    H = int(lm["hidden_size"])
    D = int(cfg["latent_dim"])
    names: list[str] = []
    n_loop_calls = 0
    for call in iter_calls(body, "plain_weight"):
        args = split_args(call)
        if re.fullmatch(r'"[^"]+"', args[1].strip()):
            names.append(string_literal(args[1]))
        else:
            n_loop_calls += 1
    templates = [m.group(1) for m in re.finditer(
        r'snprintf\s*\(\s*n\s*,\s*sizeof\s+n\s*,\s*"((?:\\.|[^"\\])*)"', body)]
    prefixes = [m.group(1) for m in re.finditer(r'audio\s*\(\s*"([^"]+)"', body)]
    if len(templates) != n_loop_calls:
        raise ParseError(f"proj: {len(templates)} snprintf templates for {n_loop_calls} "
                         "loop plain_weight calls")
    if len(prefixes) != 2:
        raise ParseError(f"proj: expected 2 audio prefixes, found {len(prefixes)}")
    for fmt in templates:
        if "%s" not in fmt:
            raise ParseError(f"proj: template without %s: {fmt!r}")
        for prefix in prefixes:
            names.append(fmt.replace("%s", prefix))

    def shape(name: str) -> tuple[int, ...]:
        if name in ("latent_scale", "latent_bias"):
            return (1,)
        if re.fullmatch(r"(model|user)_audio_proj\.fc1\.weight", name):
            return (H, D)
        if re.fullmatch(r"(model|user)_audio_proj\.fc1\.bias", name):
            return (H,)
        if re.fullmatch(r"(model|user)_audio_proj\.norm\.weight", name):
            return (H,)
        if re.fullmatch(r"(model|user)_audio_proj\.fc2\.weight", name):
            return (H, H)
        if re.fullmatch(r"(model|user)_audio_proj\.fc2\.bias", name):
            return (H,)
        raise ParseError(f"no projector shape formula for {name}")

    if len(names) != len(set(names)):
        raise ParseError("proj: a name is declared twice")
    expected = 2 + 5 * len(prefixes)
    if len(names) != expected:
        raise ParseError(f"proj: {len(names)} declarations, expected {expected}")
    for name in names:
        manifest.identity("proj", name, shape(name), "f32")
    log.append(f"proj: {len(names)} declarations (2 scalars + {len(prefixes)} x {len(templates)} "
               "projector tensors)")


def _codec_block_suffixes(block_body: str) -> list[str]:
    out: list[str] = []
    for call in iter_calls(block_body, "declare_weight"):
        args = split_args(call)
        kind, fmt, _ = parse_nm(args[1])
        if kind != "f" or "%s" not in fmt:
            raise ParseError(f"codec block: bad weight name template {fmt!r}")
        out.append(fmt)
    if len(out) == 0:
        raise ParseError("codec block: the parser found zero declarations")
    return out


def _codec_conv_suffixes(conv_body: str) -> list[str]:
    aliases: dict[str, str] = {}
    for m in re.finditer(r"const\s+char\s*\*\s*(\w+)\s*=\s*", conv_body):
        tail = conv_body[m.end():]
        if tail.lstrip().startswith("nm.f"):
            _, fmt, _ = parse_nm(tail)
            aliases[m.group(1)] = fmt
    out: list[str] = []
    for call in iter_calls(conv_body, "declare_weight"):
        args = split_args(call)
        expr = args[1].strip()
        if re.fullmatch(r"[A-Za-z_]\w*", expr):
            if expr not in aliases:
                raise ParseError(f"codec conv: unknown name alias {expr!r}")
            fmt = aliases[expr]
        else:
            kind, fmt, _ = parse_nm(expr)
            if kind != "f":
                raise ParseError(f"codec conv: bad name expression {expr!r}")
        if "%s" not in fmt:
            raise ParseError(f"codec conv: bad weight name template {fmt!r}")
        if fmt not in out:
            out.append(fmt)
    if len(out) != 2:
        raise ParseError(f"codec conv: expected weight+bias templates, found {len(out)}")
    return out


def build_codec(cfg: dict, src: str, manifest: Manifest, log: list[str]):
    c = cfg["codec"]
    base = int(c["base_channels"])
    latent = int(cfg["latent_dim"])
    ratios = [int(v) for v in c["ratios"]]
    enc_depths = [int(v) for v in c["encoder_depths"]]
    dec_depths = [int(v) for v in c["decoder_depths"]]
    dtype = config_dtype(cfg)
    if len(ratios) != 6 or len(enc_depths) != 7 or len(dec_depths) != 7:
        raise ParseError("codec: ratios/depths have the wrong length")

    conv_body = function_body(src, r"int\s+declare_conv\s*\(")
    block_body = function_body(src, r"int\s+declare_block\s*\(")
    body = function_body(src, r"int\s+codec_declare\s*\(")

    conv_suffixes = _codec_conv_suffixes(conv_body)
    block_suffixes = _codec_block_suffixes(block_body)

    conv_calls = list(iter_calls(body, "declare_conv"))
    block_calls = list(iter_calls(body, "declare_block"))
    if len(conv_calls) != 6:
        raise ParseError(f"codec: {len(conv_calls)} declare_conv call sites, expected 6")
    if len(block_calls) != 2:
        raise ParseError(f"codec: {len(block_calls)} declare_block call sites, expected 2")
    literal_prefixes: list[str] = []
    for call in conv_calls:
        args = split_args(call)
        if re.fullmatch(r'"[^"]+"', args[3].strip()):
            literal_prefixes.append(string_literal(args[3]))

    formats = [m.group(1) for m in re.finditer(r'"((?:\\.|[^"\\])*)"', body)]
    conv_formats: list[str] = []
    stage_formats: list[str] = []
    for fmt in formats:
        if re.fullmatch(r"codec\.encoder\.down\.%d", fmt) or \
           re.fullmatch(r"codec\.decoder\.up\.%d", fmt):
            conv_formats.append(fmt)
        elif re.fullmatch(r"codec\.(encoder|decoder)\.stages\.%d\.%lld", fmt):
            stage_formats.append(fmt)
    if len(conv_formats) != 2:
        raise ParseError(f"codec: expected encoder.down/decoder.up formats, found {conv_formats}")
    if len(stage_formats) != 2:
        raise ParseError(f"codec: expected encoder/decoder stage formats, found {stage_formats}")

    conv_prefixes = list(literal_prefixes)
    for fmt in conv_formats:
        direction = "encoder" if "encoder" in fmt else "decoder"
        for stage in range(6):
            conv_prefixes.append(fmt.replace("%d", str(stage)))
            # Keep the direction for the shape lookup through the name itself.

    stage_prefixes: list[str] = []
    for fmt in stage_formats:
        direction = "encoder" if "encoder" in fmt else "decoder"
        depths = enc_depths if direction == "encoder" else dec_depths
        for stage in range(7):
            for j in range(depths[stage]):
                stage_prefixes.append(expand_stage(fmt, stage, j))

    if len(conv_prefixes) != 16:
        raise ParseError(f"codec: {len(conv_prefixes)} conv layers, expected 16")
    if len(set(conv_prefixes)) != 16:
        raise ParseError("codec: a conv prefix is declared twice")

    def conv_shape(prefix: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if prefix == "codec.encoder.stem":
            return (base, 1, 7), (base,)
        if prefix == "codec.encoder.head":
            return (latent, base << 6, 7), (latent,)
        if prefix == "codec.decoder.stem":
            return (base << 6, latent, 7), (base << 6,)
        if prefix == "codec.decoder.head":
            return (1, base, 7), (1,)
        m = re.fullmatch(r"codec\.encoder\.down\.(\d+)", prefix)
        if m:
            s = int(m.group(1))
            cout = base << (s + 1)
            return (cout, base << s, 2 * ratios[5 - s]), (cout,)
        m = re.fullmatch(r"codec\.decoder\.up\.(\d+)", prefix)
        if m:
            s = int(m.group(1))
            cin = base << (6 - s)
            cout = base << (5 - s)
            return (cin, cout, 2 * ratios[s]), (cout,)
        raise ParseError(f"no codec conv shape formula for {prefix}")

    def block_shape(prefix: str) -> dict[str, tuple[int, ...]]:
        m = re.fullmatch(r"codec\.(encoder|decoder)\.stages\.(\d+)\.(\d+)", prefix)
        if not m:
            raise ParseError(f"no codec block shape formula for {prefix}")
        direction, stage = m.group(1), int(m.group(2))
        dim = base << stage if direction == "encoder" else base << (6 - stage)
        return {
            ".norm.weight": (dim,),
            ".conv.weight": (dim, 1, 7),
            ".conv.bias": (dim,),
            ".gamma": (dim,),
            ".ffn_norm.weight": (dim,),
            ".fc1.weight": (4 * dim, dim),
            ".fc1.bias": (4 * dim,),
            ".fc2.weight": (dim, 4 * dim),
            ".fc2.bias": (dim,),
            ".ffn_gamma": (dim,),
        }

    for prefix in conv_prefixes:
        wshape, bshape = conv_shape(prefix)
        manifest.identity("codec", prefix + ".weight", wshape, dtype)
        manifest.identity("codec", prefix + ".bias", bshape, dtype)

    for prefix in stage_prefixes:
        shapes = block_shape(prefix)
        for suffix_fmt in block_suffixes:
            suffix = suffix_fmt.replace("%s", "")
            name = prefix + suffix
            if suffix not in shapes:
                raise ParseError(f"codec: no shape for block suffix {suffix!r}")
            manifest.identity("codec", name, shapes[suffix], dtype)

    expected = 10 * (sum(enc_depths) + sum(dec_depths)) + 32
    got = len(manifest.logical["codec"])
    if got != expected:
        raise ParseError(f"codec: {got} declarations, expected {expected}")
    log.append(f"codec: {got} declarations; {len(conv_prefixes)} conv layers, "
               f"{len(stage_prefixes)} blocks; base {base}")


def build_manifest(cfg: dict, arch: Path) -> Manifest:
    log: list[str] = []
    manifest = Manifest()
    build_lm(cfg, (arch / "haru_lm.cpp").read_text(), manifest, log)
    build_head(cfg, (arch / "haru_head.cpp").read_text(), manifest, log)
    build_proj(cfg, (arch / "haru_proj.cpp").read_text(), manifest, log)
    build_codec(cfg, (arch / "haru_codec.cpp").read_text(), manifest, log)
    manifest.notes.extend(log)
    return manifest


# ------------------------------------------------------------------ loading and checking

def load_safetensors_header(path: Path) -> dict[str, dict]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ParseError(f"{path}: too short for a safetensors header")
        n = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(n))
    return {k: v for k, v in header.items() if not k.startswith("__")}


def load_header(path: Path) -> dict[str, dict]:
    if path.suffix == ".safetensors":
        return load_safetensors_header(path)
    with path.open() as f:
        header = json.load(f)
    return {k: v for k, v in header.items() if not k.startswith("__")}


def _fmt_list(items: list[str], limit: int = 12) -> str:
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" ... ({len(items) - limit} more)"


def check_one(label: str, checkpoint: dict[str, dict], cfg: dict, manifest: Manifest,
              verbose: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []
    expected = set(manifest.sources)
    actual = set(checkpoint)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    for name in missing:
        errors.append(f"{label}: declared-but-absent {name}")
    for name in extra:
        errors.append(f"{label}: present-but-undeclared {name}")

    widened: list[str] = []
    reshaped: list[str] = []
    for name in sorted(expected & actual):
        entry = manifest.sources[name]
        hdr = checkpoint[name]
        hshape = tuple(int(v) for v in hdr["shape"])
        if hshape != entry.shape:
            if prod(hshape) == prod(entry.shape):
                reshaped.append(f"{name} {entry.shape} -> {hshape}")
            else:
                errors.append(f"{label}: {name} shape declared {entry.shape}, header {hshape}")
        hdtype = header_dtype(hdr["dtype"])
        if hdtype == entry.dtype:
            pass
        elif hdtype in ("bf16", "f16") and entry.dtype == "f32":
            widened.append(name)
        else:
            errors.append(f"{label}: {name} declared {entry.dtype}, header {hdtype}")

    # The virtual pad/eop weight is a row view of the tied embedding table.
    for lname, src, shape, dtype in manifest.virtual:
        if src not in checkpoint:
            errors.append(f"{label}: {lname} virtual source {src} is absent")
        elif src not in expected:
            errors.append(f"{label}: {lname} virtual source {src} is undeclared")
        if verbose:
            notes.append(f"{label}: {lname} is a {shape} {dtype} view of {src}")

    if verbose or errors:
        nlogical = sum(len(v) for v in manifest.logical.values())
        notes.append(f"{label}: {nlogical} logical / {len(expected)} source tensors; "
                     f"header {len(actual)}; {len(widened)} BF16/F16 -> F32 widenings")
        for block in ("lm", "head", "proj", "codec"):
            nlog = len(manifest.logical.get(block, ()))
            nsrc = len(manifest.source_names.get(block, ()))
            notes.append(f"{label}: {block}: {nlog} declared, {nsrc} source tensors")
        if reshaped:
            notes.append(f"{label}: same-element-count reshapes: {_fmt_list(reshaped)}")
    return errors, notes


def run_checks(paths: dict[str, Path], arch: Path, verbose: bool) -> list[str]:
    errors: list[str] = []
    notes: list[str] = []
    for label, (cfg_path, hdr_path) in paths.items():
        with cfg_path.open() as f:
            cfg = json.load(f)
        manifest = build_manifest(cfg, arch)
        notes.extend(f"{label}: {line}" for line in manifest.notes)
        checkpoint = load_header(hdr_path)
        e, n = check_one(label, checkpoint, cfg, manifest, verbose)
        errors.extend(e)
        notes.extend(n)
        counts = ", ".join(
            f"{block} {len(manifest.logical.get(block, ()))} decl/"
            f"{len(manifest.source_names.get(block, ()))} src"
            for block in ("lm", "head", "proj", "codec"))
        nlogical = sum(len(v) for v in manifest.logical.values())
        print(f"haru names: {label}: {counts}; total {nlogical} declared / "
              f"{len(manifest.sources)} source tensors, header {len(checkpoint)}")

    if errors:
        for line in errors:
            print(f"FAIL {line}")
    else:
        print("PASS all declared names, shapes and dtypes match the checkpoint headers")
    if verbose or errors:
        for line in notes:
            print(f"NOTE {line}")
    return errors


# ------------------------------------------------------------------ self-test

def self_test(paths: dict[str, Path], arch: Path) -> int:
    label = "real" if "real" in paths else next(iter(paths))
    cfg_path, hdr_path = paths[label]
    with cfg_path.open() as f:
        cfg = json.load(f)
    manifest = build_manifest(cfg, arch)
    checkpoint = load_header(hdr_path)

    base_errors, _ = check_one(label, checkpoint, cfg, manifest, False)
    if base_errors:
        print(f"SELFTEST FAIL the unmodified {label} header does not pass:")
        for line in base_errors[:8]:
            print(f"  {line}")
        return 1

    name = next(iter(manifest.sources))
    # 1. A renamed tensor must be reported both ways.
    renamed = deepcopy(checkpoint)
    renamed["selftest.missing.tensor"] = renamed.pop(name)
    errors, _ = check_one(label, renamed, cfg, manifest, False)
    if not any("declared-but-absent " + name in e for e in errors) or \
       not any("present-but-undeclared selftest.missing.tensor" in e for e in errors):
        print("SELFTEST FAIL a renamed tensor was not reported")
        return 1
    print(f"SELFTEST PASS renamed {name} reported as missing and undeclared")

    # 2. A changed shape must be reported.
    resized = deepcopy(checkpoint)
    shape = list(resized[name].get("shape", []))
    if shape:
        shape[0] = int(shape[0]) + 1
    else:
        shape = [2]
    resized[name]["shape"] = shape
    errors, _ = check_one(label, resized, cfg, manifest, False)
    if not any(name in e and "shape" in e for e in errors):
        print("SELFTEST FAIL a changed shape was not reported")
        return 1
    print(f"SELFTEST PASS changed shape of {name} reported")

    # 3. A dtype that the declaration cannot hold must be reported.  Use a
    #    bf16-declared tensor: changing it to F32 is a narrowing the block does
    #    not declare (the real header is bf16 throughout).
    dtype_name = None
    for candidate in manifest.sources:
        if manifest.sources[candidate].dtype == "bf16" and \
           header_dtype(checkpoint[candidate]["dtype"]) == "bf16":
            dtype_name = candidate
            break
    if dtype_name is not None:
        retyped = deepcopy(checkpoint)
        retyped[dtype_name]["dtype"] = "F32"
        errors, _ = check_one(label, retyped, cfg, manifest, False)
        if not any(dtype_name in e and "declared" in e for e in errors):
            print("SELFTEST FAIL a changed dtype was not reported")
            return 1
        print(f"SELFTEST PASS changed dtype of {dtype_name} reported")

    return 0


# ------------------------------------------------------------------ command line

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="mutate a header in memory and verify the checker reports it")
    ap.add_argument("--tiny-dir", type=Path, default=HERE / "haru_tiny")
    ap.add_argument("--real-dir", type=Path, default=HERE / "real")
    ap.add_argument("--arch-dir", type=Path, default=ARCH)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    paths: dict[str, tuple[Path, Path]] = {}
    if args.tiny_dir:
        tiny_cfg = args.tiny_dir / "config.json"
        tiny_hdr = args.tiny_dir / "model.safetensors"
        if tiny_cfg.exists() and tiny_hdr.exists():
            paths["tiny"] = (tiny_cfg, tiny_hdr)
    if args.real_dir:
        real_cfg = args.real_dir / "config.json"
        real_hdr = args.real_dir / "header.json"
        if real_cfg.exists() and real_hdr.exists():
            paths["real"] = (real_cfg, real_hdr)
    if not paths:
        print("haru names: ERROR: no checkpoint config/header found", file=sys.stderr)
        return 1
    if "real" not in paths:
        print("haru names: SKIP real header is absent (tests/data/haru/real/)")

    try:
        if args.self_test:
            return self_test(paths, args.arch_dir)
        errors = run_checks(paths, args.arch_dir, args.verbose)
        return 1 if errors else 0
    except ParseError as e:
        print(f"haru names: ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
