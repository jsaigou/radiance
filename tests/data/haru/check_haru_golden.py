#!/usr/bin/env python3
"""Verify the committed Haru tiny checkpoint and golden exchange.

Four independent checks, each printing a PASS/FAIL line; the process exits
non-zero if any of them fails:

  A. regeneration is byte-identical to the committed files
  B. the generator's loop matches the real DuplexSession tokens and audio
  C. the tiny checkpoint's tensor-name patterns equal the real header
  D. the recorded tokens and random draws replay through the reference
     components within the measured thread-count noise floor

Run it with the pinned reference importable::

    cd ~/haru-golden
    PYTHONPATH=Haru/haru venv/bin/python \
        <worktree>/tests/data/haru/check_haru_golden.py

The golden files are those under this script's directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE_PACKAGE = Path(
    os.environ.get("HARU_REFERENCE_ROOT", Path.home() / "haru-golden" / "Haru" / "haru")
).expanduser()
REAL_DIR = Path(os.environ.get("HARU_REAL_DIR", Path.home() / "haru-golden" / "real")).expanduser()

sys.path.insert(0, str(HERE))
if REFERENCE_PACKAGE.is_dir():
    sys.path.insert(0, str(REFERENCE_PACKAGE))

import torch
from safetensors.torch import load_file

import gen_haru_tiny as gen

FAILURES: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- A

def check_regeneration() -> None:
    names = [
        "haru_tiny/config.json",
        "haru_tiny/model.safetensors",
        "haru_tiny/tokenizer.json",
        "golden.safetensors",
        "golden_meta.json",
    ]
    with tempfile.TemporaryDirectory(prefix="haru-golden-") as tmp:
        gen.generate(Path(tmp), threads=1)
        for name in names:
            committed = HERE / name
            fresh = Path(tmp) / name
            if not committed.is_file():
                report(f"A byte-identical {name}", False, "committed file missing")
                continue
            if not fresh.is_file():
                report(f"A byte-identical {name}", False, "regenerated file missing")
                continue
            a, b = sha256(committed), sha256(fresh)
            report(f"A byte-identical {name}", a == b, f"{a[:16]} vs {b[:16]}")
    total = sum(
        p.stat().st_size
        for p in HERE.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    report("A size under 4 MB", total < 4_000_000, f"{total} bytes (excluding __pycache__)")


# --------------------------------------------------------------------------- B

def check_faithful_loop(model, tokenizer, golden: dict[str, torch.Tensor]) -> None:
    from haru.session import DuplexConfig, DuplexSession

    reference = gen.reference_audio()
    user_frames = gen.user_audio_frames()
    cfg = DuplexConfig(
        guidance_scale=gen.GUIDANCE_SCALE,
        temperature=gen.TEMPERATURE,
        top_k=gen.TOP_K,
        top_p=gen.TOP_P,
        eop_bias=gen.EOP_BIAS,
        speak_first=False,
        seed=gen.SESSION_SEED,
        codec_backend="torch",
    )
    session = DuplexSession(model, system_prompt=gen.SYSTEM_PROMPT, reference=reference, config=cfg)
    frames = [session.start()]
    for audio in user_frames:
        frames.append(session.step(audio))

    token_ok = True
    audio_diff = 0.0
    for t, frame in enumerate(frames):
        expected_token = int(golden[f"frame{t}.token_out"].reshape(-1)[0])
        if int(frame.token) != expected_token:
            token_ok = False
            print(f"    frame {t}: token {frame.token} != recorded {expected_token}")
        expected_audio = golden[f"frame{t}.audio_out"]
        got = torch.as_tensor(frame.audio, dtype=torch.float32)
        audio_diff = max(audio_diff, float((got - expected_audio).abs().max()))
    report("B real DuplexSession tokens match", token_ok, f"{len(frames)} frames")
    report("B real DuplexSession audio matches", audio_diff == 0.0, f"max |diff| {audio_diff:.3e}")


# --------------------------------------------------------------------------- C

def _pattern(name: str) -> str:
    name = re.sub(r"\.stages\.\d+\.\d+", ".stages.N.M", name)
    return re.sub(r"\.layers\.\d+", ".layers.N", name)


def check_checkpoint_names(ckpt: Path) -> None:
    header_path = REAL_DIR / "header.json"
    if not header_path.is_file():
        report("C checkpoint name patterns", False, f"missing {header_path}")
        return
    header = json.loads(header_path.read_text())
    real = {name: info for name, info in header.items() if name != "__metadata__"}
    tiny = load_file(str(ckpt / "model.safetensors"))

    real_patterns = {_pattern(name) for name in real}
    tiny_patterns = {_pattern(name) for name in tiny}
    missing = sorted(real_patterns - tiny_patterns)
    extra = sorted(tiny_patterns - real_patterns)
    report(
        "C checkpoint name patterns equal real header",
        not missing and not extra,
        f"{len(tiny_patterns)} patterns; missing={missing} extra={extra}",
    )

    by_real: dict[str, list[tuple[int, ...]]] = {}
    by_tiny: dict[str, list[tuple[int, ...]]] = {}
    for name, info in real.items():
        by_real.setdefault(_pattern(name), []).append(tuple(info["shape"]))
    for name, tensor in tiny.items():
        by_tiny.setdefault(_pattern(name), []).append(tuple(tensor.shape))

    shape_problems = []
    count_notes = []
    for pattern, tiny_shapes in by_tiny.items():
        real_shapes = by_real.get(pattern)
        if real_shapes is None:
            continue
        if len(real_shapes) != len(tiny_shapes):
            count_notes.append(f"{pattern}: {len(tiny_shapes)} vs {len(real_shapes)}")
        if len(real_shapes[0]) != len(tiny_shapes[0]):
            shape_problems.append(f"{pattern} rank {len(tiny_shapes[0])} vs {len(real_shapes[0])}")
            continue
        tiny_numel = max(1, torch.tensor(tiny_shapes[0]).prod().item())
        real_numel = max(1, torch.tensor(real_shapes[0]).prod().item())
        if tiny_numel > real_numel:
            shape_problems.append(f"{pattern} numel {tiny_numel} > {real_numel}")
    report(
        "C checkpoint shapes scale down from real",
        not shape_problems,
        "; ".join(shape_problems) if shape_problems else "all tiny tensors no larger than real",
    )
    if count_notes:
        print(f"    expected count differences (tiny has fewer layers/blocks): {len(count_notes)} patterns")
    report(
        "C config geometry",
        tiny["lm.embed_tokens.weight"].shape[0] == gen.VOCAB
        and tiny["lm.embed_tokens.weight"].shape[1] == 64
        and abs(float(tiny["latent_scale"]) - 0.8) < 1e-7
        and abs(float(tiny["latent_bias"]) - 0.1) < 1e-7,
        f"vocab={tiny['lm.embed_tokens.weight'].shape[0]} hidden={tiny['lm.embed_tokens.weight'].shape[1]} "
        f"latent_scale={float(tiny['latent_scale']):.2f} latent_bias={float(tiny['latent_bias']):.2f}",
    )


# --------------------------------------------------------------------------- D

def _family(name: str) -> str:
    if name.startswith("codec_dbg."):
        return "codec_debug"
    if name.startswith("prompt.ref_codec_mean") or name.endswith(".user_codec_mean"):
        return "codec_encode"
    if name.endswith(".audio_out"):
        return "codec_decode"
    if name.endswith(".embed") and not name.endswith(".head_latent_normalized"):
        return "projectors"
    if name.endswith(".latent_out") or name.endswith(".user_latent") or name.endswith(".model_latent_in"):
        return "latents"
    if ".logits" in name:
        return "logits"
    if name.endswith(".head_latent_normalized"):
        return "head"
    if ".hidden" in name or ".lm_layer" in name:
        return "lm"
    return "other"


def replay(golden: dict[str, torch.Tensor], model, threads: int) -> dict[str, torch.Tensor]:
    """Recompute every derived golden tensor from recorded inputs and noise."""
    torch.set_num_threads(threads)
    model.eval().requires_grad_(False)
    out: dict[str, torch.Tensor] = {}
    layers: list[torch.Tensor] = []

    def put(name: str, value: torch.Tensor) -> None:
        out[name] = value.detach().to("cpu").contiguous().clone()

    def run_lm(embeddings, cache):
        layers.clear()
        hidden = model(embeddings, cache)[:, -1]
        return hidden, list(layers)

    handles = [
        layer.register_forward_hook(lambda _m, _i, o: layers.append(o[:, -1].detach().clone()))
        for layer in model.lm.layers
    ]
    try:
        # Prompt: whole-clip codec encode, recorded noise, prompt assembly,
        # projectors, LM.  Positions before the reference carry no audio.
        ref_mean = model.codec.encode(golden["prompt.ref_audio"])[0]
        put("prompt.ref_codec_mean", ref_mean)
        n = int(golden["prompt.tokens"].shape[0])
        n_ref = int(ref_mean.shape[0])
        n_card = n - n_ref
        model_latents = torch.zeros(n, model.config.latent_dim, device=model.device)
        model_latents[n_card:] = ref_mean + golden["prompt.ref_noise"]
        user_latents = torch.zeros(n, model.config.latent_dim, device=model.device)
        put("prompt.model_latents", model_latents)
        put("prompt.user_latents", user_latents)
        prompt_embeds = model.embed(
            golden["prompt.tokens"][None],
            model_latents[None],
            user_latents[None],
            (torch.arange(n, device=model.device) >= n_card)[None],
            (torch.arange(n, device=model.device) >= n)[None],
        )
        put("prompt.embed", prompt_embeds[0])
        cache = model.new_cache()
        hidden, layer_values = run_lm(prompt_embeds, cache)
        put("prompt.hidden_last", hidden[0])
        for i, value in enumerate(layer_values):
            put(f"prompt.lm_layer{i}", value[0])

        # Streaming codec + LM, feeding recorded latents and tokens.
        stream = gen.open_stream(model.codec, backend="torch")
        stream.reset()

        for t in range(0, gen.USER_FRAMES + 1):
            if t > 0:
                reference_mean = stream.encode(golden[f"frame{t}.user_audio"][None])
                put(f"frame{t}.user_codec_mean", reference_mean[0])
                user_latent = reference_mean + golden[f"frame{t}.user_noise"][None]
                put(f"frame{t}.user_latent", user_latent[0])
                emb = model.embed(
                    golden[f"frame{t}.token_in"][None],
                    golden[f"frame{t}.model_latent_in"][None],
                    user_latent,
                )
                put(f"frame{t}.embed", emb[0, 0])
                hidden, layer_values = run_lm(emb, cache)
                put(f"frame{t}.hidden", hidden[0])
                if t <= 3:
                    for i, value in enumerate(layer_values):
                        put(f"frame{t}.lm_layer{i}", value[0])

            logits = model.logits(hidden)
            put(f"frame{t}.logits", logits[0])
            stream_ids = torch.tensor([gen.PAD_ID, gen.EOP_ID], device=model.device)
            put(f"frame{t}.logits_fp32_pad_eop", model.logits_fp32(hidden, stream_ids)[0])

            noise = golden[f"frame{t}.head_noise"][None]
            latent_normalized = model.head.sample(hidden, noise, gen.GUIDANCE_SCALE)
            put(f"frame{t}.head_latent_normalized", latent_normalized[0])
            latent = model.unscale_latents(latent_normalized)
            put(f"frame{t}.latent_out", latent[0])
            audio = stream.decode(latent[None])[0].float().clamp(-1.0, 1.0)
            put(f"frame{t}.audio_out", audio)
    finally:
        for handle in handles:
            handle.remove()

    debug = gen.codec_debug(
        model.codec,
        golden["frame1.user_audio"],
        golden["frame0.latent_out"],
    )
    for name, value in debug.items():
        put(name, value)
    return out


def check_replay(golden: dict[str, torch.Tensor], model) -> None:
    fresh = replay(golden, model, threads=1)
    other = replay(golden, model, threads=4)

    floor: dict[str, float] = {}
    for name in fresh:
        fam = _family(name)
        diff = float((fresh[name] - other[name]).abs().max()) if fresh[name].numel() else 0.0
        floor[fam] = max(floor.get(fam, 0.0), diff)

    worst: dict[str, float] = {}
    mismatches: list[str] = []
    for name, value in fresh.items():
        expected = golden.get(name)
        if expected is None:
            mismatches.append(f"{name} missing from golden")
            continue
        diff = float((value - expected).abs().max()) if value.numel() else 0.0
        fam = _family(name)
        worst[fam] = max(worst.get(fam, 0.0), diff)
        tolerance = max(floor.get(fam, 0.0) * 2.0, 1e-6)
        if diff > tolerance:
            mismatches.append(f"{name}: {diff:.3e} > {tolerance:.3e}")

    for fam in sorted(worst):
        print(
            f"    {fam:14s} replay max|diff| {worst[fam]:.3e}   "
            f"thread floor {floor.get(fam, 0.0):.3e}"
        )
    report(
        "D recorded noise/tokens replay through components",
        not mismatches,
        "; ".join(mismatches[:6]) if mismatches else f"{len(fresh)} tensors reproduced",
    )

    # The recorded draws must explain the recorded latents exactly.
    n_card = int(golden["prompt.tokens"].numel()) - int(golden["prompt.ref_codec_mean"].shape[0])
    prompt_rebuilt = (
        golden["prompt.ref_codec_mean"] + golden["prompt.ref_noise"]
    )
    prompt_slice = golden["prompt.model_latents"][n_card:]
    prompt_diff = float((prompt_rebuilt - prompt_slice).abs().max()) if prompt_slice.numel() else 0.0
    user_diffs = []
    head_diffs = []
    for t in range(1, gen.USER_FRAMES + 1):
        user_diffs.append(
            float((golden[f"frame{t}.user_codec_mean"] + golden[f"frame{t}.user_noise"]
                   - golden[f"frame{t}.user_latent"]).abs().max())
        )
        head_diffs.append(
            float((golden[f"frame{t}.head_latent_normalized"] / 0.8 - 0.1
                   - golden[f"frame{t}.latent_out"]).abs().max())
        )
    report(
        "D recorded noise reconstructs latents",
        prompt_diff <= 1e-6 and max(user_diffs, default=0.0) <= 1e-6
        and max(head_diffs, default=0.0) <= 1e-6,
        f"prompt {prompt_diff:.3e}, user {max(user_diffs, default=0.0):.3e}, "
        f"head/unscale {max(head_diffs, default=0.0):.3e}",
    )


# ------------------------------------------------------------------------ main

def main() -> int:
    torch.set_num_threads(1)
    committed_ckpt = HERE / "haru_tiny"
    golden_path = HERE / "golden.safetensors"
    if not committed_ckpt.is_dir() or not golden_path.is_file():
        print("FAIL missing committed checkpoint or golden.safetensors")
        return 1

    check_regeneration()

    from haru.models.haru import HaruModel

    model = HaruModel.from_pretrained(str(committed_ckpt), device="cpu", dtype=torch.float32)
    golden = load_file(str(golden_path))
    check_faithful_loop(model, model.tokenizer, golden)
    check_checkpoint_names(committed_ckpt)
    check_replay(golden, model)

    print()
    if FAILURES:
        print(f"RESULT FAIL ({len(FAILURES)} failing checks): {', '.join(FAILURES)}")
        return 1
    print("RESULT PASS (all checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
