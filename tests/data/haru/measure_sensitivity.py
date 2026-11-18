#!/usr/bin/env python3
"""Measure numerical sensitivity of the committed Haru golden exchange.

The recorded noise and tokens are replayed through the reference twice:

* model and all float inputs in float64, compared with the committed float32
  goldens (the numerical-sensitivity measurement);
* model and all float inputs in bfloat16, compared with the same float32
  goldens (informational, because the real checkpoint is bf16).

No committed checkpoint or golden file is modified.  The output is
deterministic when torch runs with one thread; ``check_haru_golden.py`` reruns
this measurement and compares it with the JSON block in README.md.

Run from ~/haru-golden with the pinned reference importable::

    PYTHONPATH=Haru/haru venv/bin/python <worktree>/tests/data/haru/measure_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE_PACKAGE = Path(
    os.environ.get("HARU_REFERENCE_ROOT", Path.home() / "haru-golden" / "Haru" / "haru")
).expanduser()
sys.path.insert(0, str(HERE))
if REFERENCE_PACKAGE.is_dir():
    sys.path.insert(0, str(REFERENCE_PACKAGE))

import torch
from safetensors.torch import load_file

import gen_haru_tiny as gen

REL_FLOOR = 1e-3
NORM_PAIRS = (
    (1e-6, 1e-6),
    (1e-5, 1e-5),
    (1e-4, 1e-4),
    (1e-6, 1e-4),
)
FAMILY_ORDER = [
    "codec_encode_mean",
    "codec_decode_audio",
    *[f"codec_dbg.enc_stage{i}" for i in range(7)],
    *[f"codec_dbg.dec_stage{i}" for i in range(7)],
    "projector_embed",
    "lm_layer0",
    "lm_layer1",
    "lm_layer2",
    "lm_layer3",
    "lm_hidden",
    "logits",
    "head_output",
    "latent_out",
    "user_latent",
    "prompt_tensors",
]


def _family(name: str) -> str:
    if name.startswith("codec_dbg.enc_stage") or name.startswith("codec_dbg.dec_stage"):
        return name
    if name == "prompt.ref_codec_mean" or name.endswith(".user_codec_mean"):
        return "codec_encode_mean"
    if name.endswith(".audio_out"):
        return "codec_decode_audio"
    if name.endswith(".embed"):
        return "projector_embed"
    match = re.search(r"\.lm_layer(\d+)$", name)
    if match:
        return f"lm_layer{match.group(1)}"
    if name == "prompt.hidden_last" or name.endswith(".hidden"):
        return "lm_hidden"
    if ".logits" in name:
        return "logits"
    if name.endswith(".head_latent_normalized"):
        return "head_output"
    if name.endswith(".latent_out"):
        return "latent_out"
    if name.endswith(".user_latent"):
        return "user_latent"
    if name.startswith("prompt."):
        return "prompt_tensors"
    raise KeyError(f"no sensitivity family for {name}")


def _replay(golden: dict[str, torch.Tensor], dtype: torch.dtype, threads: int = 1) -> dict[str, torch.Tensor]:
    """Replay recorded inputs/noise with the reference model in ``dtype``."""
    from haru.models.haru import HaruModel
    from haru.sampling import DryConfig, DryState

    torch.set_num_threads(threads)
    model = HaruModel.from_pretrained(str(HERE / "haru_tiny"), device="cpu", dtype=dtype)
    model.eval().requires_grad_(False)
    out: dict[str, torch.Tensor] = {}
    layers: list[torch.Tensor] = []

    def cast(value: torch.Tensor) -> torch.Tensor:
        return value.to(dtype)

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
        # Prompt: recorded reference audio -> whole-clip codec mean -> recorded
        # noise -> projectors -> LM.
        ref_mean = model.codec.encode(cast(golden["prompt.ref_audio"]))[0]
        put("prompt.ref_codec_mean", ref_mean)
        n = int(golden["prompt.tokens"].numel())
        n_ref = int(ref_mean.shape[0])
        n_card = n - n_ref
        model_latents = torch.zeros(n, model.config.latent_dim, device=model.device, dtype=dtype)
        model_latents[n_card:] = ref_mean + cast(golden["prompt.ref_noise"])
        user_latents = torch.zeros(n, model.config.latent_dim, device=model.device, dtype=dtype)
        put("prompt.model_latents", model_latents)
        put("prompt.user_latents", user_latents)
        prompt_embeds = model.embed(
            golden["prompt.tokens"][None],
            model_latents[None],
            user_latents[None],
            golden["prompt.model_mask"][None],
            golden["prompt.user_mask"][None],
        )
        put("prompt.embed", prompt_embeds[0])
        cache = model.new_cache()
        hidden, layer_values = run_lm(prompt_embeds, cache)
        put("prompt.hidden_last", hidden[0])
        for i, value in enumerate(layer_values):
            put(f"prompt.lm_layer{i}", value[0])

        # Streaming codec + LM + head, using the recorded draws.
        stream = gen.open_stream(model.codec, backend="torch")
        stream.reset()
        stream_ids = torch.tensor([gen.PAD_ID, gen.EOP_ID], device=model.device)
        dry = DryState(DryConfig(), {gen.PAD_ID, gen.EOP_ID, gen.EOS_ID})

        for t in range(0, gen.USER_FRAMES + 1):
            if t > 0:
                mean = stream.encode(cast(golden[f"frame{t}.user_audio"])[None])
                put(f"frame{t}.user_codec_mean", mean[0])
                user_latent = mean + cast(golden[f"frame{t}.user_noise"])[None]
                put(f"frame{t}.user_latent", user_latent[0])
                emb = model.embed(
                    golden[f"frame{t}.token_in"][None],
                    cast(golden[f"frame{t}.model_latent_in"])[None],
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
            fp32_pad_eop = model.logits_fp32(hidden, stream_ids)
            put(f"frame{t}.logits_fp32_pad_eop", fp32_pad_eop[0])
            dry.apply(logits[0])
            logits[:, stream_ids] = fp32_pad_eop
            put(f"frame{t}.logits_final", logits[0])
            dry.update(int(golden[f"frame{t}.token_out"].reshape(-1)[0]))

            noise = cast(golden[f"frame{t}.head_noise"])[None]
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


def _family_totals(golden: dict[str, torch.Tensor], got: dict[str, torch.Tensor]) -> dict[str, dict]:
    totals: dict[str, dict] = {}
    for name, value in got.items():
        expected = golden[name]
        if tuple(value.shape) != tuple(expected.shape):
            raise ValueError(f"{name}: shape {tuple(value.shape)} != {tuple(expected.shape)}")
        fam = _family(name)
        ref = expected.to(torch.float64).reshape(-1)
        cur = value.to(torch.float64).reshape(-1)
        err = (cur - ref).abs()
        denom = ref.abs().clamp(min=REL_FLOOR)
        item = totals.setdefault(
            fam,
            {"max_abs": 0.0, "max_rel": 0.0, "norm": [0.0] * len(NORM_PAIRS)},
        )
        item["max_abs"] = max(item["max_abs"], float(err.max()) if err.numel() else 0.0)
        item["max_rel"] = max(item["max_rel"], float((err / denom).max()) if err.numel() else 0.0)
        for j, (atol, rtol) in enumerate(NORM_PAIRS):
            ratio = err / (atol + rtol * ref.abs())
            item["norm"][j] = max(item["norm"][j], float(ratio.max()) if ratio.numel() else 0.0)
    missing = [fam for fam in FAMILY_ORDER if fam not in totals]
    extra = [fam for fam in totals if fam not in FAMILY_ORDER]
    if missing or extra:
        raise ValueError(f"family mismatch: missing={missing} extra={extra}")
    return totals


def _round_up_one_significant(value: float) -> float:
    if value <= 0.0:
        return 0.0
    exponent = math.floor(math.log10(value))
    step = 10.0**exponent
    return math.ceil(value / step) * step


def measure(threads: int = 1) -> dict:
    """Return the deterministic sensitivity summary for both dtypes."""
    torch.set_num_threads(threads)
    golden = load_file(str(HERE / "golden.safetensors"))
    f64 = _family_totals(golden, _replay(golden, torch.float64, threads))
    bf16 = _family_totals(golden, _replay(golden, torch.bfloat16, threads))
    families: dict[str, dict] = {}
    for fam in FAMILY_ORDER:
        fp64 = f64[fam]
        bf = bf16[fam]
        if fp64["max_abs"] == 0.0:
            recommended = 1e-6
        else:
            recommended = _round_up_one_significant(fp64["max_abs"] * 10.0)
        families[fam] = {
            "fp64_max_abs": fp64["max_abs"],
            "fp64_max_rel": fp64["max_rel"],
            "fp64_norm": fp64["norm"],
            "bf16_max_abs": bf["max_abs"],
            "bf16_max_rel": bf["max_rel"],
            "recommended_abs": recommended,
        }
    return {
        "rel_floor": REL_FLOOR,
        "norm_pairs": [list(pair) for pair in NORM_PAIRS],
        "families": families,
    }


def markdown_table(result: dict) -> str:
    lines = [
        "| family | fp64 max abs | fp64 max rel | P1 | P2 | P3 | P4 | recommended abs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fam in FAMILY_ORDER:
        row = result["families"][fam]
        norms = " | ".join(f"{v:.3e}" for v in row["fp64_norm"])
        lines.append(
            f"| `{fam}` | {row['fp64_max_abs']:.3e} | {row['fp64_max_rel']:.3e} | "
            f"{norms} | {row['recommended_abs']:.3e} |"
        )
    return "\n".join(lines)


def markdown_bf16_table(result: dict) -> str:
    lines = [
        "| family | bf16 max abs | bf16 max rel |",
        "|---|---:|---:|",
    ]
    for fam in FAMILY_ORDER:
        row = result["families"][fam]
        lines.append(f"| `{fam}` | {row['bf16_max_abs']:.3e} | {row['bf16_max_rel']:.3e} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the canonical JSON summary")
    parser.add_argument("--table", action="store_true", help="print the fp64 Markdown table")
    parser.add_argument("--bf16-table", action="store_true", help="print the bf16 Markdown table")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    result = measure(threads=args.threads)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    if args.table:
        print(markdown_table(result))
    if args.bf16_table:
        print(markdown_bf16_table(result))
    if not (args.json or args.table or args.bf16_table):
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
