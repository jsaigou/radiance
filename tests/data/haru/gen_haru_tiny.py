#!/usr/bin/env python3
"""Build a tiny random-weight Haru checkpoint and its deterministic golden exchange.

This file is the single source of truth for everything under ``tests/data/haru``.
Run it with the pinned reference on PYTHONPATH::

    cd ~/haru-golden
    PYTHONPATH=Haru/haru venv/bin/python \
        <worktree>/tests/data/haru/gen_haru_tiny.py --out <worktree>/tests/data/haru

The script never writes into the reference checkout.  It imports the public
``haru`` package read-only and uses the reference classes unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

try:
    from haru.config import HaruConfig
    from haru.models.haru import HaruModel
    from haru.models.vibevoice_codec import CodecStream, open_stream
    from haru.prompt import build_prompt
    from haru.sampling import DryConfig, DryState, sample_token
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
except ImportError:  # pragma: no cover - a clearer message for a wrong checkout
    raise SystemExit(
        "cannot import the Haru reference; run with "
        "PYTHONPATH=<haru-golden>/Haru/haru and the pinned venv"
    )

# --------------------------------------------------------------------------- config

# The pinned public reference.  Kept here so the README, metadata and the C++
# authors all agree on the exact source of the golden values.
REFERENCE_COMMIT = "7fe6b7665f8bf035998f401f8cb779fbe53f2751"
REFERENCE_TAG = "haru-speech-v0.1.1"

MODEL_SEED = 20241118
REF_SEED = 424242
USER_SEED = 515151

SYSTEM_PROMPT = "You are Haru."
REFERENCE_SAMPLES = 44_000  # 13.75 frames -> exercises ceil() padding to 14
USER_FRAMES = 24
VOCAB = 512
PAD_ID, EOP_ID, EOS_ID = 480, 481, 500
IM_START_ID, IM_END_ID = 482, 483

SESSION_SEED = 1234
TEMPERATURE = 0.7
TOP_K = 25
TOP_P = 1.0
GUIDANCE_SCALE = 1.5
EOP_BIAS = 0.0


def tiny_config_dict() -> dict:
    """Same schema as ~/haru-golden/real/config.json, with tiny values.

    Geometry notes
    --------------
    * Qwen3.5 rotary width is ``int(head_dim * partial_rotary_factor)``; the
      inverse-frequency vector therefore has ``int(head_dim * 0.25) / 2``
      entries and ``mrope_section`` must sum to that number.  The real model
      has head_dim 256 -> rotary 64 -> 32 frequencies -> [11, 11, 10].
      Tiny uses head_dim 64 -> rotary 16 -> 8 frequencies -> [3, 3, 2].
    * GQA is preserved: 4 query heads, 2 KV heads.
    * The gated delta net keeps the real 1:2 key:value head ratio (2 key
      heads, 4 value heads) and conv kernel 4.
    * ``base_channels`` is 1 rather than the real 32: the hard 4 MB budget
      for this directory cannot hold seven 8-channel-wide stages (that is
      ~22 MB of fp32 weights).  One channel keeps all seven stages, the real
      ratios, one block per stage and every tensor name intact.  See README.
    """
    return {
        "model_type": "haru",
        "architectures": ["HaruForConditionalGeneration"],
        "dtype": "float32",
        "sample_rate": 24_000,
        "frame_samples": 3_200,
        "frame_rate": 7.5,
        "latent_dim": 64,
        "pad_token_id": PAD_ID,
        "eop_token_id": EOP_ID,
        "lm": {
            "model_type": "qwen3_5_text",
            "vocab_size": VOCAB,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "hidden_act": "silu",
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "mamba_ssm_dtype": "float32",
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [3, 3, 2],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10_000_000,
                "rope_type": "default",
            },
            "tie_word_embeddings": True,
            "initializer_range": 0.02,
            "eos_token_id": EOS_ID,
        },
        "codec": {
            "latent_dim": 64,
            "base_channels": 1,
            "ratios": [8, 5, 5, 4, 2, 2],
            "encoder_depths": [1, 1, 1, 1, 1, 1, 1],
            "decoder_depths": [1, 1, 1, 1, 1, 1, 1],
            "norm_eps": 1e-5,
            "latent_std": 0.5,
        },
        "head": {
            "hidden_size": 64,
            "latent_dim": 64,
            "num_layers": 2,
            "ffn_ratio": 2.0,
            "norm_eps": 1e-5,
            "interval_embed": 7.0,
            "num_steps": 1,
            "guidance_scale": GUIDANCE_SCALE,
            "guidance_range": [1.0, 2.0],
        },
    }


def build_config() -> HaruConfig:
    config = HaruConfig.from_dict(tiny_config_dict())
    rope = config.lm["rope_parameters"]
    head_dim = int(config.lm["head_dim"])
    rotary_dim = int(head_dim * float(rope["partial_rotary_factor"]))
    if sum(rope["mrope_section"]) != rotary_dim // 2:
        raise ValueError("mrope_section must sum to the number of rotary frequencies")
    if not rope.get("mrope_interleaved", False):
        raise ValueError("this checkpoint expects interleaved mRoPE")
    return config


# ------------------------------------------------------------------------- weights

_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "linear_attn.norm.weight",
)
_QWEN_ZERO_CENTERED = ("lm.norm.weight",)


def _is_qwen_norm(name: str) -> bool:
    if name in _QWEN_ZERO_CENTERED:
        return True
    return any(name.endswith(s) for s in _NORM_SUFFIXES)


def _is_plain_norm(name: str) -> bool:
    return (
        name.endswith("norm.weight")
        or name.endswith("ffn_norm.weight")
        or name.endswith(".gamma")
        or name.endswith(".ffn_gamma")
    )


def _name_generator(seed: int, name: str) -> torch.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(value)
    return gen


def randomize_weights(model: HaruModel, seed: int = MODEL_SEED) -> None:
    """Deterministic, non-trivial weights independent of torch's global RNG.

    Every persistent parameter and buffer gets data derived only from its own
    fully-qualified name and ``seed``.  Norm scales and latent_scale/latent_bias
    are deliberately away from their defaults so a C++ implementation that
    forgets a norm or a scale cannot pass.
    """
    persistent = set(model.state_dict().keys())
    for name, param in model.named_parameters():
        with torch.no_grad():
            if name.endswith("A_log"):
                fill = torch.randn(param.shape, generator=_name_generator(seed, name)) * 0.5
            elif name.endswith("dt_bias"):
                fill = 1.0 + 0.25 * torch.randn(param.shape, generator=_name_generator(seed, name))
            elif _is_qwen_norm(name):
                # Qwen3_5RMSNorm stores a zero-centred scale: effective 1 + w.
                fill = 0.2 * torch.randn(param.shape, generator=_name_generator(seed, name))
            elif _is_plain_norm(name):
                fill = 1.0 + 0.25 * torch.randn(param.shape, generator=_name_generator(seed, name))
            elif name.endswith(".bias"):
                fill = 0.02 * torch.randn(param.shape, generator=_name_generator(seed, name))
            else:
                fill = 0.05 * torch.randn(param.shape, generator=_name_generator(seed, name))
            param.copy_(fill.to(param.dtype))
    for name, buf in model.named_buffers():
        if name not in persistent:
            continue
        with torch.no_grad():
            if name == "latent_scale":
                buf.fill_(0.8)
            elif name == "latent_bias":
                buf.fill_(0.1)
            else:  # no other persistent buffers exist today
                buf.copy_(torch.randn(buf.shape, generator=_name_generator(seed, name)).to(buf.dtype))
    if not torch.isfinite(torch.stack([p.float().flatten().sum() for p in model.parameters()])).all():
        raise RuntimeError("non-finite parameter after randomisation")


# ----------------------------------------------------------------------- tokenizer

def build_tokenizer() -> Tokenizer:
    """A byte-level BPE tokenizer with 512 ids and the reserved Haru ids.

    Ids 0..255 are the 256 byte-level alphabet symbols, so any UTF-8 string
    round-trips.  Ids 480 (pad), 481 (eop), 482 (<|im_start|>), 483
    (<|im_end|>) and 500 (eos) are special tokens at fixed ids.
    """
    # ByteLevel.alphabet() is a set, so sort for a stable id assignment.
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    if len(alphabet) != 256:
        raise RuntimeError(f"unexpected byte-level alphabet size {len(alphabet)}")
    specials = {
        PAD_ID: "<|pad|>",
        EOP_ID: "<|eop|>",
        IM_START_ID: "<|im_start|>",
        IM_END_ID: "<|im_end|>",
        EOS_ID: "<|eos|>",
    }
    vocab: dict[str, int] = {token: i for i, token in enumerate(alphabet)}
    for i in range(256, VOCAB):
        vocab[specials.get(i, f"<|unused_{i}|>")] = i
    if len(vocab) != VOCAB:
        raise RuntimeError(f"tokenizer vocab is {len(vocab)}, expected {VOCAB}")
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(text, special=True) for text in specials.values()])
    if tokenizer.get_vocab_size(with_added_tokens=True) != VOCAB:
        raise RuntimeError("tokenizer grew while adding special tokens")
    for token_id, text in specials.items():
        if tokenizer.token_to_id(text) != token_id:
            raise RuntimeError(f"special token {text} did not keep id {token_id}")
    return tokenizer


# ----------------------------------------------------------------------- checkpoint

def _safetensors_state(model: HaruModel) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {
        name: state[name].detach().to("cpu", torch.float32).contiguous().clone()
        for name in sorted(state)
    }


def save_checkpoint(model: HaruModel, tokenizer: Tokenizer, outdir: Path) -> None:
    ckpt = outdir / "haru_tiny"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text(
        json.dumps(tiny_config_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    from safetensors.torch import save_file

    save_file(_safetensors_state(model), str(ckpt / "model.safetensors"), metadata={"format": "pt"})
    tokenizer.save(str(ckpt / "tokenizer.json"))


def build_and_save(outdir: Path, seed: int = MODEL_SEED) -> tuple[HaruModel, Tokenizer]:
    config = build_config()
    torch.manual_seed(seed)
    model = HaruModel(config)
    randomize_weights(model, seed)
    model.eval().requires_grad_(False)
    tokenizer = build_tokenizer()
    save_checkpoint(model, tokenizer, outdir)
    return model, tokenizer


# --------------------------------------------------------------------------- audio

def _sines(seconds: float, freqs: tuple[float, ...], seed: int, noise: float, rate: int = 24_000):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    t = torch.arange(int(round(seconds * rate)), dtype=torch.float32) / rate
    x = torch.zeros_like(t)
    for k, f in enumerate(freqs):
        x += (0.35 / (k + 1)) * torch.sin(2 * math.pi * f * t + 0.7 * k)
    x += noise * torch.randn(t.shape, generator=gen)
    return x.clamp(-0.95, 0.95).to(torch.float32)


def reference_audio() -> torch.Tensor:
    return _sines(REFERENCE_SAMPLES / 24_000, (220.0, 331.0, 97.0), REF_SEED, 0.04)[:REFERENCE_SAMPLES]


def user_audio_frames() -> list[torch.Tensor]:
    frames = []
    for i in range(USER_FRAMES):
        f0 = 160.0 + 17.0 * i
        frames.append(_sines(3_200 / 24_000, (f0, f0 * 1.5, f0 * 2.25), USER_SEED + i, 0.06))
    return frames


# ------------------------------------------------------------------- sample helper

def sample_codec(codec, mean: torch.Tensor, generator: torch.Generator):
    """Exact copy of VibeVoiceCodec.sample, returning every random draw.

    ``vibevoice_codec.py`` draws one scale per batch row and one noise tensor;
    the caller records both so a C++ engine can replay without torch's RNG.
    """
    scale = torch.randn(mean.shape[0], 1, 1, device=mean.device, dtype=mean.dtype, generator=generator)
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    added = scale * (codec.config.latent_std / 0.8) * noise
    return mean + added, scale, noise, added


class _RecordingCodec:
    """Delegates to the real codec but records build_prompt's random draws."""

    def __init__(self, codec):
        self.codec = codec
        self.draws: list[dict[str, torch.Tensor]] = []
        self.last_encode: torch.Tensor | None = None

    @property
    def device(self):
        return self.codec.device

    @property
    def config(self):
        return self.codec.config

    @property
    def dtype(self):
        return self.codec.dtype

    def encode(self, audio):
        self.last_encode = self.codec.encode(audio)
        return self.last_encode

    def sample(self, mean, generator=None):
        value, scale, noise, added = sample_codec(self.codec, mean, generator)
        self.draws.append({"scale": scale, "noise": noise, "added": added})
        return value


# ------------------------------------------------------------------------ recording

def _put(out: dict[str, torch.Tensor], name: str, tensor: torch.Tensor) -> None:
    out[name] = tensor.detach().to("cpu").contiguous().clone()


def _run_lm(model: HaruModel, embeddings: torch.Tensor, cache, layer_capture: list) -> tuple[torch.Tensor, list[torch.Tensor]]:
    layer_capture.clear()
    hidden = model(embeddings, cache)[:, -1]
    return hidden, [t[0] for t in layer_capture]


def _stage_boundaries(depths) -> list[int]:
    boundaries, pos = [], 0
    for depth in depths:
        pos += 1 + depth
        boundaries.append(pos - 1)
    return boundaries


def _run_stack_capture(stack, x: torch.Tensor, prefix: str, depths) -> dict[str, torch.Tensor]:
    """Run one stack with forward hooks and keep the output of every stage."""
    sequence = stack.sequence()
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        # Every module in a stack returns (tensor, new_state).
        captured.append(output[0].detach().float().cpu().clone())

    handles = [module.register_forward_hook(hook) for module in sequence]
    try:
        stack.run(x, stack.init_state(x.shape[0]))
    finally:
        for handle in handles:
            handle.remove()

    boundaries = _stage_boundaries(depths)
    out: dict[str, torch.Tensor] = {}
    for stage, index in enumerate(boundaries):
        out[f"codec_dbg.{prefix}_stage{stage}"] = captured[index]
    return out


def codec_debug(codec, encode_frame: torch.Tensor, decode_latent: torch.Tensor) -> dict[str, torch.Tensor]:
    """Fresh-state stage-by-stage enc/dec outputs for one frame each."""
    out: dict[str, torch.Tensor] = {}
    audio = encode_frame.reshape(1, 1, -1).to(codec.device, codec.dtype)
    out.update(_run_stack_capture(codec.encoder, audio, "enc", codec.config.encoder_depths))
    latents = decode_latent.reshape(1, -1, 1).to(codec.device, codec.dtype)
    out.update(_run_stack_capture(codec.decoder, latents, "dec", codec.config.decoder_depths))
    return out


# ------------------------------------------------------------------------ generation

def run_generation(
    model: HaruModel,
    tokenizer: Tokenizer,
    reference: torch.Tensor,
    user_frames: list[torch.Tensor],
    seed: int = SESSION_SEED,
    threads: int = 1,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Replay session.py's exact order of operations, recording every draw."""
    torch.set_num_threads(threads)
    model.eval().requires_grad_(False)
    out: dict[str, torch.Tensor] = {}
    per_frame: dict[int, dict] = {}
    draws: dict[str, list] = {"prompt": [], "user": []}
    last_prompt_layers: list[torch.Tensor] = []

    layer_capture: list[torch.Tensor] = []

    def record_last_position(name: str, tensor: torch.Tensor) -> None:
        _put(out, name, tensor[0])

    handles = [
        layer.register_forward_hook(
            lambda _m, _i, o: layer_capture.append(o[:, -1].detach().clone())
        )
        for layer in model.lm.layers
    ]
    try:
        # --- DuplexSession.start(), session.py lines 102-107 ------------------
        generator = torch.Generator(device=model.device)
        generator.manual_seed(seed)                       # session.py:105
        codec_stream = open_stream(model.codec, backend="torch")
        codec_stream.reset()                              # session.py:106
        cache = model.new_cache()                         # session.py:107

        # --- session.py lines 109-111 (build_prompt) -------------------------
        proxy = _RecordingCodec(model.codec)
        prompt = build_prompt(
            model.config,
            proxy,
            tokenizer,
            SYSTEM_PROMPT,
            reference,
            False,
            generator,
        )
        draws["prompt"] = list(proxy.draws)
        n = prompt.length

        _put(out, "prompt.tokens", prompt.tokens.to(torch.int64))
        _put(out, "prompt.model_latents", prompt.model_latents)
        _put(out, "prompt.user_latents", prompt.user_latents)
        _put(out, "prompt.model_mask", prompt.model_mask.to(torch.int64))
        _put(out, "prompt.user_mask", prompt.user_mask.to(torch.int64))
        _put(out, "prompt.ref_audio", reference)
        _put(out, "prompt.ref_codec_mean", proxy.last_encode[0].float())
        prompt_draw = proxy.draws[0]
        _put(out, "prompt.ref_noise", prompt_draw["added"][0].float())
        _put(out, "prompt.ref_noise_raw", prompt_draw["noise"][0].float())
        _put(out, "prompt.ref_noise_scale", prompt_draw["scale"].reshape(-1).float())

        # --- session.py lines 115-123 (prefill) ------------------------------
        prompt_embeds = []
        hidden = None
        for s in range(0, n, 512):                        # session.py:116
            e = min(n, s + 512)                           # session.py:117
            span = slice(s, e)
            emb = model.embed(                             # session.py:119-122
                prompt.tokens[None, span],
                prompt.model_latents[None, span],
                prompt.user_latents[None, span],
                prompt.model_mask[None, span],
                prompt.user_mask[None, span],
            )
            prompt_embeds.append(emb)
            hidden, layers = _run_lm(model, emb, cache, layer_capture)  # session.py:123
            last_prompt_layers = layers
        prompt_embed = torch.cat(prompt_embeds, dim=1)
        _put(out, "prompt.embed", prompt_embed[0])
        _put(out, "prompt.hidden_last", hidden[0])
        for i, layer in enumerate(last_prompt_layers):
            _put(out, f"prompt.lm_layer{i}", layer)

        # --- session.py lines 112-113, 124-126 -------------------------------
        force_text = 0                                    # session.py:112
        prompt_length = n                                 # session.py:124
        index = 0                                         # session.py:125

        stream_ids = torch.tensor([model.config.pad_token_id, model.config.eop_token_id],
                                  device=model.device)
        banned = sorted({model.config.pad_token_id, model.config.eop_token_id,
                         model.config.eos_token_id})
        dry_cfg = DryConfig()
        dry = DryState(dry_cfg, set(banned))

        def emit(frame_index: int, hidden: torch.Tensor, token_in, latent_in, embed, user_payload=None):
            nonlocal force_text
            # session.py lines 147-157 -------------------------------------
            logits = model.logits(hidden)                 # session.py:148
            logits_raw = logits[0].clone()
            dry.apply(logits[0])                          # session.py:150
            fp32_pad_eop = model.logits_fp32(hidden, stream_ids)  # session.py:151
            logits[:, stream_ids] = fp32_pad_eop          # session.py:151
            if EOP_BIAS:                                  # session.py:152-153
                logits[:, model.config.eop_token_id] += EOP_BIAS
            if force_text > 0:                            # session.py:154-156
                logits[:, banned] = float("-inf")
                force_text -= 1
            token = sample_token(logits, TEMPERATURE, TOP_K, TOP_P, generator)  # session.py:157

            # session.py lines 159-163 -------------------------------------
            noise = torch.randn(                             # session.py:159-160
                1, model.config.latent_dim, device=hidden.device,
                dtype=model.dtype, generator=generator,
            )
            latent_normalized = model.head.sample(hidden, noise, GUIDANCE_SCALE)
            latent = model.unscale_latents(latent_normalized)  # session.py:162
            audio = codec_stream.decode(latent[None])[0].float().clamp(-1.0, 1.0)  # session.py:163

            # session.py lines 165-168 -------------------------------------
            token_id = int(token)
            dry.update(token_id)                          # session.py:167-168

            _put(out, f"frame{frame_index}.token_in", token_in.reshape(-1).to(torch.int64))
            _put(out, f"frame{frame_index}.model_latent_in", latent_in.reshape(-1))
            _put(out, f"frame{frame_index}.embed", embed.reshape(-1))
            _put(out, f"frame{frame_index}.hidden", hidden[0])
            _put(out, f"frame{frame_index}.logits", logits_raw)
            _put(out, f"frame{frame_index}.logits_fp32_pad_eop", fp32_pad_eop[0])
            _put(out, f"frame{frame_index}.logits_final", logits[0])
            _put(out, f"frame{frame_index}.token_out", token.reshape(-1).to(torch.int64))
            _put(out, f"frame{frame_index}.head_noise", noise[0])
            _put(out, f"frame{frame_index}.head_latent_normalized", latent_normalized[0])
            _put(out, f"frame{frame_index}.latent_out", latent[0])
            _put(out, f"frame{frame_index}.audio_out", audio)
            if user_payload is not None:
                for key, value in user_payload.items():
                    _put(out, f"frame{frame_index}.{key}", value)
            per_frame[frame_index] = {
                "token": token_id,
                "latent": latent.detach().clone(),
                "audio": audio.detach().clone(),
                "hidden": hidden.detach().clone(),
            }
            return token, latent

        # --- session.py line 126: frame 0 from the prompt's last position ----
        frame0_token, frame0_latent = emit(
            0,
            hidden,
            prompt.tokens[-1],
            prompt.model_latents[-1].float(),
            prompt_embed[0, -1],
        )
        self_token, self_latent = frame0_token, frame0_latent
        index = 1

        # --- session.py lines 137-143, once per user frame ------------------
        for t in range(1, USER_FRAMES + 1):
            audio = torch.as_tensor(user_frames[t - 1], dtype=torch.float32).reshape(1, -1)
            if audio.shape[1] != model.config.frame_samples:
                raise ValueError("user frame has the wrong length")
            user_mean = codec_stream.encode(audio)        # session.py:140
            user_latent, u_scale, u_noise, u_added = sample_codec(
                model.codec, user_mean, generator
            )                                             # session.py:140
            _put(out, f"frame{t}.user_audio", audio[0])
            _put(out, f"frame{t}.user_codec_mean", user_mean[0].float())
            _put(out, f"frame{t}.user_noise", u_added[0].float())
            _put(out, f"frame{t}.user_noise_raw", u_noise[0].float())
            _put(out, f"frame{t}.user_noise_scale", u_scale.reshape(-1).float())
            _put(out, f"frame{t}.user_latent", user_latent[0].float())

            emb = model.embed(self_token[None], self_latent[None], user_latent)  # session.py:141
            hidden, layers = _run_lm(model, emb, cache, layer_capture)           # session.py:142
            if t <= 3:
                for i, layer in enumerate(layers):
                    _put(out, f"frame{t}.lm_layer{i}", layer)

            self_token, self_latent = emit(
                t,
                hidden,
                self_token,
                self_latent[0],
                emb[0, 0],
                {
                    "user_audio": audio[0],
                    "user_codec_mean": user_mean[0].float(),
                    "user_noise": u_added[0].float(),
                    "user_noise_raw": u_noise[0].float(),
                    "user_noise_scale": u_scale.reshape(-1).float(),
                    "user_latent": user_latent[0].float(),
                },
            )
            index += 1
    finally:
        for handle in handles:
            handle.remove()

    # Codec stage debug on fresh state; no RNG involved.
    out.update(codec_debug(model.codec, user_frames[0], per_frame[0]["latent"][0]))

    meta = {
        "reference_commit": REFERENCE_COMMIT,
        "reference_tag": REFERENCE_TAG,
        "seed": seed,
        "model_seed": MODEL_SEED,
        "reference_audio_seed": REF_SEED,
        "user_audio_seed": USER_SEED,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_length": int(out["prompt.tokens"].numel()),
        "reference_samples": int(reference.numel()),
        "reference_frames": int(out["prompt.ref_codec_mean"].shape[0]),
        "user_frames": USER_FRAMES,
        "sampling": {
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "guidance_scale": GUIDANCE_SCALE,
            "eop_bias": EOP_BIAS,
            "speak_first": False,
            "max_frames": 27_000,
            "reference_seconds": 180.0,
            "dry": {
                "multiplier": DryConfig().multiplier,
                "base": DryConfig().base,
                "allowed_length": DryConfig().allowed_length,
                "lookback": DryConfig().lookback,
                "max_match_length": DryConfig().max_match_length,
                "max_penalty": DryConfig().max_penalty,
            },
            "banned_tokens": banned,
            "stream_ids": [int(stream_ids[0]), int(stream_ids[1])],
        },
        "frame_tokens": [int(per_frame[i]["token"]) for i in range(USER_FRAMES + 1)],
        "threads": threads,
        "torch_version": torch.__version__,
        "note": (
            "frame t (t>=1) is produced by step(user_audio[t-1]); frame 0 comes "
            "from the end of the prompt. Frame t's *_in tensors are the inputs of "
            "the position that produced frame t."
        ),
    }
    return out, meta


# ---------------------------------------------------------------------- generation

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def generate(outdir: Path, threads: int = 1) -> dict[str, torch.Tensor]:
    outdir.mkdir(parents=True, exist_ok=True)
    build_and_save(outdir)
    # Load exactly what was written: the golden values then describe the
    # committed checkpoint, and the HF-style loader is exercised as well.
    model = HaruModel.from_pretrained(str(outdir / "haru_tiny"), device="cpu", dtype=torch.float32)
    tokenizer = model.tokenizer
    reference = reference_audio()
    user_frames = user_audio_frames()
    tensors, meta = run_generation(model, tokenizer, reference, user_frames, threads=threads)

    from safetensors.torch import save_file

    save_file(
        {name: tensors[name] for name in sorted(tensors)},
        str(outdir / "golden.safetensors"),
        metadata={"format": "pt"},
    )
    ckpt = outdir / "haru_tiny"
    meta = dict(meta)
    meta["hashes"] = {
        "model.safetensors": _sha256(ckpt / "model.safetensors"),
        "config.json": _sha256(ckpt / "config.json"),
        "tokenizer.json": _sha256(ckpt / "tokenizer.json"),
        "golden.safetensors": _sha256(outdir / "golden.safetensors"),
        "state_dict_sha256": _tensor_digest(_safetensors_state(model)),
        "golden_tensor_sha256": _tensor_digest(tensors),
    }
    meta["tensor_count"] = len(tensors)
    (outdir / "golden_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tensors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                        help="directory that receives haru_tiny/, golden.safetensors and golden_meta.json")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch CPU threads; fixed at 1 by default for bit-identical output")
    args = parser.parse_args(argv)
    if args.out.resolve() == Path(__file__).resolve().parent and args.threads != 1:
        raise SystemExit("the committed golden files were produced with --threads 1")
    tensors = generate(args.out, threads=args.threads)
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {len(tensors)} golden tensors ({total} bytes) to {args.out}")


if __name__ == "__main__":
    main()
