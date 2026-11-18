# Haru tiny checkpoint and golden exchange (H0)

This directory contains the Python reference's exact boundary tensors for a
tiny random-weight Haru, plus the weights and tokenizer needed to reproduce
them.  It exists so the C++ inference engine can be built and tested piece by
piece against the reference without a GPU or the real 4B checkpoint.

Pinned public reference:

* repository `haru-golden/Haru`
* commit `7fe6b7665f8bf035998f401f8cb779fbe53f2751`
* tag `haru-speech-v0.1.1`
* package `Haru/haru/haru` (read-only; never modified by this work)

All values in `golden.safetensors` were produced by that code on CPU with
torch 2.15.0 in float32.

## Contents

```
gen_haru_tiny.py         builds the checkpoint, tokenizer and golden exchange
check_haru_golden.py     four PASS/FAIL checks; exits non-zero on failure
golden.safetensors       486 float32/int64 tensors (1,110,964 bytes of payload)
golden_meta.json         sampling config, seeds, reference commit, hashes
haru_tiny/config.json    HF-style Haru config (same schema as the real one)
haru_tiny/model.safetensors  model.state_dict(), float32, real key names
haru_tiny/tokenizer.json small byte-level BPE tokenizer, 512 ids
```

Total size of this directory is below 4 MB (3.94 MB at the time of writing).

To regenerate everything (the committed files are byte-identical to a fresh
run with `--threads 1`).  Bit-identical regeneration required one fix: the
tokenizers byte-level alphabet is returned as a `set`, so its iteration order
and therefore the id assignment changed between processes.  The generator sorts
it before assigning ids; after that all five generated files hash the same on
every run.

```sh
cd ~/haru-golden
PYTHONPATH=Haru/haru venv/bin/python \
  <worktree>/tests/data/haru/gen_haru_tiny.py --out <worktree>/tests/data/haru
```

To verify:

```sh
cd ~/haru-golden
PYTHONPATH=Haru/haru venv/bin/python \
  <worktree>/tests/data/haru/check_haru_golden.py
```

The check prints one `PASS`/`FAIL` line per check and ends with
`RESULT PASS (all checks)` or `RESULT FAIL (...)`.  It regenerates into a
temporary directory, runs the real `DuplexSession` with the same seed, compares
tensor-name patterns against `~/haru-golden/real/header.json`, and replays the
recorded noise/tokens through the reference components.

`HARU_REFERENCE_ROOT` overrides the reference package path and `HARU_REAL_DIR`
overrides the directory holding `header.json`.

## Tiny model choices

The checkpoint uses the reference's own classes (`HaruModel`, `Qwen3_5TextModel`,
`MeanFlowHead`, `VibeVoiceCodec`) and `HaruConfig.from_dict`.  Weights are
deterministic: every tensor is filled from a generator seeded by
`sha256("20241118:" + tensor_name)`, so the result does not depend on torch's
global RNG or on module construction order.  Norm scales are `1 + 0.25*N(0,1)`
(Qwen norms are zero-centred, `0.2*N(0,1)`), and `latent_scale = 0.8`,
`latent_bias = 0.1`, so omitting a norm or a latent scale cannot pass.

LM (Qwen3.5 hybrid, 4 layers):

* vocab 512, hidden 64, intermediate 128, 4 query heads, 2 KV heads (GQA kept)
* layers `[linear, linear, linear, full]` (`full_attention_interval = 4`)
* gated delta net: 2 key heads vs 4 value heads (the real 1:2 ratio), head
  dims 16, conv kernel 4
* `attn_output_gate = true`, tied embeddings, `partial_rotary_factor = 0.25`,
  `mrope_interleaved = true`
* `pad = 480`, `eop = 481`, `eos = 500`

Rotary/mRoPE geometry.  Transformers computes
`rotary_dim = int(head_dim * partial_rotary_factor)` and the inverse-frequency
vector has `rotary_dim / 2` entries; `mrope_section` must sum to that number
(`Qwen3_5TextRotaryEmbedding.recomposition_frequencies`,
`modeling_qwen3_5.py:193-200`).  The real model has `head_dim = 256`,
rotary 64, 32 frequencies and `[11, 11, 10]`.  Tiny uses `head_dim = 64`,
rotary 16, 8 frequencies and `[3, 3, 2]`.  This keeps every index used by the
interleaved recomposition valid.

Codec: seven resolution stages, real ratios `(8, 5, 5, 4, 2, 2)` so
`frame_samples` stays 3200, one `Block` per stage in both encoder and decoder.
`base_channels` is **1**, not the real 32.  With 7 stages the channels are
`base * 2^i`, so the real 8-channel floor already needs about 22 MB of fp32
weights for encoder+decoder, and even `base_channels = 2` needs about 5 MB for
weights alone.  One channel keeps all seven stages, all block kinds, every
tensor name and the full 3200-sample frame while staying inside the 4 MB
budget.  Channels at the seven encoder stages are `1, 2, 4, 8, 16, 32, 64`.

Head: hidden 64, latent 64, 2 AdaLN layers, `ffn_ratio = 2.0` (128-wide FFN)
and the real `interval_embed = 7.0`, `num_steps = 1`, `guidance = 1.5`.
`ffn_ratio` is the only reduced head dimension; `ffn_ratio = 3.0` would add
about 128 KB of weights.  AdaLN shift/scale/gate and the final AdaLN projection
are unchanged.

Tokenizer: byte-level BPE with an empty merge table.  Ids 0..255 are the 256
byte-level alphabet symbols (sorted for stable ids), so arbitrary UTF-8 text
encodes; 480 `<|pad|>`, 481 `<|eop|>`, 482 `<|im_start|>`, 483 `<|im_end|>`
and 500 `<|eos|>` are special tokens at fixed ids.  No other id is special.

The loader check is part of `gen_haru_tiny.py`: after writing the directory it
calls `HaruModel.from_pretrained(dir, device="cpu", dtype=torch.float32)` and
generates the golden tensors from that loaded model, so a missing checkpoint
key or a loader mismatch would fail generation.

## The scripted exchange

* system prompt: exactly `"You are Haru."`, wrapped by `system_prompt_ids`
  (`prompt.py:46-50`) as `<|im_start|>system\nYou are Haru.<|im_end|>\n`
  -> 23 ids.
* reference clip: 44,000 samples = 13.75 frames.  `reference_frames` uses
  `ceil` (`prompt.py:57-58`), so the codec produces 14 latents; the final
  partial hop is handled inside `CausalConv1d.forward`
  (`vibevoice_codec.py:46-53`).  The clip is three seeded sines plus low noise.
* prompt length: 23 + 14 = 37 positions.  Tokens are the 23 card ids, then
  13 PAD and one EOP.  `model_mask` is true on positions 23..36;
  `user_mask` is false everywhere (`speak_first = False`).
* 24 user frames, each 3200 samples, seeded sines plus noise (not silence).
* session seed 1234, temperature 0.7, top-k 25, top-p 1.0, guidance 1.5,
  EOP bias 0, DRY defaults.  All of this is in `golden_meta.json`.

Frame indexing: frame 0 comes from the last prompt position.  Frame `t`
(`t = 1..24`) comes from `step(user_audio[t-1])`.  Frame `t`'s `*_in` tensors
are the inputs of the position that produced frame `t`: `token_in` is the
previous sampled token, `model_latent_in` is the previous `latent_out`, and
`user_latent` is the current user frame's codec latent.  This mirrors
`session.py:130-177` exactly; the generator cites those line numbers in
comments.

Every random draw is recorded: the codec's per-row scale and per-element noise
for the reference and each user frame, the head noise for each frame, and every
sampled token.  `*_noise` is the combined added term
`scale * (latent_std / 0.8) * raw`; `*_noise_raw` and `*_noise_scale` are the
two raw `randn` draws.  A replayer can therefore run without torch's RNG:
`latent = mean + noise`, and it uses the recorded `token_out`.

## Tensor schema

All tensors are float32 unless the dtype is `int64`.  `t` is 0..24; some
tensors exist only for a subset of `t`, as noted.  Line numbers refer to the
pinned reference.

### Prompt

| name | shape | dtype | what / produced by |
|---|---|---|---|
| `prompt.tokens` | `[37]` | int64 | card ids + reference PAD/EOP, `prompt.py:104,119` |
| `prompt.model_latents` | `[37, 64]` | float32 | zeros on card; reference latents at 23..36, `prompt.py:110-113` |
| `prompt.user_latents` | `[37, 64]` | float32 | zeros (no user audio in the prompt), `prompt.py:111` |
| `prompt.model_mask` | `[37]` | int64 | 1 from position 23, `prompt.py:122` |
| `prompt.user_mask` | `[37]` | int64 | all 0, `prompt.py:123` |
| `prompt.ref_audio` | `[44000]` | float32 | synthetic reference waveform (input) |
| `prompt.ref_codec_mean` | `[14, 64]` | float32 | whole-clip `codec.encode`, `vibevoice_codec.py:203-208` |
| `prompt.ref_noise` | `[14, 64]` | float32 | combined codec noise, `vibevoice_codec.py:198-200` |
| `prompt.ref_noise_raw` | `[14, 64]` | float32 | raw `torch.randn` draw, `vibevoice_codec.py:199` |
| `prompt.ref_noise_scale` | `[1]` | float32 | per-row scale draw, `vibevoice_codec.py:198` |
| `prompt.embed` | `[37, 64]` | float32 | `model.embed` for all prompt positions, `haru.py:83-107`, `session.py:119-122` |
| `prompt.hidden_last` | `[64]` | float32 | LM output after final norm at position 36, `session.py:123` |
| `prompt.lm_layer0..3` | `[64]` each | float32 | that layer's output at position 36 (pre-final-norm), `session.py:123` |

### Per frame

| name | shape | dtype | what / produced by |
|---|---|---|---|
| `frame{t}.user_audio` | `[3200]` | float32 | input user frame, only `t >= 1` (`session.py:137`) |
| `frame{t}.user_codec_mean` | `[1, 64]` | float32 | streaming encode, only `t >= 1`, `session.py:140`, `vibevoice_codec.py:236-242` |
| `frame{t}.user_noise` | `[1, 64]` | float32 | combined codec noise, only `t >= 1` |
| `frame{t}.user_noise_raw` | `[1, 64]` | float32 | raw noise draw, only `t >= 1` |
| `frame{t}.user_noise_scale` | `[1]` | float32 | scale draw, only `t >= 1` |
| `frame{t}.user_latent` | `[1, 64]` | float32 | `mean + noise`, only `t >= 1`, `session.py:140` |
| `frame{t}.token_in` | `[1]` | int64 | input token (prompt EOP for t=0, previous `token_out` after), `session.py:141,165` |
| `frame{t}.model_latent_in` | `[64]` | float32 | raw codec-space latent input (prompt latent for t=0, previous `latent_out` after) |
| `frame{t}.embed` | `[64]` | float32 | sum of token, model-audio and user-audio streams, `haru.py:83-107`, `session.py:119-122,141` |
| `frame{t}.hidden` | `[64]` | float32 | LM output after final norm, `session.py:123,142` |
| `frame{t}.lm_layer0..3` | `[64]` each | float32 | per-layer outputs at this position, only `t = 1,2,3` |
| `frame{t}.logits` | `[512]` | float32 | raw `model.logits(hidden)` before DRY, `haru.py:113-115`, `session.py:148` |
| `frame{t}.logits_fp32_pad_eop` | `[2]` | float32 | float32 logits for `[pad, eop]`, `haru.py:117-123`, `session.py:151` |
| `frame{t}.logits_final` | `[512]` | float32 | raw logits after DRY and the PAD/EOP replacement, exactly what sampling sees, `session.py:148-157` |
| `frame{t}.token_out` | `[1]` | int64 | sampled token, `sampling.py:80-100`, `session.py:157` |
| `frame{t}.head_noise` | `[64]` | float32 | `torch.randn` draw for the head, `session.py:159-160` |
| `frame{t}.head_latent_normalized` | `[64]` | float32 | `head.sample(hidden, noise, 1.5)`, `meanflow_head.py:117-134`, `session.py:162` |
| `frame{t}.latent_out` | `[64]` | float32 | `unscale_latents` -> raw codec space, `haru.py:75-76`, `session.py:162` |
| `frame{t}.audio_out` | `[3200]` | float32 | streaming decode then clamp `[-1, 1]`, `vibevoice_codec.py:245-249`, `session.py:163` |

### Codec debug stages

`codec_dbg.enc_stage{i}` and `codec_dbg.dec_stage{i}` are the output of every
resolution stage, captured with forward hooks, on fresh codec state:

* encoder input is `frame1.user_audio` (`[1, 1, 3200]`); stage shapes are
  `[1, 1, 3200]`, `[1, 2, 1600]`, `[1, 4, 800]`, `[1, 8, 200]`,
  `[1, 16, 40]`, `[1, 32, 8]`, `[1, 64, 1]`.
* decoder input is `frame0.latent_out` (`[1, 64, 1]`); stage shapes are
  `[1, 64, 1]`, `[1, 32, 8]`, `[1, 16, 40]`, `[1, 8, 200]`,
  `[1, 4, 800]`, `[1, 2, 1600]`, `[1, 1, 3200]`.

A C++ codec can run one 3200-sample encode and one 3200-sample decode and
compare stage by stage; the first differing stage localises the bug.

## What each C++ component should compare

* **Codec encoder.**  Feed `prompt.ref_audio` to a whole-clip encode and compare
  `prompt.ref_codec_mean` (the clip is not a frame multiple: the encoder pads the
  last hop).  Feed `frame{t}.user_audio` through a streaming encoder in order and
  compare `frame{t}.user_codec_mean`.  For stage-level debugging compare
  `codec_dbg.enc_stage0..6`.
* **Codec noise.**  Draw/replay both values: `user_latent = user_codec_mean +
  user_noise`; `prompt.model_latents[23:37] = ref_codec_mean + ref_noise`
  (the card positions are zero).  `latent_std = 0.5` and the reference divides
  by `0.8` (`vibevoice_codec.py:200`).
* **Latent scaling.**  `scale_latents(raw) = (raw + 0.1) * 0.8` and
  `unscale_latents(z) = z / 0.8 - 0.1` (`haru.py:71-76`).  These are used inside
  the projectors and when the head output becomes the next `model_latent_in`.
* **Projectors.**  `embed = embed_tokens(token) + model_audio_proj(scale(model_latent))
  * model_mask + user_audio_proj(scale(user_latent)) * user_mask`
  (`haru.py:96-107`).  Compare `prompt.embed` and every `frame{t}.embed`.
* **LM trunk.**  Compare per-layer outputs (`prompt.lm_layer*`, `frame1..3.lm_layer*`)
  and the final normalized hidden (`prompt.hidden_last`, `frame{t}.hidden`).
  `logits = hidden @ embed_tokens.weight.T` in float32 (`haru.py:113-115`).
  `logits_fp32_pad_eop` exists because the PAD/EOP decision is recomputed in
  float32 at those two ids (`haru.py:117-123`).
* **Head.**  `cond = cond_proj(hidden)`, start at the recorded `head_noise`,
  run one MeanFlow step with `h = 1`, guidance 1.5 (the guidance scalar is part
  of the network input, `meanflow_head.py:106-113,130-134`).  Compare
  `head_latent_normalized`, then unscale to `latent_out`.
* **Sampling.**  `frame{t}.logits` is the raw `model.logits` output;
  `frame{t}.logits_final` is that vector after the DRY penalties from the
  recorded token history (`sampling.py:33-77`) and after replacing the PAD/EOP
  entries with `logits_fp32_pad_eop`.  A sampler should consume
  `logits_final`, divide by temperature and top-k the result.  Compare the
  recorded `token_out`; the engine should replay tokens, not the multinomial
  RNG.
* **Prompt assembly.**  Compare `prompt.tokens`, `prompt.model_latents`,
  `prompt.user_latents`, `prompt.model_mask` and `prompt.user_mask` directly.

`golden_meta.json` also records the frame tokens, all sampling settings, seeds,
the reference commit and sha256 hashes of the checkpoint files, the state dict
and the golden tensor set.

## Noise floor

Measured, not assumed.  `check_haru_golden.py` replays the recorded draws with
`torch.set_num_threads(1)` and again with `torch.set_num_threads(4)`, and the
same comparison was run over the complete generator (`run_generation`), reusing
the same seed and therefore the same random draws.  Maximum absolute difference
per tensor family, 1 thread vs 4 threads:

| family | max abs diff |
|---|---|
| codec_decode (`audio_out`) | 0.000000e+00 |
| codec_encode (`*_codec_mean`) | 0.000000e+00 |
| codec_debug (stage outputs) | 0.000000e+00 |
| projectors (`*_embed`) | 0.000000e+00 |
| latents | 0.000000e+00 |
| lm (hidden and layer outputs) | 0.000000e+00 |
| logits | 0.000000e+00 |
| head | 0.000000e+00 |
| tokens | identical (all 25 tokens) |

So the measured floor for this reference on this CPU build is exactly zero:
these tensors are small enough that torch does not split the reductions across
threads in a way that changes rounding, and the sampled tokens do not change.
There is no positive tolerance to hand out from this measurement.  That does
**not** mean an independent C++ implementation must be bit-identical; it means
any difference a C++ test sees is not caused by thread scheduling in this
reference.  If a component needs a numerical tolerance, measure it against the
C++ build's own arithmetic order and record that separately.

The check output for the committed files is:

```
PASS A byte-identical haru_tiny/config.json
PASS A byte-identical haru_tiny/model.safetensors
PASS A byte-identical haru_tiny/tokenizer.json
PASS A byte-identical golden.safetensors
PASS A byte-identical golden_meta.json
PASS A size under 4 MB
PASS B real DuplexSession tokens match: 25 frames
PASS B real DuplexSession audio matches: max |diff| 0.000e+00
PASS C checkpoint name patterns equal real header: 103 patterns; missing=[] extra=[]
PASS C checkpoint shapes scale down from real
PASS C config geometry
PASS D recorded noise/tokens replay through components: 281 tensors reproduced
PASS D recorded noise reconstructs latents
RESULT PASS (all checks)
```

## Checkpoint-name validation

`check_haru_golden.py` replaces `.layers.<n>` with `.layers.N` and
`.stages.<n>.<m>` with `.stages.N.M` in both the tiny state dict and
`~/haru-golden/real/header.json`.  The 103 name patterns are equal: no missing
and no extra patterns.  Counts differ where the tiny model intentionally has
fewer layers/blocks (4 LM layers vs 32; 7 codec blocks per stack vs 26), and all
tiny shapes are no larger than the corresponding real shapes.  Non-persistent
buffers (`ScalarEmbed.freqs`, `Qwen3_5TextRotaryEmbedding.inv_freq`) are absent
from both state dicts, which is why the sets match without exceptions.
