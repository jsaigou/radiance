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
check_haru_golden.py     five PASS/FAIL checks; exits non-zero on failure
check_haru_names.py      block-declaration coverage against tiny and real headers
measure_sensitivity.py   float64 and bfloat16 sensitivity over the same goldens
golden.safetensors       486 float32/int64 tensors (1,110,964 bytes of payload)
golden_meta.json         sampling config, seeds, reference commit, hashes
haru_tiny/config.json    HF-style Haru config (same schema as the real one)
haru_tiny/model.safetensors  model.state_dict(), float32, real key names
haru_tiny/tokenizer.json small byte-level BPE tokenizer, 512 ids
real/config.json         the real checkpoint's config
real/header.json         the real checkpoint's 1022-tensor safetensors header
```

`check_haru_names.py` parses the weight declarations in `arch/haru_common/`, expands the layer and
stage loops with each config, and checks names, shapes and dtypes against both the tiny header and
`real/header.json`.  Run it with `python3 tests/data/haru/check_haru_names.py`; `--self-test` mutates
one name, shape and dtype in memory and confirms each is reported.  `real/` holds only the header
and config (about 112 KB), not the real weights.

Total size of this directory is below 5 MB (about 4.0 MB at the time of writing, including the
real header/config copy).

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
PASS E sensitivity measurement reproduces README: 27 families, rel_floor=1e-03
RESULT PASS (all checks)
```

## Numerical sensitivity (H0b)

A zero run-to-run noise floor says nothing about a different operation order.
To give C++ authors a starting point, `measure_sensitivity.py` replays exactly
the same recorded noise and sampled tokens through the reference with the
model parameters and float inputs in float64, and compares every derived tensor
with the committed float32 goldens.  This is not a new golden: it is a
sensitivity probe over the existing data.

Honest float64 limitations of the unmodified reference: several pieces
explicitly compute in float32 even when their inputs are float64.  These are
`Qwen3_5RMSNorm` and `Qwen3_5RMSNormGated` (`x.float()` / `weight.float()`),
the codec `RMSNorm`, `MeanFlowHead._rms_norm`, the `ScalarEmbed` sinusoidal
features (`.float()`), and the final casts in `HaruModel.logits`,
`logits_fp32`, `unscale_latents` and the decoder's `.float().clamp()`.
Convolution, attention, the recurrent gated-delta-net path and the remaining
matmuls do run in float64.  The causal-conv1d and gated-delta-rule fallbacks
accepted float64 and bfloat16 without a dtype assertion.  The float64 run is
therefore the nearest honest alternative the reference supports, and its errors
are a lower bound on what a fully float64 oracle would show.

Run it with:

```sh
cd ~/haru-golden
PYTHONPATH=Haru/haru venv/bin/python \
  <worktree>/tests/data/haru/measure_sensitivity.py --table --bf16-table
```

Definitions used below: `max rel` is `max |error| / max(|ref|, 0.001)`
(denominator floored at 0.001 in the tensor's own units).  P1..P4 are
the maximum of `|error| / (atol + rtol * |ref|)` for
P1=(1e-6, 1e-6), P2=(1e-5, 1e-5), P3=(1e-4, 1e-4), P4=(1e-6, 1e-4); a value
below 1 means that pair would pass.

### float32 goldens vs float64 replay

| family | fp64 max abs | fp64 max rel | P1 | P2 | P3 | P4 | recommended abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `codec_encode_mean` | 8.251e-07 | 2.384e-04 | 7.482e-01 | 7.482e-02 | 7.482e-03 | 2.977e-01 | 9.000e-06 |
| `codec_decode_audio` | 1.350e-07 | 1.133e-04 | 1.343e-01 | 1.343e-02 | 1.343e-03 | 1.050e-01 | 2.000e-06 |
| `codec_dbg.enc_stage0` | 6.752e-08 | 3.270e-05 | 6.700e-02 | 6.700e-03 | 6.700e-04 | 4.371e-02 | 7.000e-07 |
| `codec_dbg.enc_stage1` | 1.490e-07 | 8.357e-05 | 1.272e-01 | 1.272e-02 | 1.272e-03 | 8.178e-02 | 2.000e-06 |
| `codec_dbg.enc_stage2` | 7.451e-08 | 2.715e-05 | 6.428e-02 | 6.428e-03 | 6.428e-04 | 2.624e-02 | 8.000e-07 |
| `codec_dbg.enc_stage3` | 1.490e-07 | 6.950e-05 | 1.188e-01 | 1.188e-02 | 1.188e-03 | 7.824e-02 | 2.000e-06 |
| `codec_dbg.enc_stage4` | 2.086e-07 | 5.115e-05 | 1.848e-01 | 1.848e-02 | 1.848e-03 | 6.601e-02 | 3.000e-06 |
| `codec_dbg.enc_stage5` | 2.831e-07 | 2.642e-05 | 2.373e-01 | 2.373e-02 | 2.373e-03 | 1.104e-01 | 3.000e-06 |
| `codec_dbg.enc_stage6` | 3.129e-07 | 1.358e-05 | 2.881e-01 | 2.881e-02 | 2.881e-03 | 7.548e-02 | 4.000e-06 |
| `codec_dbg.dec_stage0` | 1.788e-07 | 3.543e-06 | 9.127e-02 | 9.127e-03 | 9.127e-04 | 2.340e-02 | 2.000e-06 |
| `codec_dbg.dec_stage1` | 2.384e-07 | 6.119e-05 | 1.839e-01 | 1.839e-02 | 1.839e-03 | 7.887e-02 | 3.000e-06 |
| `codec_dbg.dec_stage2` | 2.012e-07 | 9.811e-05 | 1.800e-01 | 1.800e-02 | 1.800e-03 | 9.421e-02 | 3.000e-06 |
| `codec_dbg.dec_stage3` | 2.757e-07 | 6.301e-05 | 2.503e-01 | 2.503e-02 | 2.503e-03 | 9.138e-02 | 3.000e-06 |
| `codec_dbg.dec_stage4` | 5.141e-07 | 1.529e-04 | 4.686e-01 | 4.686e-02 | 4.686e-03 | 1.608e-01 | 6.000e-06 |
| `codec_dbg.dec_stage5` | 6.780e-07 | 2.016e-04 | 6.109e-01 | 6.109e-02 | 6.109e-03 | 3.256e-01 | 7.000e-06 |
| `codec_dbg.dec_stage6` | 8.270e-07 | 1.338e-04 | 8.035e-01 | 8.035e-02 | 8.035e-03 | 3.162e-01 | 9.000e-06 |
| `projector_embed` | 9.816e-07 | 5.170e-04 | 8.337e-01 | 8.337e-02 | 8.337e-03 | 5.115e-01 | 1.000e-05 |
| `lm_layer0` | 7.025e-07 | 6.511e-05 | 6.101e-01 | 6.101e-02 | 6.101e-03 | 1.073e-01 | 8.000e-06 |
| `lm_layer1` | 7.462e-07 | 4.295e-04 | 5.695e-01 | 5.695e-02 | 5.695e-03 | 4.190e-01 | 8.000e-06 |
| `lm_layer2` | 8.142e-07 | 1.971e-05 | 5.987e-01 | 5.987e-02 | 5.987e-03 | 1.041e-01 | 9.000e-06 |
| `lm_layer3` | 8.809e-07 | 5.518e-05 | 6.367e-01 | 6.367e-02 | 6.367e-03 | 1.633e-01 | 9.000e-06 |
| `lm_hidden` | 1.907e-06 | 1.779e-04 | 1.035e+00 | 1.035e-01 | 1.035e-02 | 4.104e-01 | 2.000e-05 |
| `logits` | 1.073e-06 | 7.655e-04 | 9.792e-01 | 9.792e-02 | 9.792e-03 | 7.645e-01 | 2.000e-05 |
| `head_output` | 2.500e-07 | 4.601e-05 | 2.041e-01 | 2.041e-02 | 2.041e-03 | 7.251e-02 | 3.000e-06 |
| `latent_out` | 4.768e-07 | 4.886e-05 | 2.603e-01 | 2.603e-02 | 2.603e-03 | 1.162e-01 | 5.000e-06 |
| `user_latent` | 7.004e-07 | 2.707e-04 | 6.638e-01 | 6.638e-02 | 6.638e-03 | 2.657e-01 | 8.000e-06 |
| `prompt_tensors` | 8.186e-07 | 5.491e-04 | 7.663e-01 | 7.663e-02 | 7.663e-03 | 5.425e-01 | 9.000e-06 |

Recommended starting tolerance is the measured float64-vs-float32 `max abs`
times 10, rounded up to one significant digit (rightmost column).  No family
measured exactly zero here, so every row uses that rule.  The script keeps a
**1e-6 absolute** fallback for an exactly zero family: two identical torch runs
only prove the same operation order twice, not a bound on a different C++
order, so zero must not become a zero tolerance.

### bfloat16 replay vs float32 goldens (informational)

The same recorded draws cast to bfloat16, with the model in bfloat16.  The real
checkpoint is bf16, so this is how much wider real-parity error can be than the
float32 numbers above.  These are not recommended C++ tolerances.

| family | bf16 max abs | bf16 max rel |
|---|---:|---:|
| `codec_encode_mean` | 2.337e-02 | 7.326e+00 |
| `codec_decode_audio` | 4.240e-03 | 2.941e+00 |
| `codec_dbg.enc_stage0` | 6.073e-03 | 3.893e+00 |
| `codec_dbg.enc_stage1` | 8.738e-03 | 3.342e+00 |
| `codec_dbg.enc_stage2` | 5.120e-03 | 1.180e+00 |
| `codec_dbg.enc_stage3` | 7.038e-03 | 2.944e+00 |
| `codec_dbg.enc_stage4` | 9.664e-03 | 2.585e+00 |
| `codec_dbg.enc_stage5` | 1.401e-02 | 2.809e+00 |
| `codec_dbg.enc_stage6` | 1.910e-02 | 5.106e-01 |
| `codec_dbg.dec_stage0` | 9.185e-03 | 9.625e-02 |
| `codec_dbg.dec_stage1` | 7.727e-03 | 5.162e-01 |
| `codec_dbg.dec_stage2` | 6.772e-03 | 2.061e+00 |
| `codec_dbg.dec_stage3` | 1.266e-02 | 2.923e+00 |
| `codec_dbg.dec_stage4` | 1.896e-02 | 8.644e+00 |
| `codec_dbg.dec_stage5` | 1.317e-02 | 4.901e+00 |
| `codec_dbg.dec_stage6` | 1.670e-02 | 8.929e+00 |
| `projector_embed` | 2.731e-02 | 1.012e+01 |
| `lm_layer0` | 2.567e-02 | 1.125e+00 |
| `lm_layer1` | 2.718e-02 | 1.583e+01 |
| `lm_layer2` | 2.897e-02 | 8.189e-01 |
| `lm_layer3` | 3.111e-02 | 2.137e+00 |
| `lm_hidden` | 7.188e-02 | 7.645e+00 |
| `logits` | 3.181e-02 | 2.760e+01 |
| `head_output` | 1.877e-02 | 6.703e-01 |
| `latent_out` | 2.668e-02 | 9.195e-01 |
| `user_latent` | 2.356e-02 | 1.612e+01 |
| `prompt_tensors` | 1.815e-02 | 1.306e+01 |

<!-- BEGIN H0B SENSITIVITY JSON -->
```json
{"families":{"codec_dbg.dec_stage0":{"bf16_max_abs":0.009184598922729492,"bf16_max_rel":0.09624700350959987,"fp64_max_abs":1.7881393432617188e-07,"fp64_max_rel":3.5433091979710054e-06,"fp64_norm":[0.09126564365384195,0.009126564365384194,0.0009126564365384193,0.02340154021297395],"recommended_abs":2e-06},"codec_dbg.dec_stage1":{"bf16_max_abs":0.007727444171905518,"bf16_max_rel":0.5161687731742859,"fp64_max_abs":2.384185791015625e-07,"fp64_max_rel":6.118898454587907e-05,"fp64_norm":[0.1838531943274487,0.01838531943274487,0.001838531943274487,0.07886754482857722],"recommended_abs":3e-06},"codec_dbg.dec_stage2":{"bf16_max_abs":0.0067721158266067505,"bf16_max_rel":2.061261795461178,"fp64_max_abs":2.0116567611694336e-07,"fp64_max_rel":9.810901246964931e-05,"fp64_norm":[0.17996073256815365,0.017996073256815362,0.0017996073256815362,0.09421498850333651],"recommended_abs":3e-06},"codec_dbg.dec_stage3":{"bf16_max_abs":0.012655586004257202,"bf16_max_rel":2.9233237728476524,"fp64_max_abs":2.7567148208618164e-07,"fp64_max_rel":6.300979293882847e-05,"fp64_norm":[0.2503112349599987,0.02503112349599987,0.002503112349599987,0.09137872793805779],"recommended_abs":3e-06},"codec_dbg.dec_stage4":{"bf16_max_abs":0.018958523869514465,"bf16_max_rel":8.644196903333068,"fp64_max_abs":5.140900611877441e-07,"fp64_max_rel":0.00015291152521967888,"fp64_norm":[0.46856362181643363,0.04685636218164336,0.0046856362181643355,0.16077049828817058],"recommended_abs":6e-06},"codec_dbg.dec_stage5":{"bf16_max_abs":0.013171255588531494,"bf16_max_rel":4.9010569782522495,"fp64_max_abs":6.780028343200684e-07,"fp64_max_rel":0.00020160939985561528,"fp64_norm":[0.6109400538636977,0.06109400538636976,0.006109400538636977,0.32557626424224534],"recommended_abs":7e-06},"codec_dbg.dec_stage6":{"bf16_max_abs":0.016700856387615204,"bf16_max_rel":8.929077535867691,"fp64_max_abs":8.270144462585449e-07,"fp64_max_rel":0.00013384635582594493,"fp64_norm":[0.803493697730224,0.0803493697730224,0.00803493697730224,0.31618129274522117],"recommended_abs":9e-06},"codec_dbg.enc_stage0":{"bf16_max_abs":0.006072931457310915,"bf16_max_rel":3.8927451241761446,"fp64_max_abs":6.752088665962219e-08,"fp64_max_rel":3.2698153518140316e-05,"fp64_norm":[0.06700428084389241,0.006700428084389241,0.000670042808438924,0.043709374115874736],"recommended_abs":7e-07},"codec_dbg.enc_stage1":{"bf16_max_abs":0.008738160133361816,"bf16_max_rel":3.341655363328755,"fp64_max_abs":1.4901161193847656e-07,"fp64_max_rel":8.357164915651083e-05,"fp64_norm":[0.12719790507085516,0.012719790507085512,0.0012719790507085514,0.08178127210798555],"recommended_abs":2e-06},"codec_dbg.enc_stage2":{"bf16_max_abs":0.00512019544839859,"bf16_max_rel":1.1803414672613144,"fp64_max_abs":7.450580596923828e-08,"fp64_max_rel":2.7153873816132545e-05,"fp64_norm":[0.06428341064377714,0.006428341064377715,0.0006428341064377714,0.026238279780571968],"recommended_abs":8e-07},"codec_dbg.enc_stage3":{"bf16_max_abs":0.00703766942024231,"bf16_max_rel":2.9441751539707184,"fp64_max_abs":1.4901161193847656e-07,"fp64_max_rel":6.949994713068008e-05,"fp64_norm":[0.11877214751221259,0.011877214751221256,0.0011877214751221255,0.07824387190103349],"recommended_abs":2e-06},"codec_dbg.enc_stage4":{"bf16_max_abs":0.009663760662078857,"bf16_max_rel":2.5854018814648136,"fp64_max_abs":2.086162567138672e-07,"fp64_max_rel":5.1149982027709484e-05,"fp64_norm":[0.1847709487593601,0.018477094875936007,0.0018477094875936006,0.06601200910084559],"recommended_abs":3e-06},"codec_dbg.enc_stage5":{"bf16_max_abs":0.014012917876243591,"bf16_max_rel":2.8094835579395294,"fp64_max_abs":2.8312206268310547e-07,"fp64_max_rel":2.641687938637029e-05,"fp64_norm":[0.23728462268010883,0.02372846226801088,0.0023728462268010884,0.11035297299130127],"recommended_abs":3e-06},"codec_dbg.enc_stage6":{"bf16_max_abs":0.0191020667552948,"bf16_max_rel":0.5106344319135008,"fp64_max_abs":3.129243850708008e-07,"fp64_max_rel":1.3578590179243391e-05,"fp64_norm":[0.28813934484577164,0.02881393448457716,0.002881393448457716,0.07548220746645262],"recommended_abs":4e-06},"codec_decode_audio":{"bf16_max_abs":0.004239553585648537,"bf16_max_rel":2.940748818218708,"fp64_max_abs":1.3504177331924438e-07,"fp64_max_rel":0.00011327210813760757,"fp64_norm":[0.13429032465249544,0.01342903246524954,0.001342903246524954,0.10495624541756503],"recommended_abs":2e-06},"codec_encode_mean":{"bf16_max_abs":0.023368075489997864,"bf16_max_rel":7.325809448957443,"fp64_max_abs":8.251130282910246e-07,"fp64_max_rel":0.0002384077137205565,"fp64_norm":[0.7481513383961461,0.0748151338396146,0.007481513383961461,0.29769721255824677],"recommended_abs":9e-06},"head_output":{"bf16_max_abs":0.018767356872558594,"bf16_max_rel":0.6703403940663686,"fp64_max_abs":2.500151432549558e-07,"fp64_max_rel":4.600582405274939e-05,"fp64_norm":[0.20409227798734025,0.020409227798734022,0.0020409227798734025,0.0725118452046983],"recommended_abs":3e-06},"latent_out":{"bf16_max_abs":0.02667689323425293,"bf16_max_rel":0.919470687479701,"fp64_max_abs":4.76837158203125e-07,"fp64_max_rel":4.886138599632677e-05,"fp64_norm":[0.26025874247504144,0.026025874247504136,0.0026025874247504136,0.11622979286936069],"recommended_abs":4.9999999999999996e-06},"lm_hidden":{"bf16_max_abs":0.0718846321105957,"bf16_max_rel":7.644648920348933,"fp64_max_abs":1.9073486328125e-06,"fp64_max_rel":0.00017790316014392157,"fp64_norm":[1.035002169105796,0.10350021691057959,0.010350021691057958,0.4103790077441212],"recommended_abs":2e-05},"lm_layer0":{"bf16_max_abs":0.025665760040283203,"bf16_max_rel":1.1246857220046442,"fp64_max_abs":7.024832759383592e-07,"fp64_max_rel":6.511332084437089e-05,"fp64_norm":[0.6101061548176938,0.06101061548176937,0.0061010615481769375,0.10732765422939539],"recommended_abs":8e-06},"lm_layer1":{"bf16_max_abs":0.02717500925064087,"bf16_max_rel":15.831660479307175,"fp64_max_abs":7.462438809113792e-07,"fp64_max_rel":0.0004294530722009915,"fp64_norm":[0.5694552268205753,0.05694552268205751,0.005694552268205752,0.41895772570592177],"recommended_abs":8e-06},"lm_layer2":{"bf16_max_abs":0.02896517515182495,"bf16_max_rel":0.818895458180716,"fp64_max_abs":8.141631893665568e-07,"fp64_max_rel":1.971409573076366e-05,"fp64_norm":[0.598720202225726,0.05987202022257259,0.005987202022257259,0.10410790755023946],"recommended_abs":9e-06},"lm_layer3":{"bf16_max_abs":0.031112849712371826,"bf16_max_rel":2.136624817751149,"fp64_max_abs":8.808559666118043e-07,"fp64_max_rel":5.517763913961896e-05,"fp64_norm":[0.6366912027624971,0.0636691202762497,0.00636691202762497,0.16328668004804073],"recommended_abs":9e-06},"logits":{"bf16_max_abs":0.03180765360593796,"bf16_max_rel":27.600690722465515,"fp64_max_abs":1.0728836059570312e-06,"fp64_max_rel":0.00076550441008294,"fp64_norm":[0.979159534000877,0.0979159534000877,0.00979159534000877,0.7645258095951681],"recommended_abs":2e-05},"projector_embed":{"bf16_max_abs":0.02730536460876465,"bf16_max_rel":10.123245418071747,"fp64_max_abs":9.816061558032274e-07,"fp64_max_rel":0.0005170330574521733,"fp64_norm":[0.8337490303839941,0.0833749030383994,0.008337490303839939,0.5115464952924312],"recommended_abs":9.999999999999999e-06},"prompt_tensors":{"bf16_max_abs":0.018153026700019836,"bf16_max_rel":13.060005381703377,"fp64_max_abs":8.185937702687163e-07,"fp64_max_rel":0.000549083885435786,"fp64_norm":[0.7663477796356586,0.07663477796356584,0.007663477796356585,0.5425434022852842],"recommended_abs":9e-06},"user_latent":{"bf16_max_abs":0.023560643196105957,"bf16_max_rel":16.124188899993896,"fp64_max_abs":7.003773085934029e-07,"fp64_max_rel":0.0002706551230080567,"fp64_norm":[0.6638236945883136,0.06638236945883134,0.0066382369458831345,0.26570518637954066],"recommended_abs":8e-06}},"norm_pairs":[[1e-06,1e-06],[1e-05,1e-05],[0.0001,0.0001],[1e-06,0.0001]],"rel_floor":0.001}
```
<!-- END H0B SENSITIVITY JSON -->

## Checkpoint-name validation

`check_haru_golden.py` replaces `.layers.<n>` with `.layers.N` and
`.stages.<n>.<m>` with `.stages.N.M` in both the tiny state dict and
`~/haru-golden/real/header.json`.  The 103 name patterns are equal: no missing
and no extra patterns.  Counts differ where the tiny model intentionally has
fewer layers/blocks (4 LM layers vs 32; 7 codec blocks per stack vs 26), and all
tiny shapes are no larger than the corresponding real shapes.  Non-persistent
buffers (`ScalarEmbed.freqs`, `Qwen3_5TextRotaryEmbedding.inv_freq`) are absent
from both state dicts, which is why the sets match without exceptions.
