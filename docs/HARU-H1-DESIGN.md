# Haru H1 design note: the VibeVoice codec as a streaming Radiance block

Status: design note for review before any C++ is written. Scope is `arch/haru_common/haru_codec.{h,cpp}`
(not created yet), its test plugin under `tests/haru_plugins/codec/`, and `tests/haru_codec_test.cpp`.
Nothing here is implemented or built: this host is macOS and cannot build the Linux x86-64 C++ tree. I did
run the pinned Python reference read-only against the tiny checkpoint to check the streaming equivalences in
§2.3; that is reference evidence, not a C++ test result, and no C++ test has passed.

Reference: `~/haru-golden/Haru/haru/haru` at commit `7fe6b7665f8bf035998f401f8cb779fbe53f2751`
(`tests/data/haru/README.md:8-13`). Tiny checkpoint: `tests/data/haru/haru_tiny`. Real geometry:
`~/haru-golden/real/config.json`; real tensor shape list: `~/haru-golden/real/header.json`.

Citation paths: `vibevoice_codec.py`, `config.py`, `prompt.py`, `session.py`, `models/haru.py` and
`codec_triton/...` are relative to `~/haru-golden/Haru/haru/haru`; `libref/`, `libhalo/`, `arch/`,
`abi/`, `core/`, `docs/` and `tests/` are relative to this worktree.

---

## 1. What the codec is

### 1.1 The two stacks

`VibeVoiceCodec` owns one `Encoder` and one `Decoder`, each a `_Stack` of stateful layers
(`models/vibevoice_codec.py:181-186`, `:98-114`). Both are causal convolution stacks over time; the only
non-convolution layers are a per-channel RMSNorm, a depthwise causal convolution, a two-linear GELU FFN and
per-channel layer scales (`:20-95`).

* Encoder: `[B, 1, T]` audio -> `[B, ceil(T/3200), 64]` latent means (`:117-146`, `:202-208`).
* Decoder: `[B, T, 64]` latents -> `[B, T*3200]` audio (`:149-178`, `:210-215`).

`frame_samples = 3200` is the product of the ratios `(8,5,5,4,2,2)` and is checked against the config
(`config.py:28-33`, `:89-90`). `latent_dim = 64`, `norm_eps = 1e-5`, `latent_std = 0.5`
(`config.py:20-26`; tiny `tests/data/haru/haru_tiny/config.json`; real `~/haru-golden/real/config.json`).

### 1.2 Every layer, in order

The `_Stack.sequence()` order is the forward order (`vibevoice_codec.py:135-142`, `:167-174`):

Encoder (7 resolution stages, stage `i` has `encoder_depths[i]` blocks):

```
stem
stage 0 blocks (C = base<<0)
down[0]                       # stride ratios[::-1][0]
stage 1 blocks (C = base<<1)
...
down[5]                       # stride ratios[::-1][5]
stage 6 blocks (C = base<<6)
head
```

Decoder:

```
stem
stage 0 blocks (C = base<<6)
up[0]                         # stride ratios[0]
stage 1 blocks (C = base<<5)
...
up[5]
stage 6 blocks (C = base<<0)
head
```

Every non-block convolution is a `CausalConv1d` with `kernel`, optional `stride`, optional `groups`
(`vibevoice_codec.py:33-52`):

* stem and all four heads: kernel 7, stride 1, `groups = 1`;
* encoder `down[i]`: kernel `2 * s`, stride `s`, `groups = 1`, where `s = ratios[::-1][i]`;
* decoder `up[i]`: a `CausalConvTranspose1d` with kernel `2 * s`, stride `s`, `groups = 1`
  (`:55-72`).

Every `Block` (`vibevoice_codec.py:75-95`) is, in order:

1. `RMSNorm(dim, eps)` over the channel axis, computed in float32 and multiplied by its gain
   (`:20-30`);
2. depthwise **causal** `CausalConv1d(dim, dim, 7, groups=dim)`, stride 1, with bias
   (`:81`);
3. residual `x = x + y * gamma[:, None]` with a per-channel `gamma` (`:82`, `:93`);
4. `RMSNorm(dim, eps)`;
5. `fc1: Linear(dim, 4*dim)` (bias), exact GELU (`F.gelu`, the erf form), `fc2: Linear(4*dim, dim)`
   (bias) (`:83-85`, `:94`);
6. residual `x = x + (h * ffn_gamma)` with a per-channel `ffn_gamma` (`:86`, `:95`).

There is no attention, no skip connection beyond these two residual adds, and no other normalisation.
The transposed convolution has no dilation and no groups; its state is the overlap tail
(`:55-72`).

### 1.3 Tiny and real shapes

Tiny uses `base_channels = 1` and one block per stage (`tests/data/haru/README.md:91-98`). Real uses
`base_channels = 32`, encoder depths `(3,3,3,3,3,3,8)` and decoder depths `(8,3,3,3,3,3,3)`
(`~/haru-golden/real/config.json`).

Per-stage channel counts and block depths:

| stage | tiny C | real C | tiny depth | real depth (enc / dec) |
|---|---:|---:|---:|---:|
| 0 | 1 | 32 | 1 | 3 / 8 |
| 1 | 2 | 64 | 1 | 3 / 3 |
| 2 | 4 | 128 | 1 | 3 / 3 |
| 3 | 8 | 256 | 1 | 3 / 3 |
| 4 | 16 | 512 | 1 | 3 / 3 |
| 5 | 32 | 1024 | 1 | 3 / 3 |
| 6 | 64 | 2048 | 1 | 8 / 3 |

Convolution geometry (both configs; only channels and block depth change):

| layer | weight shape | kernel | stride | context (= k - stride) |
|---|---|---:|---:|---:|
| encoder stem | `[base, 1, 7]` | 7 | 1 | 6 |
| encoder down 0 | `[2b, b, 4]` | 4 | 2 | 2 |
| encoder down 1 | `[4b, 2b, 4]` | 4 | 2 | 2 |
| encoder down 2 | `[8b, 4b, 8]` | 8 | 4 | 4 |
| encoder down 3 | `[16b, 8b, 10]` | 10 | 5 | 5 |
| encoder down 4 | `[32b, 16b, 10]` | 10 | 5 | 5 |
| encoder down 5 | `[64b, 32b, 16]` | 16 | 8 | 8 |
| encoder head | `[64, 64b, 7]` | 7 | 1 | 6 |
| block depthwise | `[C, 1, 7]` | 7 | 1 | 6 |
| decoder stem | `[64b, 64, 7]` | 7 | 1 | 6 |
| decoder up 0 | `[64b, 32b, 16]` | 16 | 8 | (transposed) |
| decoder up 1 | `[32b, 16b, 10]` | 10 | 5 | (transposed) |
| decoder up 2 | `[16b, 8b, 10]` | 10 | 5 | (transposed) |
| decoder up 3 | `[8b, 4b, 8]` | 8 | 4 | (transposed) |
| decoder up 4 | `[4b, 2b, 4]` | 4 | 2 | (transposed) |
| decoder up 5 | `[2b, b, 4]` | 4 | 2 | (transposed) |
| decoder head | `[1, b, 7]` | 7 | 1 | 6 |

`b = base_channels` (1 tiny, 32 real). These shapes agree with the real header: e.g.
`codec.encoder.stem.weight [32,1,7]`, `codec.encoder.down.0.weight [64,32,4]`, ...
`codec.encoder.head.weight [64,2048,7]`, `codec.decoder.stem.weight [2048,64,7]`,
`codec.decoder.up.0.weight [2048,1024,16]`, `codec.decoder.head.weight [1,32,7]`
(`~/haru-golden/real/header.json`); and with the tiny checkpoint, e.g. encoder stem `[1,1,7]`,
down 0 `[2,1,4]`, encoder head `[64,64,7]`, decoder up 0 `[64,32,16]`
(`tests/data/haru/haru_tiny/model.safetensors`).

Block weight shapes: `conv.weight [C,1,7]`, `conv.bias [C]`, `norm.weight [C]`, `gamma [C]`,
`ffn_norm.weight [C]`, `fc1.weight [4C,C]`, `fc1.bias [4C]`, `fc2.weight [C,4C]`, `fc2.bias [C]`,
`ffn_gamma [C]`. Full real codec: 26 encoder blocks + 26 decoder blocks, 68 convolutions in total
(8 non-block each direction + 26 blocks each direction).

Scale, measured from the headers, matters:

| quantity | tiny | real |
|---|---:|---:|
| codec parameters | 228,299 | 687,392,001 |
| codec weight bytes | 913,196 B (f32) | 1,374,784,002 B (bf16) |
| encoder / decoder weights | ~0.46 MB each (f32) | ~687 MB each (bf16) |
| MAC per frame (encode + decode, from the shapes) | negligible | ~4.44 GMAC (8.88 GFLOP) |
| 7.5 frames/s | - | ~66.6 GFLOP/s |

The real codec is **not** a small side model: it is 1.37 GB of bf16 weights, a material fraction of a
4B-parameter bf16 LM's weight traffic, and ~4.44 GMAC per 3200-sample frame (~8.9 GFLOP; the two FFN
linears dominate, not the convolutions). The tiny test exercises geometry and names, not scale.

---

## 2. Streaming semantics

### 2.1 State per layer

The reference states are plain tensors with a batch row, initialised to **zeros** (`_Stack.init_state`,
`vibevoice_codec.py:104-107`; `CodecStream.__init__`, `:221-225`; `reset`, `:227-233`). Nothing is
replicated from the first frame.

* `CausalConv1d.state_shape() = (in_channels, context)`, `context = kernel - stride`
  (`:41-44`), holding the last `context` **input** rows.
* `CausalConvTranspose1d.state_shape() = (out_channels, stride)` (`:64-65`), holding the overlap
  tail of the previous full output.

The Triton backend states the same thing in the form this block should use: one tensor
`[B, context, C_in]` per layer, `context = kernel - stride` for a conv, 6 for a block's depthwise conv,
and **1 for a transposed conv**, holding the last input row (`codec_triton/stream.py:8-12`, `:29-41`).
It also stores the last input frame for a transposed conv in `_keep_context`
(`codec_triton/kernels.py:464-497`).

Per-layer state for this model (context in input rows; `b` = base channels):

| layer | tiny state | real state |
|---|---|---|
| encoder stem | `[1,6]` | `[1,6]` |
| encoder down 0 | `[1,2]` | `[32,2]` |
| encoder down 1 | `[2,2]` | `[64,2]` |
| encoder down 2 | `[4,4]` | `[128,4]` |
| encoder down 3 | `[8,5]` | `[256,5]` |
| encoder down 4 | `[16,5]` | `[512,5]` |
| encoder down 5 | `[32,8]` | `[1024,8]` |
| encoder block stage i | `[b<<i, 6]` each | `[b<<i, 6]` each |
| encoder head | `[64,6]` | `[2048,6]` |
| decoder stem | `[64,6]` | `[64,6]` |
| decoder up i | last **input** row `[b<<(6-i), 1]` | last input row `[b<<(6-i), 1]` |
| decoder block stage i | `[b<<(6-i), 6]` each | `[b<<(6-i), 6]` each |
| decoder head | `[b,6]` | `[32,6]` |

The reference decoder's `[out_channels, stride]` state is *equivalent* to the Triton/our `[in_channels,1]`
form: the overlap tail is `x_last @ W[:, :, stride:2*stride]` for `kernel = 2*stride`
(verified below). Storing the input row is smaller or equal for every up layer
(e.g. real up 0: 2048 floats instead of 8192).

Total persistent state per lane, from the tables: **2,828 floats tiny** (11 KB f32) and
**298,822 floats real** (1.20 MB f32 / 0.60 MB bf16).

### 2.2 How the reference pads a partial hop

`CausalConv1d.forward` right-pads its *input* to a whole number of strides before concatenating the
state (`vibevoice_codec.py:46-52`):

```python
tail = x.shape[-1] % self.stride
if tail:
    x = F.pad(x, (0, self.stride - tail))
x = torch.cat([state, x], dim=-1)
y = F.conv1d(x, self.weight, self.bias, stride=self.stride, groups=self.groups)
```

This is a per-conv right pad of the **new input**, not a pad of the audio. The streaming wrapper refuses
a partial frame (`CodecStream.encode`, `:235-242`); the whole-clip `encode` takes any length and thus
exercises this (`:202-208`). The Triton backend floors `t_out = t_in // stride` and therefore requires
whole frames (`codec_triton/kernels.py:430-462`, `codec_triton/stream.py:180-184`).

The 44,000-sample reference clip is 13.75 frames; `reference_frames` uses `ceil`
(`prompt.py:57-58`), so the prompt gets 14 latents and the last is produced through this padding
(`tests/data/haru/README.md:121-124`).

### 2.3 Streaming == whole clip for whole frames

The reference's own contract is that the same code gives identical results whole-clip or frame-by-frame
(`vibevoice_codec.py:1-7`). For a clip that is an exact multiple of 3200, the streamed encode is the same
arithmetic with the state carrying the left context. For a clip with a partial last frame, **the streamed
path must also support the partial last frame and apply the per-conv right pad of §2.2**; padding the
audio with zeros to a whole frame is not equivalent, because the padded zeros pass through every biased
convolution and produce nonzero rows where the reference inserts zeros at the final strided conv.

I checked this with the pinned reference on CPU in float32, tiny checkpoint, a seeded random 44,000-sample
clip:

| comparison | max abs difference |
|---|---:|
| 10 full frames streamed vs whole clip of 32,000 | 7.15e-07 |
| 13 full + one 2,400-sample partial, per-conv pad, vs whole 44,000 | 8.34e-07 |
| 14 full frames with the last 800 samples zero **padded as audio**, vs whole 44,000 | **9.13e-02** |
| decoder: 5 latents whole vs one-at-a-time streamed | 1.34e-07 |

So: exact multiples agree to f32 rounding (the reference docstring's "identical" is not bitwise across
these call shapes, but the difference is f32 noise); a partial last frame agrees only with the per-conv
pad; the naive audio pad is wrong by ~0.09 on the last latent. The C++ block should therefore expose one
streaming path that takes the true number of new samples (0..3200) and applies the §2.2 pad per conv,
plus a flag/length so the caller can say "the last frame is 2,400 samples", not infer it from a zero
tail. The duplex driver knows the true length as `RadDuplexBatch::ref_samples` and `ref_cu`
(`abi/rad_runtime.h:673-682`; `core/gen/duplex_driver.cpp:768-780`), and the test knows the golden's
44,000.

### 2.4 What the goldens contain

* `prompt.ref_codec_mean [14,64]`: the whole-clip `codec.encode(prompt.ref_audio)` (`:492`, README
  `:162-163`).
* `frame{t}.user_codec_mean [1,64]`: the **streaming** encode of each 3200-sample user frame, in order,
  from a codec stream reset before the prompt (`gen_haru_tiny.py:468-469`, `:599-604`).
* `frame{t}.audio_out [3200]`: the streaming decode of the model's raw latent, then clamp to [-1,1]
  (`gen_haru_tiny.py:548-571`; `session.py:159-163`).
* `codec_dbg.enc_stage0..6` and `dec_stage0..6`: one frame each on **fresh state**, captured after the
  *last block* of each stage (`gen_haru_tiny.py:394-432`). Encoder input is `frame1.user_audio`;
  decoder input is `frame0.latent_out` (`gen_haru_tiny.py:637`; README `:197-208`).

The goldens do not contain a same-clip whole-vs-stream pair for the user clip, so the reference A/B
above is the evidence for that claim, not a golden.

---

## 3. Reference operation -> Radiance op

The exact signatures below are the schemas in `docs/OPS.md` and the libref registry
(`libref/ref_registry.cpp`), not a restatement. `host` is libref; `device` is the fork's libhalo
(`libhalo/halo_rows.cpp`) and, for the upstream tree, libr4d (`docs/KERNELS.md:910-1167`). There is **no
`docs/radiance-op-coverage*.md` in this tree**; the completeness table in `docs/KERNELS.md` is the only
op-by-library table, and it covers libavx/libr4d/libref, not libhalo. I checked libhalo's rows directly
in `libhalo/halo_rows.cpp`.

### 3.1 The mapping

| reference operation | Radiance op | exact parameters (from `docs/OPS.md`) | operands | libref | libhalo |
|---|---|---|---|---|---|
| regular causal conv (stem, down, head) | `conv1d` | `M`(range) `cin` `cout` `k` `dil` `act`/opt `dtype` `causal`/opt `stride`/opt `pad`/opt `cu_div`/opt | `x`, `w`w, `cu_seqlens`, `res`?, `y`; `b`w? | yes (`ref_audio.cpp:62-126`) | `conv1d_f32`, `conv1d_bf16` (`halo_rows.cpp:9501-9521`) |
| depthwise causal conv (Block) | `conv1d_dw` | `M`(range) `n` `k` `cu_div`/opt `dtype` | `x`, `w`w, `cu_seqlens`, `y`; `b`w? | yes (`ref_elementwise.cpp:833-868`) | `conv1d_dw_bf16` only (`halo_rows.cpp:5794-5800`) |
| causal transposed conv (up) | `conv_t1d` | `M`(range) `cin` `cout` `stride` `dtype` `pad`/opt `k`/opt | `x`, `w`w, `b`w, `cu_seqlens`, `y` | yes (`ref_audio.cpp:139-173`) | `conv_t1d_f32`, `conv_t1d_bf16` (`halo_rows.cpp:9522-9540`) |
| RMSNorm | `rmsnorm` | `M`(range) `n` `eps`:f64 `dtype` `wadd`:f64/opt `mult`:f64/opt | `x`, `w`w, `y` | yes (`ref_elementwise.cpp:23-53`) | `rmsnorm_bf16` only (`halo_rows.cpp:5855-5863`) |
| Linear | `gemm_nt_bias` | `M`(range) `N` `K` `dtype` `act`/opt | `a`, `b`w, `bias`w, `res`?, `y` | yes (`ref_registry.cpp:1540`) | `gemm_f32_nt_bias`, bf16 tiled (`halo_rows.cpp:5911-5940`) |
| exact GELU | `gelu` | `M`(range) `n` `dtype` | `x`, `y` | yes | `gelu_bf16` only (`halo_rows.cpp:5759-5766`) |
| residual add | `add` | `M`(range) `n` `dtype` `div`:f64/opt | `a`, `b`, `y` | yes | `add_f32`, `add_bf16` (`halo_rows.cpp:5922-5940`) |
| layer scale `* gamma` | `mul_weight` | `M`(range) `n` `dtype` `scalar`/opt | `x`, `w`w, `y` | yes | `mul_weight_f32`, `mul_weight_bf16` (`halo_rows.cpp:5944-5967`) |
| copy for history/concat | `cast` (`from = to`) | `M`(range) `n` `from` `to` | `x`, `y` | yes (`ref_elementwise.cpp:1105-1125`) | `cast_f32_f32`, `cast_bf16_bf16` (`halo_rows.cpp:5657-5694`) |
| optional stateful depthwise conv | `causal_conv1d` | `M`(range) `n` `width` `dtype` | `x`, `w`w, `conv_state`:inout, `cu`, `cache_idx`?, `has_init`?, `num_accepted`? -> `y` | yes (`ref_ple.cpp:316-321`) | `causal_conv1d_bf16` (`halo_rows.cpp:5601-5607`) |
| optional stateful depthwise conv with bias+SiLU | `ple_conv_bias` | `M`(range) `n` `width` `dilation` `dtype` | `x`, `w`w, `conv_state`:inout, `cu`, `resid`?, `cache_idx`?, `has_init`?, `num_accepted`?, `b`w -> `y` | yes (`ref_ple.cpp:308-313`) | no libhalo row; not needed |
| gather/scatter for a ring history (alternative) | `scatter_rows`, `gather_rows`, `iota` | `M`(range) `n` `dtype` / `M`(range) `start` `step` `period`/opt | see `OPS.md` | yes | yes |
| latent sample (optional) | `mul`, `add`, `mul_weight` | `M`(range) `n` `dtype` | see §5 | yes | `add_f32`; `mul` bf16 only; `mul_weight_f32` |

`docs/KERNELS.md:943-957, 999-1005, 1077-1078, 1120, 1161` is the upstream completeness table for
`cast`, `causal_conv1d`, `conv1d`, `conv1d_dw`, `conv_t1d`, `gelu`, `gemm_nt_bias`, `mul`, `mul_weight`,
`rmsnorm`, `scatter_rows`: all are `host` and `device` there. The libhalo column above is narrower and is
the one that matters for this fork.

### 3.2 What the conv ops actually compute

`conv1d` (`docs/OPS.md:1553-1569`, `libref/ref_audio.cpp:48-126`): rows are samples, channels innermost.
`x` is `[rows, cin]`, `y` is `[rows_out, cout]`, `w` is PyTorch's `[cout, cin, k]`, `b` is `[cout]`.
Without `causal`, `k` must be odd and each side is padded `dil*(k-1)/2`; with `causal = 1`, all
`dil*(k-1)` is on the left. A causal stride `s > 1` pads `dil*(k-1)+1-s` on the left and keeps every
`s`-th row: output row `o` of a segment reads input rows `o*s - pad + t*dil`, `t = 0..k-1`, zeros
outside the segment. Every `cu_seqlens` bound must be a multiple of `s` (or `cu_div*s`); the op writes
`M/s` rows. It is **stateless**: before a segment's start it reads zeros, never a history tensor. It has
no `groups`, so it cannot serve the depthwise Block conv.

`conv1d_dw` (`docs/OPS.md:226, 448-450`, `libref/ref_elementwise.cpp:829-868`): causal, stride 1, no
dilation, depthwise. `y[t,c] = sum_j w[c,0,j] * x[t-(k-1)+j,c] + b[c]`; rows before the segment's
`cu[s]/cu_div` read as zeros; `b` optional. Stateless.

`conv_t1d` (`docs/OPS.md:1570-1578`, `libref/ref_audio.cpp:128-173`): `w` is PyTorch's `[cin, cout, k]`,
`b` `[cout]`; sequence `e`'s input rows `[cu[e],cu[e+1])` become `[s*cu[e], s*cu[e+1])`. With
`pad = 0` (the causal form) the last `k - stride` rows of the full output are dropped, so the output has
`s*L` rows. It is stateless and has no overlap-add state. In this model `k = 2s`, so the dropped tail is
exactly the next frame's overlap.

`causal_conv1d` (`docs/OPS.md:2638, 2891-2902`, `libref/ref_ple.cpp:201-321`) is the only conv op with a
persistent `conv_state`. It is depthwise, undilated, and **has no bias** (the libref passes a null `b`,
`ref_ple.cpp:316-321`). Its state is `[n_slots, n, state_len]`, slot `i` holding relative position
`i - (width-1)*dilation`, `has_init = 0` meaning zeros; without `num_accepted`, a step keeps the last
`width-1` inputs. `ple_conv_bias` has the bias but also applies SiLU, so it is not the Block's plain
conv+bias.

### 3.3 The gap: no conv op takes persistent history (except the bias-less depthwise one)

`conv1d`, `conv1d_dw` and `conv_t1d` are stateless. `causal_conv1d` has state but cannot apply the
Block's per-channel bias. Therefore **the existing ops cannot express a streaming causal convolution
with persistent history and bias as one op**. For v0 the block can compose them with explicit history
buffers and copies; the exact composition is below. This costs 2-3 extra `cast` launches per conv per
frame (and, for the transposed conv, no extra launch beyond the copy of the previous row). For 68 convs a
frame that is ~140-200 extra small launches; on libref it is cheap relative to the arithmetic, on device
it is launch overhead in the 100 ms budget and must be measured.

The smallest engine change that would remove the copies, if the reviewer prefers it:

* add optional `state`/`has_init`/`cache_idx` operands to `conv1d` and `conv_t1d`, exactly as
  `causal_conv1d`/`ple_conv` have them, with the state contract "read the `context` rows before the
  segment and write the window after it"; or
* add a bias operand to `causal_conv1d` (then the blocks use it directly, but the dense and transposed
  convs still need the same extension).

Either is a schema change in `docs/OPS.md` plus libref, libavx and libhalo/libr4d rows; it is larger than
the v0 plugin change and should be a separate decision. I recommend v0 composition and a measured launch
count.

### 3.4 The composition that is exact (and was checked against the reference)

For a `CausalConv1d` with kernel `k`, stride `s`, `context = k-s`:

1. keep the last `context` **input** rows in a per-lane history;
2. right-pad the new input of `L` rows to `L' = L + ((s - L%s) % s)` rows (the reference's §2.2 pad;
   for whole frames `L' = L`);
3. build `x' = [history(context); new(L')]` of `M = context + L'` rows (divisible by `s` for this
   model, where every strided conv has `k = 2s`, so `context = s`);
4. run `conv1d` with `causal = 1`, `stride = s`, `k`, `cu_seqlens = [0, M]`;
5. the new outputs are rows `[context/s, M/s)` (for `s=1`, rows `[k-1, k-1+L)`); discard the front;
6. the next history is the last `context` rows of `x'`.

The op's causal left pad before the history is what makes step 5 need the discard; the tail rows read
only real rows of `x'` and match the reference window exactly. I checked this in the reference with
Torch `F.conv1d(padding=k-s)` for the model's `(k,s)` pairs `(7,1)`, `(4,2)`, `(8,4)`, `(10,5)`,
`(16,8)`: max abs difference vs `CausalConv1d` 0.0-7e-7 (f32 order). For the Block's `conv1d_dw`, the
same composition without a stride: `x' = [history(6); new]`, take rows `[6, 6+L)`; `conv1d_dw` has the
bias, so no extra add.

For a `CausalConvTranspose1d` with `k = 2s`, store the previous **last input row** `[cin,1]` (the Triton
form):

1. build `x' = [prev(1); new(L)]` of `M = 1+L` rows;
2. run `conv_t1d` with `stride = s`, `k = 2s`, `pad = 0`, the real `b`, `cu_seqlens = [0, M]`;
3. the output has `M*s` rows; the current frame's output is rows `[s, s + L*s)`, the first `s` are
   discarded;
4. the next `prev` is the last input row.

This is exactly the Triton `conv_transpose` two-frame window (`codec_triton/kernels.py:464-497`), and it
includes the bias once per output row. I checked it against the reference `CausalConvTranspose1d` with
Torch `F.conv_transpose1d(x', w, stride=s) + bias` for `(cin,cout,s) = (5,7,8)`, `(4,3,5)` over three
frames: **bit-exact (0.0)**. The reference's `[cout,s]` state and this `[cin,1]` state are equivalent.

Limits of the composition: `cast` copies the whole operand extent, so a shared scratch buffer must use
`bcol_at`/`brow_slice` views with the exact row/column extents (`arch/common/rad_arch.h:1247-1295`,
`abi/rad_runtime.h:906-912`); `conv1d`'s strided form requires every segment bound to be a multiple of
`stride`, which step 2's right pad guarantees; and `conv1d`'s `M` is a range, so the actual row count
comes from the narrowed operands, not the band (`arch/common/rad_arch.h:1236-1246`).

---

## 4. Per-lane state in a duplex plugin

### 4.1 What the driver gives the plugin

`RadDuplexBatch` (ABI `abi/rad_runtime.h:661-690`) carries, for PROMPT and STEP, `n_lane`, `lane` and
`lane_h`; for PROMPT `n_tok`, `last_row`; for STEP `frame`, `frame_h`, `seed`, `flags`, `flags_h`. The
declaration's `audio_in`, `audio_out`, `text_out`, `status` buffers are **indexed by lane**, not by
entry: `build_step` writes `audio_in[lane]` (`core/gen/duplex_driver.cpp:860-861`) and reads
`audio_out[lane]`/`text_out[lane]`/`status[lane]` (`:979-1006`). A STEP pass may carry several lanes;
`lane[i]` is the lane of entry `i` (`:851-852`), and `cu_seqlens` is one row per entry (`:847`).

### 4.2 FIRST and reset

`RAD_DUPLEX_F_FIRST` is set when `j.frame == 0` (`core/gen/duplex_driver.cpp:853-854`); `j.frame` starts
at zero (`core/gen/duplex_driver.h:179`) and increments after every successful step
(`core/gen/duplex_driver.cpp:1009-1010`). `j.pos` starts at the prompt length and advances by one
(`:972`, `:1009`). This is the first live step of a lane, *before* the first user frame is encoded; it
is exactly when the codec stream is reset in the reference (`session.py:105-107`).

The lane index is assigned from `free_lanes_` on open (`:462-463`) and returned on release
(`:339-351`); a reopened lane can therefore reuse an index. The engine zeroes its KV/linear/conv state
groups, but it cannot and does not zero plugin-private buffers. There is no plugin callback on close:
`close` only marks, fails tickets and (outside a pass) releases the lane. **Therefore the plugin must
zero/reset its per-lane codec state on the FIRST step**, not on close. A packed state arena makes this
one `fill` per lane on FIRST; per-layer state buffers would need one `fill` each (~68).

`RAD_DUPLEX_F_FLUSH` exists (`abi/rad_runtime.h:644-648`) but the current driver's `flush` returns an
empty reply and changes no state (`duplex_driver.cpp:521-555`), so it is not a reset hook.

### 4.3 Keeping two interleaved lanes apart

Persistent plugin buffers get a lane axis, e.g. `state[max_lanes, ...]` and
`prev[max_lanes, ...]`, laid out as one packed arena at declare. In `step`, loop over the pass's
entries `i = 0..n_lane-1` and use `d->lane_h[i]` as the slot for both the state and the
`audio_in[lane]`/`audio_out[lane]` buffers; never use `i` as the lane. `radtest_duplex.cpp` does exactly
this for its per-lane text tables, taking `d->lane_h[i]` and writing `text_out[lane]`
(`tests/duplex_plugin/radtest_duplex.cpp:186-193`); its persistent buffers carry a lane axis
(`:51-59`). Two lanes in one pass then use disjoint state rows and disjoint audio buffers; the fact that
they share one launch stream is irrelevant because the plugin either issues one op per lane (the simple
v0) or packs per-lane `cu_seqlens`/`cache_idx` into one op. A codec pass that packs lanes must pass a
per-lane `cu_seqlens` and a per-lane slot array; it must not assume entry `i` is lane `i`.

Close/reopen is clean because FIRST resets the slot before any read. A lane that is closed mid-pass is
deferred until `done()` (`duplex_driver.cpp:557-589`, `:960-1023`), so the plugin never sees a half-freed
slot in the same pass.

### 4.4 State storage form

The declaration's buffers are device memory. The ABI note's v0 recommendation is a plugin-private
`RAD_BUF_PERSIST` buffer with a lane axis, zeroed by the plugin on the first step of a lane and on close
(`docs/DUPLEX-PLUGIN-ABI.md:425-429`). The engine has no plugin close callback (see §4.2), so H1 must
take the "first step" half of that and reset on FIRST. The engine cannot checkpoint/offload the buffer,
which is acceptable for v0. The codec is not a transformer layer and has no `rad_bind_layer_kv` index, so a KV state kind is not
appropriate. A packed state arena (one persistent buffer, offsets computed at declare) is the cleanest:
one fill per lane on FIRST, and `brow_slice`/`bcol_at` views for each layer's `[context, C]` window.

Memory: 1.20 MB f32 / 0.60 MB bf16 per lane for the persistent state (real), plus transient scratch.
The scratch can be one per size class and reused by liveness; the largest activation is ~3200x32
(real encoder stage 0 / decoder stage 6) and the deepest is 1x2048, so a scratch of a few hundred KB per
lane, not 26 MB, suffices if the buffers are sized per stage rather than as one max-width buffer.

---

## 5. The sampling step and latent scale

The codec's `sample` (`vibevoice_codec.py:196-200`) is

```
scale = randn(B, 1, 1)                    # one per batch row
noise = randn(mean.shape)                 # [B, T, D]
return mean + scale * (latent_std / 0.8) * noise
```

with `latent_std = 0.5` (`config.py:26`). The goldens record the combined term
`user_noise = scale * (latent_std/0.8) * raw` (`tests/data/haru/README.md:140-145`, `:176-180`), so a
test needs no RNG: `user_latent = user_codec_mean + user_noise` (`:180`, `:219-220`).

Where it belongs:

* **`encode` and `decode` are the block.** They produce/consume raw codec-space latents. The session
  calls `codec.sample(codec.encode(audio))` and `codec.decode(unscale_latents(head_out))`
  (`session.py:130-142`, `:159-163`).
* **Sampling can be a small helper on the block** (mean + recorded noise/scale), or the caller can add
  the recorded noise. The RNG is the caller's; the test replays recordings. I recommend the block expose
  `sample(mean, noise, scale)` with no RNG, because the `/0.8` and `latent_std` constants are the
  codec's, but the test can also just do the add in C++.
* **The model's latent scale/bias are NOT the codec's.** `HaruModel.scale_latents/unscale_latents`
  are `(raw + latent_bias) * latent_scale` and `z/latent_scale - latent_bias` with the tiny
  checkpoint's `latent_scale = 0.8`, `latent_bias = 0.1` (`models/haru.py:71-76`;
  `tests/data/haru/README.md:67-71`). They belong to the projectors/head, in raw<->normalised space.
  The codec's `/0.8` in `sample` is a separate hard-coded constant; do not apply the model's
  scale/bias inside the codec.
* The decode output is raw and unbounded; the reference clamps `[-1,1]` **after** the codec
  (`session.py:163`). The block must not clamp; the test tenant (or test) clamps to compare
  `frame{t}.audio_out`.

Test inputs for this block: `prompt.ref_codec_mean` (whole clip), `frame{t}.user_codec_mean` (stream),
`frame{t}.user_audio`, `frame{t}.user_noise`/`user_noise_raw`/`user_noise_scale` (t = 1..3 only),
`frame{t}.user_latent` (t = 1..3), `frame{t}.latent_out` (t = 0..24), `frame{t}.audio_out` (t = 0..24),
and `codec_dbg.*`.

---

## 6. Test plan

### 6.1 Wiring

The CMake wiring already exists: a test `tests/haru_codec_test.cpp` and a plugin target
`radtest_haru_codec` (from `tests/haru_plugins/codec/CMakeLists.txt`) get `RAD_E2E_SERVER`,
`RAD_E2E_HOME`, `RAD_HARU_PLUGIN` and `RAD_HARU_DATA` (`tests/CMakeLists.txt:397-408`). The plugin is an
`archtest` like `radtest_duplex` (`tests/duplex_plugin/CMakeLists.txt`), and the test copies it into a
temporary `RADIANCE_HOME/architectures`. The codec sources are compiled into the plugin by path
(`tests/HARU_HARNESS.md:80`). `tests/haru_harness.h` **does not exist in this worktree**: only the API
doc `tests/HARU_HARNESS.md` is committed; step 2 needs the header (or the test must carry it) and the
directory `tests/haru_plugins/codec/` currently holds only `.gitkeep`.

`Engine::start` (host libref, tiny model directory) must pass `--http-threads 4` (the engine's minimum;
`tests/duplex_plugin_test.cpp:108-117` uses exactly 4) and `--debug-accept-reference-kernels`; libref is
the only kernel library. The plugin still needs a minimal `RAD_KV_FULL` group and a PROMPT pass because
the duplex driver requires exactly one full-attention group and at least one prompt consumer
(`core/gen/duplex_driver.cpp:200-226`; `tests/duplex_plugin/radtest_duplex.cpp:64-83, 164-177`). The
codec work runs in STEP.

### 6.2 One tenant, several modes: the ABI fact to settle first

The harness doc says "a plugin may declare several tenants (for example encode and decode); the test
picks one in the start message" (`tests/HARU_HARNESS.md:15-17`). **The current engine does not support
that**: `Program` holds exactly one `RadDuplexDecl`, and a second `rad_declare_duplex_tenant` overwrites
it (`core/rad_core.h:464-474`, `core/build/rad_builder.cpp:2189-2197, 3047-3051`); `/duplex/v1/info`
registers one engine tenant plus the test-only echo tenants (`core/server/duplex.cpp:324-365`). So the
codec test cannot select encode/decode by tenant name without an ABI/engine change.

Options, in my order of preference:

1. **One tenant, mode from a host-visible field set at open.** The plugin's `tmpl` can be exactly
   `"{card}"`; the server tokenises the card, and the PROMPT pass sees `d->n_tok` (host-readable) and
   can store `mode[lane] = n_tok`. The test opens with a card of 1/2/3/4 characters for
   encode/decode/debug/clip. This uses no device read and no extra step. (Check the server's exact card
   rendering; `tmpl` is the plugin's, so `"{card}"` should make `n_tok` the card's token count.)
2. The same with the reference-frame count `d->n_ref` (host field, `abi/rad_runtime.h:676`), opening
   with 1/2/3/4 whole reference frames of dummy audio; this depends on the server's reference byte
   format.
3. A control FIRST step: the first STEP carries a mode word in `audio_in[0]` and replies with no audio;
   the codec state is reset on that FIRST, and real data starts at frame 1. This is simple but consumes
   a step and requires the test to ignore one reply.
4. Extend the engine to multiple duplex tenants (ABI change; out of scope for H1).
5. Build the plugin twice (one mode each) and start two engines; the harness `Engine` takes no plugin
   path argument (it is `RAD_HARU_PLUGIN` at compile time), so this needs harness changes.

I recommend option 1 and ask the reviewer to confirm the card/token assumption.

### 6.3 Modes and what each compares

With one tenant and `mode[lane]` set in PROMPT:

* **`encode`** (mode 1): each STEP consumes `audio_in[lane]` as 3200 samples of `[3200,1]`, runs the
  streaming encoder with that lane's state, and writes the 64 latent means as the first 64 floats of
  `audio_out[lane]`. The test sends `frame{t}.user_audio` for t = 1..24 and compares the first 64 floats
  with `frame{t}.user_codec_mean [1,64]`. The FIRST step is t = 1: reset state, then encode.
* **`decode`** (mode 2): each STEP reads the first 64 floats of `audio_in[lane]` as one `[1,64]` latent
  row and writes 3200 audio samples. The test sends `frame{t}.latent_out` for t = 0..24 (first STEP is
  t = 0; reset state, then decode) and compares the clamp of the reply with `frame{t}.audio_out`.
* **`clip`** (mode 3): the test sends `prompt.ref_audio` as 14 steps (13 x 3200 then 2400 samples; the
  harness zero-pads the last to 3200). The plugin streams the encoder over the 14 frames and, because
  this is the clip mode, uses the **true length 2,400** for the last frame (a test constant; the real
  plugin gets it from `ref_samples`) so the per-conv partial-hop pad of §2.2 is applied. After the 14th
  step it writes the 14 x 64 = 896 accumulated means and the test compares
  `prompt.ref_codec_mean [14,64]`. Do **not** encode all 14 x 3200 = 44,800: the last 800 zeros are
  audio, not the reference's per-conv pad, and the measured error is 0.091 on the last latent.
* **`dbg`** (mode 4): each STEP runs a **fresh-state** encode or decode of the input and returns one
  stage, selected by `d->frame_h[i]` (0..6 = enc_stage0..6, 7..13 = dec_stage0..6). The test sends
  `frame1.user_audio` on steps 0..6 and `frame0.latent_out` on steps 7..13. Each stage is the output of
  the **last block** of the stage (`gen_haru_tiny.py:394-422`).

Multi-lane: open two lanes (same mode or encode/decode) and submit steps for both before waiting for
either (or use two threads), so one STEP pass carries two lanes. Compare each lane with its single-lane
run. If the harness's `Lane::step` is strictly synchronous, this test needs a non-blocking step or an
engine-level driver test; that is a harness gap to resolve.

### 6.4 Packing and layout

* Encode input: the 3200 floats are 3200 rows x 1 channel (`[T,1]`), but `audio_in` is declared
  `[max_lanes, samples_per_frame]`. The lane's row is `bcol_at(audio_in, lane, 3200, 0, 3200, 1)`
  (`arch/common/rad_arch.h:1265-1272`); copy that 3200-element `[1,3200]` row into a `[3200,1]`
  staging buffer with one `cast` (the op copies the operand extent, and the elements are contiguous).
  Do not try to view the 2-D buffer as `[3200,1]`: its row stride is 3200, not 1.
* Decode input: the first 64 floats are one row x 64 channels (`[1,64]`). View them directly with
  `bcol_at(audio_in, lane, 3200, 0, 64, 1)`.
* Encode output: one row x 64 channels; the lane's `audio_out` row is
  `bcol_at(audio_out, lane, 3200, 0, 64, 1)`. The driver has already zeroed `audio_out` for the pass
  (`core/gen/duplex_driver.cpp:828-831`), so the unused tail is zero.
* Decode output: 3200 rows x 1 channel; `cast`/write it into the lane's `[1,3200]` `audio_out` row
  (`bcol_at(audio_out, lane, 3200, 0, 3200, 1)`, numel 3200 in both).
* **Stage tensors are `[1,C,T]` in the goldens** (`codec_dbg.*`, README `:197-205`) while the block's
  natural layout is `[T,C]`. The test must transpose: `want[c*T + t]` vs the block's `got[t*C + c]`.
  The final `user_codec_mean [1,64]` has T = 1, so channel order is already the block's order; the same
  is true for decoder stage 0 `[1,64,1]`.
* The whole-clip `ref_codec_mean [14,64]` is `[T,D]`, already the block's `[rows,channels]` order.

### 6.5 Tolerances

The README's thread A/B is exactly zero (`tests/data/haru/README.md:250-278`) because torch's reductions
do not change order there; it gives **no** C++ tolerance. The reference's own streaming-vs-whole f32
difference is ~7-8e-7 (§2.3), which is the floor from call shape alone. libref's `conv1d`, `gemm_nt_bias`
and `rmsnorm` accumulate in their own orders (`libref/ref_audio.cpp:104-124`,
`libref/ref_elementwise.cpp:23-41`), so a C++ constant-by-constant implementation will differ from torch
by more than that. Start at `Tol{1e-4, 1e-4}` (the harness default, `HARU_HARNESS.md:81-82`), compare
stage 0 first and localise with `codec_dbg`; record the measured max for every family. Do not tighten a
final-latent tolerance to the ~1e-6 reference A/B without measuring the C++ build: 7 stages and 52
blocks can amplify a single-stage rounding difference. For the audio output use the same `{1e-4,1e-4}`
first, with the clamp applied to both sides. All of this is a starting point for the reviewer's Linux
run, not a claimed result.

---

## 7. Size, risks, order of work, questions

### 7.1 Size

* `arch/haru_common/haru_codec.{h,cpp}`: ~700-1000 lines. The weight declaration is 552 tensors real
  (172 tiny), each with a name-map entry before the weight; the layer/state descriptors are table-driven;
  the encode/decode step is a loop over the same sequence in both directions.
* `tests/haru_plugins/codec/radtest_haru_codec.cpp`: ~300 lines (minimal KV group + PROMPT from
  radtest, the codec declaration, the four modes, per-lane state).
* `tests/haru_codec_test.cpp`: ~400 lines plus whatever `tests/haru_harness.h` needs.
* Real weight memory 1.37 GB bf16; real state 0.60 MB/lane bf16; real compute ~4.44 GMAC/frame.

### 7.2 Risks

1. **No stateful conv op with bias.** §3.3. The composition is exact but adds copies/launches; the
   device step budget must be measured.
2. **One tenant per program.** §6.2. The harness doc and the engine disagree; settle the mode channel
   before coding the test plugin.
3. **f32 device coverage in libhalo.** `conv1d_dw`, `rmsnorm`, `gelu` and `mul` have no f32 device row
   in this fork (only bf16); `conv1d`/`conv_t1d`/`add`/`mul_weight`/`cast` do (`libhalo/halo_rows.cpp`
   rows above). The real codec is bf16, so the production path is covered; an f32 device run would fall
   back to libref for those four and is not a production configuration. The tiny test is f32 on libref,
   which is fine.
4. **bf16 RMSNorm rounding.** The reference rounds `rms_norm(x.float())` to the activation dtype
   *before* multiplying the gain (`vibevoice_codec.py:28-30`); `rmsnorm` and libhalo's row keep more f32
   and round once at the store (`libref/ref_elementwise.cpp:23-41`; `libhalo/halo_rows.cpp:5855-5863`).
   For f32 this is identical; for the real bf16 codec it can differ by a bf16 ulp per norm. Flag it for
   the real-model stage; the tiny golden cannot see it.
5. **Partial-hop semantics.** §2.2-2.3. The clip test must use 44,000, not 44,800.
6. **Lane reset.** FIRST is the only hook; close has no plugin callback. A missed reset silently mixes
   a reopened lane's history, which is exactly the kind of fluent-wrong failure the project has seen.
7. **Arena size.** Per-lane state plus scratch at `max_lanes`; a max-width shared scratch would be
   tens of MB per lane unnecessarily. Size per stage.
8. **Test tensor layout.** Stage goldens are `[C,T]`; forgetting the transpose looks like a broken
   first stage.
9. **Harness missing.** `tests/haru_harness.h` is only documented; step 2 needs it, and the two-lane
   batching test may need a non-blocking step.
10. **Name map order.** Every `rad_decl_weight` needs its `map_copy`/map entry declared **before** it
    (`tests/duplex_plugin/radtest_duplex.cpp:98-111`); a missing entry is "absent from the container"
    and the model cannot run.

### 7.3 Order of work (each step independently checkable)

1. **Declare-only.** Weights + name maps, no ops; build the plugin and confirm the loader finds all 172
   tiny tensors and the declared shapes match `model.safetensors`. This is the cheapest place to catch a
   missing map entry or a wrong shape.
2. **Fresh-state single stage.** Stem + stage 0 blocks (encode a 3200 frame on fresh state) and compare
   `codec_dbg.enc_stage0` (after the transpose). Validates `conv1d`, `conv1d_dw`, `rmsnorm`, `gelu`,
   `gemm_nt_bias`, `mul_weight`, `add`, and the stage boundary.
3. **Fresh-state full encoder.** Compare `enc_stage0..6` and then `user_codec_mean` for
   `frame1.user_audio`. Validates the downsampling/strides and the head.
4. **Streaming encoder.** Reset on FIRST, per-conv history and partial pad; compare `user_codec_mean`
   for t = 1..24 and then the 14-step 44,000 clip against `prompt.ref_codec_mean`. Validates state,
   reset, partial hop.
5. **Fresh-state full decoder.** Compare `dec_stage0..6` for `frame0.latent_out`. Validates the
   transposed conv window and the decoder channel order.
6. **Streaming decoder.** Compare `audio_out` for t = 0..24 with the clamp in the test. Validates the
   transposed state and lane reset.
7. **Sampling helper (optional).** `mean + noise` against `user_latent` for t = 1..3.
8. **Two lanes and reopen.** Interleaved lanes and a close/reopen with the same index; compare against
   single-lane runs.
9. **Tighten tolerances and record the measured maxima** in the test comment and the report.

### 7.4 Questions for the reviewer

1. **Mode selection** (§6.2): accept the one-tenant + card-length (`n_tok`) convention, or do you want a
   different host-visible field (reference-frame count / a control FIRST step), or a multi-tenant engine
   change?
2. **Sampling:** should the codec block expose `sample(mean, noise, scale)`, or is `encode`/`decode`
   only plus a caller-side add preferred?
3. **Dtype policy:** the test is f32 on libref; the real codec is bf16. Confirm the block is
   parameterised by dtype and that the f32 libhalo gaps (`conv1d_dw`, `rmsnorm`, `gelu`, `mul`) are
   acceptable for v0.
4. **Stateful conv:** accept the explicit-history composition and its copies for v0, or should H1
   include the `conv1d`/`conv_t1d` state-operand extension?
5. **Partial hop:** confirm the unified streaming path with a true sample length (recommended), and that
   the clip test is 44,000 samples with the last frame 2,400.
6. **Tolerance:** is `{1e-4,1e-4}` the expected starting point, with the measured maxima to be recorded,
   or does the reviewer want a per-family tolerance from the start?
7. **Harness:** who provides `tests/haru_harness.h` in step 2, and can `Lane` keep two steps in flight
   for the interleaving test?
8. **Two-lane reset:** confirm the plugin-private `RAD_BUF_PERSIST` lane arena with a FIRST-step fill is
   the intended v0, with no engine checkpoint/offload (as `DUPLEX-PLUGIN-ABI.md:425-429` says).
9. **The real codec's 1.37 GB**: is that expected/acceptable for H1, or is the codec expected to be
   quantised or partially offloaded before the real-model stage? The tiny test cannot answer this.

---

## Appendix: what I could not verify

* The C++ tree does not build or run here; no C++ test has been run, and none of the §6 tolerances or
  the launch-count estimate is measured on the target.
* libhalo's device rows were read, not run. The bf16 RMSNorm rounding difference (§7.2 item 4) is from
  reading the reference and the row description, not a measured device result.
* The server's exact card/reference rendering for the mode channel (§6.2) was not exercised; if `tmpl`
  `"{card}"` adds special tokens, the mode numbers shift.
* The exact arena/scratch sizing is a plan, not a buffer-plan result; the builder's liveness may allow
  more reuse than the estimate.
* The real checkpoint's 1.37 GB and 4.44 GMAC/frame are computed from `header.json` and the shapes; they
  are not a measured runtime.
