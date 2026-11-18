# Haru H4: the language-model trunk as a Radiance block

Status: design note for step 1 of H4. **No code is written in this step.** This note states what
the trunk needs, what the tree can already do, and what it cannot, with a citation for every claim
into the pinned Python reference (commit `7fe6b76`, package `Haru/haru/haru/`, referred to below as
`haru/`) or into this tree. Where I could not check something I say so explicitly; I cannot build
or run the C++ tree here, so nothing below is a test result.

The block lives in `arch/haru_common/haru_lm.{h,cpp}` and is exercised by a duplex test plugin
(`tests/haru_plugins/lm/`) through the tensor tunnel of `tests/HARU_HARNESS.md:10-17`, against the
goldens in `tests/data/haru/golden.safetensors` (schema: `tests/data/haru/README.md:147-193`).

---

## 0. Summary

The trunk itself is the Qwen3.5 hybrid the tree already serves
(`arch/qwen35_bf16/qwen35_bf16.cpp`, `docs/ARCHITECTURES.md:255-368`). The geometry is read, not
hard-coded (`geom_from`, `arch/common/rad_arch.h:1004-1122`), and the op vocabulary the gated
delta net (GDN), gated attention and SwiGLU blocks issue is all in `docs/OPS.md` and implemented by
libref. Three things are genuinely different and drive the design:

1. **The input is an embedding row, not a token id.** `haru.py:83-107` sums the token embedding
   and two projected audio latents; the LM is then run with `inputs_embeds=`
   (`haru.py:109-112`). The duplex STEP pass carries no per-row embedding channel, and
   `RadBatch::mm_embd/mm_rows` is not staged by the duplex driver
   (`core/gen/duplex_driver.cpp:656-706`).
2. **The tiny checkpoint is float32.** Every LM tensor in
   `tests/data/haru/haru_tiny/model.safetensors` is F32 (verified by reading its safetensors
   header); the goldens are float32 (`tests/data/haru/README.md:15-17`). The shared `*BF16` blocks
   declare every linear weight `RAD_BF16` (`rad_block_attn_gated.h:119-133`,
   `rad_block_gdn.h:156-178`, `rad_block_mlp.h:55-60`), so declaring them over the tiny checkpoint
   would *narrow* f32→bf16 at load (allowed by `core/build/rad_builder.cpp:210-227`); a bf16
   rounding is ≈2^-8 relative, some four orders above the harness's 1e-4 tolerance, so they cannot
   be called as they stand for the H4 test. (That is an argument from bf16's precision, not a
   measurement.)
3. **A duplex tenant cannot declare logits.** `rad_declare_duplex_tenant` refuses a program that
   declared logits (`core/build/rad_builder.cpp:2065-2082`; the logits declaration is
   `:1225-1226`), which qwen35 does (`arch/qwen35_bf16/qwen35_bf16.cpp:293-295`). The trunk must
   own an ordinary f32 output buffer and issue `logits_gemm` itself, for the rows of this pass.

Recommended shape: `haru_lm` is a self-contained, dtype-parameterised mirror of the three qwen35
blocks' *declaration and issue order*, reusing `arch/common/rad_arch.h` helpers and the exact op
parameter names from `docs/OPS.md`; it does **not** edit the shared block headers. Section 1.3
gives the alternative and its cost. Section 5.1 gives the one engine-side change that makes the
tiny checkpoint loadable at all (also outside the block; it needs a decision).

---

## 1. Reuse

### 1.1 What can be called as it is

`arch/common/rad_arch.h` is a header of inline helpers; qwen35 uses it (`qwen35_bf16.cpp:39-44`)
and so can Haru. The parts H4 needs, all behaviour-neutral:

| helper | where | used for |
|---|---|---|
| `Names`, `map_copy`, `map_copy_alt`, `map_concat`, `map_rows` | `rad_arch.h:318-341, 528-544, 619-627, 455-465` | checkpoint name map (maps before weights, qwen35 `:302-309`) |
| `decl_w`, `decl_b`, `decl_view`, `rw` | `rad_arch.h:391-397, 429-439, 404-413, 661-668` | weights, buffers, read/write sets |
| `geom_from` | `rad_arch.h:1004-1122` | dimensions, shard, eps/theta, `rope_mc` |
| `is_full_attention`, `check_layer_schedule` | `rad_arch.h:948-950, 957-978` | layer kinds vs `layer_types` |
| `meta_int2`, `divide_or_fail`, `rad_kv_cache_dtype` | `rad_arch.h:854-861, 815-823, 808-812` | geometry keys, head split, KV dtype |
| `yarn_from`, `decl_rope`, `decl_attn` | `rad_arch.h:1384-1469, 1474-1484, 1356-1362` | rotary declarations (the same calls qwen35 makes) |
| `brows`, `bcol`, `bcol_at`, `brow_slice`, `praw`, `praw2` | `rad_arch.h:1247-1310` | step-sized operands |
| `rope_posmc` | `rad_arch.h:1333-1336` | one-/three-component positions |
| `batch_split`, `kv_batch` | `rad_arch.h:1492-1498, 1501-1505` | phase split, group batch |
| `RAD_ARCH_PLUGIN`, `RAD_ARCH_PROBE`, `RAD_ARCH_TRY` | `rad_arch.h:1622-1644, 344` | plugin exports, probe |

The three blocks are also header-only (`struct … { … inline … }`), so they *can* be included and
called by another plugin — but only with the geometry/dtype they were written for (see 1.3).

### 1.2 What is file-local to qwen35 and not reusable

These live in `qwen35_bf16.cpp` and would have to be copied or re-derived:

- the metadata reads for `full_attention_interval`, `nextn_predict_layers`, `n_rot`
  (`:131-178`), the GDN `Config` construction (`:183-214`) and the three KV groups
  (`:227-254`);
- the buffer set and the layer loop that builds each layer's checkpoint `Src` names
  (`:259-294, 360-424`);
- the final norm, gather and lm_head issue (`:426-448`), the `step()` orchestration
  (`:481-540`), the vocabulary all-reduce (`:331-339`), calibration taps and the vision tower
  (`:476-478`), `plugin_words` (`:528-531`) — none of which H4 needs.

Copying these is cheap (a few dozen lines each) and carries no risk to anyone else. Changing
`qwen35_bf16.cpp` to expose them *would* touch a shipped plugin for no benefit to it, and moving
them to a new shared header would create a header with one consumer — the extraction cost buys no
reuse.

### 1.3 The blocks themselves: bf16-only

`AttnGatedBF16`, `GdnBF16` and `MlpBF16` fix `RAD_BF16` on every linear weight
(`rad_block_attn_gated.h:121-133`, `rad_block_gdn.h:158-178`, `rad_block_mlp.h:57-60`) and
`RAD_BF16` on the within-chunk inverse plane (`rad_block_gdn.h:282-284`). Their activation
buffers follow `g.act_dtype` and their op dtype strings follow `g.dtype` (`rad_arch.h:789-794`),
so a bf16 checkpoint is served correctly. The tiny checkpoint is f32 everywhere, and the loader
turns a declared bf16 weight over an f32 tensor into a bf16 copy
(`core/build/rad_builder.cpp:210-227`), so using the blocks would test a bf16 model against f32
goldens.

*Cost/risk of extracting a dtype knob into the shared headers:* three structs gain a
`weight_dtype` (and `kkt_dtype`) member defaulting to `RAD_BF16`, used at the `decl_w` sites;
defaults preserve every existing plugin byte for byte. The headers are included by
`qwen35_bf16.cpp`, `qwen35moe_gguf.cpp` (`rad_block_gdn.h`), `qwen3asr_bf16.cpp` (`rad_block_mlp.h`)
and `unlimitedocr_bf16` (`attn_llama.h`, all three), so a review and a rebuild of four plugins
follows. It is a small, mechanical change, but it is a shared-ABI-adjacent surface changed for one
block.

*Recommendation:* do **not** change them in H4. Write `haru_lm`'s declaration and issue code as a
dtype-parameterised mirror, with a comment naming the block and line range it mirrors
(`GdnBF16::declare/step` at `rad_block_gdn.h:114-354`, `AttnGatedBF16` at
`rad_block_attn_gated.h:78-308`, `MlpBF16` at `rad_block_mlp.h:47-116`). It is ~120 lines of
open-coded issue calls over the same helpers, and it is the only way to get the f32 tiny path
without touching four shipped plugins. If the reviewer prefers DRY, the knob above is the
alternative to schedule separately, not inside H4.

### 1.4 What Haru's LM differs from in qwen35's declaration

Values: tiny / real (`tests/data/haru/haru_tiny/config.json`, the operator's `real/config.json`);
both are nested under `"lm"`.

| key | tiny | real | qwen35's handling |
|---|---|---|---|
| `num_hidden_layers` | 4 | 32 | `geom_from` `rad_arch.h:1016` |
| `hidden_size` | 64 | 2560 | `:1017` |
| `intermediate_size` | 128 | 9216 | `:1021, 1112` |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 4 / 2 / 64 | 16 / 4 / 256 | `:1019-1020, 1085, 1102` |
| `layer_types` | `[linear, linear, linear, full]` | full at 3,7,…,31 | checked against the interval, `qwen35_bf16.cpp:141-142` |
| `full_attention_interval` | 4 | 4 | `:131-142`, `rad_arch.h:948-950` |
| `attn_output_gate` | true | true | **not read**: the block always declares query and gate (`rad_block_attn_gated.h:85, 108-110`); Haru should refuse 0 rather than assume |
| `partial_rotary_factor` | 0.25 | 0.25 | `qwen35_bf16.cpp:162-178`; `n_rot` = 16 / 64 |
| `mrope_section`, `mrope_interleaved` | `[3,3,2]`, true | `[11,11,10]`, true | `rad_arch.h:1043-1056` sets `rope_mc`, mode `imrope`, sections |
| `linear_num_key_heads` / `linear_num_value_heads` | 2 / 4 | 16 / 32 | `qwen35_bf16.cpp:184-187`; `GdnBF16` requires value % key == 0 (`rad_block_gdn.h:120-126`) — 4%2 and 32%16 pass |
| `linear_key_head_dim` / `linear_value_head_dim` | 16 / 16 | 128 / 128 | `:188-194` |
| `linear_conv_kernel_dim` | 4 | 4 | `:190`; conv group `:248-254` |
| `mamba_ssm_dtype` | float32 | float32 | linear state declared f32, `:235-244` |
| `rms_norm_eps` | 1e-6 | 1e-6 | `geom_from`, `rad_arch.h:1038`; Qwen3.5 norms are `(1+w)` (`qwen35_bf16.cpp:149-157`), verified numerically in section 7 |
| `tie_word_embeddings` | true | true | `map_copy_alt` (`qwen35_bf16.cpp:306-320`) |
| vocabulary | 512 | 248320 | `geom_from` |

Beyond the values, the differences that change the declaration:

- **Config location.** The LM dict is under `"lm"` (`haru/config.py:51-60`); the engine's metadata
  reader aliases `text_config.*` and `llm_config.*` to bare keys but not `lm.*`
  (`core/format/checkpoint.cpp:213-238`), and reads `num_hidden_layers` etc. from those
  (`:264-303`). A Haru config therefore yields `n_layers = 0` and the engine refuses it before any
  plugin runs (`core/engine_bringup.cpp:866, 911`). See 5.1.
- **Input.** `inputs_embeds` on every row (`haru.py:96-107, 109-112`), where qwen35 embeds token
  ids and optionally overwrites media rows (`qwen35_bf16.cpp:322-339, 490-493`; `mm_rows` at
  `abi/rad_runtime.h:229-242`).
- **Output.** Final normed hidden per row *and* logits for the same rows; no `rad_declare_logits`,
  no `RadBatch::out_ids` (the duplex driver never sets `n_out`/`out_ids`,
  `core/gen/duplex_driver.cpp:656-706`).
- **No MTP, no vision, no routed FFN, no drafter, no tensor parallelism required for H4** (world 1;
  all-reduce paths declared only at `world > 1`).

---

## 2. Embeddings in

### 2.1 The three candidate paths

**(a) `RadBatch::mm_embd` / `mm_rows`.** This replaces a token's embedding with an encoder row
(`abi/rad_runtime.h:229-242`; qwen35 consumes it through `MediaRows`,
`qwen35_bf16.cpp:328-330, 493`). It is **not available**: the duplex driver never sets
`n_mm_rows`, `mm_rows` or `mm_embd` in either pass (`core/gen/duplex_driver.cpp:656-706`), and
`mm_embd` is bf16 only (`abi/rad_runtime.h:242`), which is wrong for the f32 tiny path.

**(b) `op_embed` + projector ops, the fishomni pattern.** `fishomni` runs `embed_lookup` into
`b_x`, then `op_pfb` writes the projected reference-frame rows over those rows before the trunk
(`arch/fishomni/fishomni.cpp:1113-1134, 1986-1997`; the op is `rvq_embed` there). This is the
production shape for Haru: `embed_lookup` for the token stream, then two
`gemm_nt`+`rmsnorm`+`gemm_nt` projectors added into `b_x` (the projector block is a separate
package, `arch/haru_common/README.md:15`). Note the difference: fishomni's op *replaces* a frame
row's embedding (`rvq_embed` writes only rows whose code is valid, `libref/ref_audio.cpp:417-444`),
while a Haru prompt row can carry a token embedding *and* a model projection at once
(`haru.py:96-107`), so Haru's projector path must end in `add`. Neither version can carry the
harness's already-summed golden row through the wire by itself.

**(c) The trunk consumes `b_x`; the plugin fills it.** The block declares `b_x`
`[max_tok, n_embd]` and exposes its token-embedding weight and `embed_lookup` op, but
`step_layers()` runs only the trunk from whatever `b_x` holds. A caller fills it:
`op_embed` plus projectors in production (b), or a single `cast` from the tunnelled `audio_in`
row in the H4 test. `cast` is a dtype copy (`docs/OPS.md:232`,
`libref/ref_elementwise.cpp:1108-1125`) and f32→f32 is an exact copy.

**Chosen: (c), with (b) as the production filler.** It is the only path that accepts an exact f32
embedding row through the one input channel the duplex driver stages on STEP (`audio_in`,
`core/gen/duplex_driver.cpp:860-861, 828-831`), and it keeps the trunk independent of how the row
was made — which is exactly what H4 tests. It also gives the production plugin the natural place
to add the projectors without a new ABI.

### 2.2 What the test plugin needs

- A `cast` op declared `M` range 1..max_tok, `n = n_embd`, `from = "f32"`, `to = "f32"`; issued on
  STEP with the lane's 64 floats as `bcol_at(audio_in, lane, samples_per_frame, 0, n_embd, 1)`
  and row `i` of `b_x` as the output (`brow_slice`).
- One prompt row: a `fill` or the batch tokens for `op_embed`, see 8.1. `prompt.embed[0]` is
  exactly `embed_tokens[prompt.tokens[0]]` (verified: max abs difference 0.0 over all 64 values;
  row 0 is `model_mask = user_mask = 0`, `haru.py:96-107`), and `prompt.tokens[0]` is 482 in the
  tiny tokenizer.
- `audio_in`/`audio_out` f32 `[max_lanes, 3200]` and `status` i32, as in
  `tests/duplex_plugin/radtest_duplex.cpp:51-59`, declared `RAD_BUF_PERSIST`.

---

## 3. State

qwen35 is the template: one paged full-attention group for the full layers, one recurrent-state
group and one convolution group for the GDN layers, bound per layer
(`qwen35_bf16.cpp:227-254, 374, 388-390`).

| group | kind | declaration (mirroring `qwen35_bf16.cpp:227-254`) | bound layers |
|---|---|---|---|
| full | `RAD_KV_FULL` | dtype = f32 (tiny) / bf16 (real), `n_head_kv` = 2 / 4, `head_dim` = 64 / 256 | layer 3 / 3,7,…31 |
| recurrent | `RAD_KV_LINEAR` | f32, `n_head_kv = linear_num_value_heads` (4 / 32), `state_dim = {value_head_dim, key_head_dim}` | the linear layers |
| conv | `RAD_KV_CONV` | conv state dtype, `n_head_kv = 2*key_heads + value_heads` (8 / 64), `head_dim` = key head dim, `conv_width` 4 | the linear layers |

The duplex driver accepts exactly this: it walks the groups from handle 1, requires exactly one
`RAD_KV_FULL`, and accepts any number of `RAD_KV_LINEAR`/`RAD_KV_CONV` beside it, refusing every
other kind (`core/gen/duplex_driver.cpp:198-220`). It is the audio driver's rule copied
(`core/gen/audio_driver.cpp:169-197`, whose header states the contract at
`core/gen/audio_driver.h:12-16`), so the same three-group hybrid declaration works on either
driver. It stages each state group as one block-table word and one `state_index` row per lane
(`core/gen/duplex_driver.cpp:683-693, 708-731`, the audio driver's `stage_state` at
`core/gen/audio_driver.cpp:535-558`), with a width from
`KVManager::state_index_width` (`core/mem/kv.cpp:1769-1774`, ≤ 8 as
`core/gen/duplex_driver.h:130` requires). The per-lane memory follows from
`core/mem/kv.cpp:183-230`.

Tiny (f32 state, 1 full + 3 linear layers, 61-row test lane):
- full KV: `2 × 64 × 2 × 4 = 1024 B` per token per layer × 1 layer = 1024 B/token → 62.5 KB at 61
  rows (131 KB at the test's 128-row context bound);
- recurrent: `4 × 16 × 16 × 4 = 4096 B` per layer × 3 = 12,288 B;
- conv: `(4-1) × 8 × 16 × 4 = 1536 B` per layer × 3 = 4,608 B.

Real (bf16 KV/conv, f32 recurrent state, 8 full + 24 linear layers):
- full KV: `4 × 256 × 2 × 2 = 4096 B` per token per layer × 8 = 32,768 B/token; at a 30-minute
  lane (13,500 rows) ≈ **422 MiB per lane** (13,500 × 32,768 ≈ 442 MB), 844 MiB at 27,000;
- recurrent: `32 × 128 × 128 × 4 = 2 MiB` per layer × 24 = **48 MiB**;
- conv: `3 × 64 × 128 × 2 = 49,152 B` per layer × 24 ≈ **1.1 MiB**.

So a real lane costs its reserved context in full-attention KV (the driver reserves the whole bound
at open, `core/gen/duplex_driver.h:8-11`; `core/gen/duplex_driver.cpp:455-456`) plus ~49 MiB of
state. The state pool is allocated for `max_seqs` sequences, not for `max_lanes`
(`core/mem/kv.cpp:230`), and the engine's default `max_seqs` for a duplex tenant is 8
(`core/engine_bringup.cpp:1150-1153`); a real deployment should set `--max-num-seqs` to its lane
count or pay 8×48 MiB of state. That memory is a deployment decision (section 9.3), not part of
H4, but the note flags it because it dominates the per-lane budget.

*Not verified:* the KV manager's exact block rounding for the tiny geometry (only the arithmetic
above is derived from `core/mem/kv.cpp`); the test's 128-row bound is an assumption of this note.

---

## 4. Positions

The reference gives every text/audio row one position, and transformer's Qwen3.5 rotary is fed
three equal copies of it:

- with `position_ids=None`, `Qwen3_5TextModel.forward` builds
  `arange(T) + past_seen_tokens`, expands it to 4 planes, uses plane 0 for masking and passes the
  other 3 to the rotary (`modeling_qwen3_5.py:1241-1251, 1259-1277`; `haru.py:109-112` does not
  pass `position_ids`).
- the rotary expands to 3 planes again and its `recomposition_frequencies` only overwrites
  entries with the *same* values when the components are equal
  (`modeling_qwen3_5.py:177-200`), so three equal components are bit-for-bit a one-component
  rotation. The prefill chunk of 512 (`haru/session.py:29, 115-123`) does not change this: the
  tiny prompt is 37 rows, one chunk, positions 0..36. (The real prompt, up to ~1400 rows with a
  180 s reference, is several chunks in the reference; H4 tests the tiny one.)
- The inverse frequencies are `1 / theta^(2i/rotary_dim)` over `rotary_dim/2` pairs
  (`modeling_qwen3_5.py:166-173`), i.e. a rotary ladder of width `rotary_dim`.

This tree stages exactly those positions and nothing else:

- PROMPT: `pos[i] = i` for the lane's rows (`core/gen/duplex_driver.cpp:750-753`); STEP: one row
  per lane at `pos = j.pos`, the lane's next row (`:842-847`), which the driver advances by one per
  completed prompt and step (`:972, 1009-1010`).
- `RadBatch::rope_pos` is left null and `rope_mixed` 0 (`base_batches`, `:663-705`; neither field
  is set anywhere in the file).
- `rope_posmc` therefore returns the rank-1 `positions` operand (`arch/common/rad_arch.h:1333-1336`).
- libref's `rope` with a rank-1 position sets every component to that one value
  (`libref/ref_attn.cpp:925-931`) and `rope_freqs` reads no sections when the operand is rank 1
  (`:903, 753-780`); the `imrope`/`mrope` path is documented as reducing to `neox` then
  (`:740-747`). The pairing is halves-based (`rope_head`, `:783-795`), matching the reference's
  `rotate_half` (`modeling_qwen3_5.py:659-663, 690-703`).

**Conclusion:** the positions the duplex driver stages give the same rotary result as the
reference, provided `n_rot` and `theta` are read from the `lm` config (1e7, 0.25) and the rope is
declared through `decl_rope` (`rad_arch.h:1474-1484`). Nothing must change in the driver. The
only thing to get right is that the lane's first STEP follows one PROMPT row (see 8.1): the
driver's `pos` starts at the prompt length, which is the reference's `past_seen_tokens`.

*Not verified:* that libref's `ref_pow`/`ref_cos`/`ref_sin` reproduce torch's f32 trig to the last
bit; the reference's cos/sin are computed in f32 (`modeling_qwen3_5.py:181-186`) and libref's are
its own f32 approximations. This is inside the tolerance reasoning of 8.4, not a semantic
difference.

---

## 5. PREFILL vs STEP, and loadability

### 5.1 The loadability gap (decide before coding)

The tiny checkpoint is served directly from its safetensors directory (`Engine::read_checkpoint_metadata`,
`core/engine_bringup.cpp:895-930`), and its LM geometry is nested under `"lm"`. Without a change,
`n_layers` is 0 and the engine refuses the model (`:911`). Two ways out:

- **(recommended) one line in the metadata reader:** add `"lm."` to the prefix list that already
  aliases `text_config.`/`llm_config.` to bare keys (`core/format/checkpoint.cpp:213-230`). The
  flattened `lm.num_hidden_layers`, `lm.rope_parameters.mrope_section`,
  `lm.partial_rotary_factor`, `lm.layer_types`, … become the bare keys every reader
  (`geom_from`, `qwen35`-style metadata reads) already looks for. The reader's "a bare key wins
  where both exist" rule (`core/format/checkpoint.cpp:216-229`) keeps the top-level
  `model_type: "haru"` from being overwritten by `lm.model_type: "qwen3_5_text"`. It is
  behaviour-preserving for every checkpoint without an `lm.*` key, and it also makes
  `rad-convert` able to convert the real Haru checkpoint (the converter uses the same reader).
  **This is outside H4's files; it needs the reviewer's decision.**
- **test-only alternative:** the harness's `Engine` prepares a temporary copy of
  `tests/data/haru/haru_tiny` whose `config.json` has the `lm` dict promoted to the top level
  (tensors unchanged). No engine change, but every Haru block test then depends on the harness
  doing this; H4 must not edit the committed goldens (`tests/data/haru/README.md:32-37`).

*Could not verify:* which route the (not yet written) `tests/haru_harness.h` takes; its API says
the model directory is `tests/data/haru/haru_tiny` (`tests/HARU_HARNESS.md:47-49`).

### 5.2 The same blocks serve both passes

The reference processes the prompt in segments of at most 512 rows (`haru/session.py:29,
116-123`) and every later frame as one row (`:141-142`); the duplex driver's PROMPT pass is the
segment and its STEP pass the one row. The same blocks serve both, mathematically: the GDN block
branches on phase, using the convolution's prefill form and the chunked scan for prefill and its
update/recurrent forms for decode (`rad_block_gdn.h:4-17, 285-350`); attention is the same
`attn_paged` with a different `q_len`; the norm/MLP/GDN-norm are per-row. libref implements every
one, and its arithmetic is dtype-generic (`ldt`/`stt`, `libref/ref_common.h:126-221, 339-340`):

| pass | ops | libref |
|---|---|---|
| PROMPT (T rows) | `rmsnorm`, `gemm_nt` ×(qg,k,v / qkv,a,b,z), `cast`/`embed_lookup`, `rope` ×2, `kv_store`, `attn_paged`, `sigmoid`, `mul`, `gemm_nt` (o), `gdn_conv_prep`, `gdn_kkt_solve`, `gdn_chunk_scan`, `gdn_gated_rmsnorm`, `gemm_nt` (out), `add`, `silu_mul`, `gemm_nt` (down) | `ref_registry.cpp:2579, 2599, 2696-2715`; GDN contract `docs/OPS.md:2375-2387` |
| STEP (1 row/lane) | the same minus `gdn_conv_prep/kkt_solve/chunk_scan`, plus `gdn_conv_update` and `gdn_recurrent_update` (which folds the gated norm) | same rows; `ref_gdn.cpp:308-384, 666-917` |
| both | final `rmsnorm`; `logits_gemm` (tied weight); optional `gemm_nt` over a 2-row PAD/EOP view | `ref_registry.cpp:2579, 2727-2729` |

The norm, elementwise and vocabulary rows are `ref_registry.cpp:2371-2410` (`rmsnorm` and its
variants), `:2424-2488` (`add`, `mul`, `silu_mul`, `sigmoid`, `fill`), `:2509` (`cast`), and
`:2722-2723` (`embed_lookup`).

Exact parameter names to use (do not invent any; this is where earlier work went wrong). The
declaration *order* should be the shared block's / qwen35's, which is the order the first-declared
schema fixed (`docs/OPS.md:170-184`); the list below is the set of keys:
`rmsnorm` `M n eps dtype wadd`; `gemm_nt` `M N K dtype`; `rope` `M head_dim n_head n_head_kv theta
scale mode rotary_dim sections`; `attn_paged` `q_len head_dim gqa causal window n_head block_size
max_seqs max_ctx q_dtype kv_dtype`; `kv_store` `M head_dim n_head_kv block_size kv_dtype`;
`gdn_conv_prep` `M head_k head_v chunk conv_width`; `gdn_conv_update` `q_len head_k head_v
conv_width`; `gdn_kkt_solve` `M head_k chunk`/derived; `gdn_chunk_scan` `M head_k head_v scale`;
`gdn_recurrent_update` `q_len head_k head_v n_head_v scale eps act state_form`;
`gdn_gated_rmsnorm` `M channels eps act`; `embed_lookup` `M n_embd n_vocab dtype vocab_offset
mult`; `logits_gemm` `M n_vocab n_embd dtype`; `cast` `M n from to`; `fill` `M n dtype value`;
`add` `M n dtype`; `sigmoid`/`mul`/`silu_mul` `M n dtype` (all `docs/OPS.md:188-238, 789-795,
957-975, 2375-2387, 2626-2633`).

### 5.3 Rows/lanes the declarations must allow

- `RAD_RANGE("M", 1, max_tok)` on every row-wise op, `RAD_RANGE("q_len", 1, max_tok)` on
  `attn_paged` and the two GDN decode ops (as qwen35: `rad_block_gdn.h:197-244`,
  `rad_block_attn_gated.h:198-208`).
- `max_seqs = ctx->max_seqs`, `max_ctx = ctx->max_ctx` in `attn_paged`; the driver sets
  `max_q_len = T` for PROMPT and 1 for STEP (`core/gen/duplex_driver.cpp:785, 872-880`).
- Buffers at `[max_tok, …]` and KV/state groups as in section 3; the driver's PROMPT is T ≤
  `max_prompt` (`:411-416`) and its STEP is one row a ready lane (`:809-895`).
- `max_tok` is the largest pass: for H4's 1-row prompt + single-lane steps it only has to be ≥ 1,
  but declare it at 64 so a later 37-row PROMPT check fits without changing the program.

*Not verified:* that the selector binds f32 operands to libref for every one of these ops. The
schemas accept the tensors and libref's kernels are dtype-generic; the band constraints live in
each kernel row and I could not run the selector.

---

## 6. Weights

### 6.1 Names

The checkpoint uses `lm.*` (tiny state dict; real `header.json`). Declared names can be qwen35's
(`blk.N.*`, `token_embd.weight`, `output.weight` — `qwen35_bf16.cpp:302-320, 364-414`), mapped
from:

| declared | checkpoint source | rename? |
|---|---|---|
| `token_embd.weight` | `lm.embed_tokens.weight` | 1:1 |
| `output.weight` (tied head) | `lm_head.weight` if present, else `lm.embed_tokens.weight` | **one source feeds two declared names**; no `lm_head` tensor exists in either checkpoint |
| `output_norm.weight` | `lm.norm.weight` | 1:1 |
| `blk.N.attn_norm.weight` | `lm.layers.N.input_layernorm.weight` | 1:1 |
| `blk.N.ffn_norm.weight` | `lm.layers.N.post_attention_layernorm.weight` | 1:1 |
| `blk.N.attn_qg.weight` | `lm.layers.N.self_attn.q_proj.weight` | 1:1 (query **and** gate, already interleaved: `modeling_qwen3_5.py:780-787`) |
| `blk.N.attn_k/v.weight` | `lm.layers.N.self_attn.k_proj/v_proj.weight` | 1:1 |
| `blk.N.attn_q_norm.weight`, `attn_k_norm.weight` | `lm.layers.N.self_attn.q_norm/k_norm.weight` | 1:1 |
| `blk.N.attn_output.weight` | `lm.layers.N.self_attn.o_proj.weight` | 1:1 |
| `blk.N.ssm_in.weight` | `lm.layers.N.linear_attn.in_proj_qkv.weight` + `in_proj_a.weight` + `in_proj_b.weight` | **three-to-one CONCAT** (`map_concat`, `rad_block_gdn.h:142-149`) |
| `blk.N.ssm_z.weight` | `lm.layers.N.linear_attn.in_proj_z.weight` | 1:1 |
| `blk.N.ssm_conv1d.weight` | `lm.layers.N.linear_attn.conv1d.weight` `[C,1,4]`→`[C,4]` | rank view only; `rad_enc_view` collapses trailing dims (`abi/rad_encoding.h:244-252`), so the logical shape matches |
| `blk.N.ssm_a_log` | `lm.layers.N.linear_attn.A_log` | 1:1 (it is already a log: `modeling_qwen3_5.py:531-533, 616`) |
| `blk.N.ssm_dt.bias` | `lm.layers.N.linear_attn.dt_bias` | 1:1 |
| `blk.N.ssm_norm.weight` | `lm.layers.N.linear_attn.norm.weight` | 1:1 (plain gain, initialized ones: `modeling_qwen3_5.py:202-217, 534`) |
| `blk.N.ssm_out.weight` | `lm.layers.N.linear_attn.out_proj.weight` | 1:1 |
| `blk.N.ffn_gate_up.weight` | `lm.layers.N.mlp.gate_proj.weight` + `up_proj.weight` | **two-to-one CONCAT** (`rad_block_mlp.h:52`) |
| `blk.N.ffn_down.weight` | `lm.layers.N.mlp.down_proj.weight` | 1:1 |
| `pad_eop.weight` (H4 extra) | rows `pad_token_id`, `eop_token_id` of `lm.embed_tokens.weight` | row view, not a checkpoint tensor |

Every map entry must be declared **before** the weight it names (`rad_block_gdn.h:145-147`,
`tests/duplex_plugin/radtest_duplex.cpp:98-104`, and qwen35's order at `qwen35_bf16.cpp:302-320`);
`map_copy(b, declared, checkpoint_name)` is the helper.

### 6.2 Dtypes

| tensor family | tiny | real (`header.json`) | declaration |
|---|---|---|---|
| linears, embedding, tied head, MLP, conv1d | F32 | BF16 | the checkpoint's own dtype — detect it with `weight_codes_dtype` after mapping `token_embd.weight` (`rad_arch.h:424-427`); a `RAD_F32` result selects the `"f32"`/`RAD_F32` pair for `geom_from`, anything else `"bf16"`/`RAD_BF16` |
| norms, `A_log`, `dt_bias`, q/k norms, final norm | F32 | BF16 | `RAD_F32` is safe for both: exact on the tiny, an exact bf16→f32 *widen* on the real (`core/build/rad_builder.cpp:210-218`), which is what qwen35 does (`rad_block_gdn.h:156, 170-176`) |
| activation buffers | F32 | BF16 | `g.act_dtype` |
| full-attention KV cache | F32 | BF16 | group `dtype`; f32 is needed on the tiny to reproduce the f32 reference cache (`haru.py:80-81`, model dtype f32) |
| GDN recurrent state | F32 | F32 | `RAD_KV_LINEAR`, `mamba_ssm_dtype: float32` (`config.json`) |
| GDN conv state and kkt plane | F32 | BF16 | `RAD_KV_CONV` dtype; the kkt buffer follows the activation dtype (`rad_block_gdn.h:282-284`) |

The operand-combination string (`g.dtype`) is `"f32"` on the tiny and `"bf16"` on the real; libref
selects a plain f32×f32 kernel for it (`libref/ref_gemm.cpp:70-91`), and all the GDN/attention ops
read their operands through `ldt`/`stt`, so their arithmetic follows the tensor dtypes
(`libref/ref_common.h:126-221`). For the full-attention cache, libref's registry *description*
says "bf16 or fp8 kv cache" (`libref/ref_registry.cpp:2601`), but the row's only constraint is
`block_size >= 1` (`:445`), and both `attn_paged` and `kv_store` read the cache with `ldt`/`stt`
(`libref/ref_attn.cpp:199-200, 608-612, 619-630`), so an f32 cache is accepted by the
implementation. **This is the
kind of mismatch I cannot settle without running the selector.**

*Not verified:* that `weight_codes_dtype` reports F32 for a plain safetensors source before any
weight declaration (the helper only needs the map, `rad_arch.h:418-427`). If it does not, pass the
dtype as an explicit `HaruLmOptions` field set by the plugin.

*Not verified:* that **one** declared weight can feed both `embed_lookup` and `logits_gemm`.
qwen35 declares two (`token_embd.weight`, `output.weight`) over one source for tied models; using
one would avoid a duplicate 1.27 GB embedding on the real model, but no existing plugin exercises
it. If the builder or a kernel row objects, fall back to qwen35's two declarations.

---

## 7. Outputs

- **The trunk's product** is the final normed hidden `b_h [n_tok, n_embd]`, one row per input row
  (`rmsnorm`, `wadd = 1` — verified numerically: normalising `prompt.lm_layer3` with
  `lm.norm.weight` as `(1+w)` reproduces `prompt.hidden_last` to 1.2e-7; qwen35 uses the same
  `final_wadd = 1.0f`, `qwen35_bf16.cpp:155-157, 426-430`). That row is exactly what the MeanFlow
  head takes (`haru/session.py:159-162`).
- **Logits** are `hidden @ embed_tokens.weight.T` in f32 (`haru.py:113-115`): `logits_gemm` over
  the tied table (`docs/OPS.md:2633`), output buffer f32 (as qwen35 declares its logits buffer,
  `qwen35_bf16.cpp:293-295`) but written to the block's own buffer, not `rad_declare_logits`.
- **The PAD/EOP decision** is recomputed in f32 from the same rows: `hidden.float() @
  w[[pad,eop]].float().T` (`haru.py:117-123`; `haru/session.py:148-151`). On the tiny this is the
  same buffer; on the real it needs an f32 activation row and an f32 two-row view of the embedding
  table. Declare the two-row view with `map_rows` (rows `pad_token_id`, `eop_token_id`) at
  `RAD_F32` (exact widen on the real) and a `gemm_nt` `M=1 N=2 K=n_embd dtype="f32"`; or, for the
  real, `gemm_nt_f32` with the bf16 hidden and the f32 weight (`docs/OPS.md:794, 868-870`).
- **Per-layer taps for debugging.** If the block is given a persistent `tap` buffer
  `[n_layers, n_embd]`, it issues a `cast` of each layer's just-written `b_x` row into row `l`
  after that layer. That is exactly the reference's hook: `gen_haru_tiny.py:459-463` captures
  `layer(...)`'s `o[:, -1]`, and `:512-518` stores them as `prompt.lm_layerN`; the frames capture
  the same for `t = 1..3` (`:612-614`). The block should expose this as an opt-in tap (a zero
  handle = no tap) so production pays nothing.
- **Through the tunnel:** the test plugin writes the current row's output into `audio_out[lane]`
  and sets `status = 0` (the driver reads `audio_out`, `text_out` and `status` back only on STEP,
  `core/gen/duplex_driver.cpp:887-892`; `status = 0` means audio present,
  `core/gen/duplex_driver.h:51-60`). The 64-float hidden goes at `[0,64)`; the taps at `[64,320)`;
  a 512-float logit row and the two f32 PAD/EOP values still fit in the 3200-sample buffer. The
  harness's `Lane::run(tensor, n_out, out)` returns the first `n_out` floats
  (`tests/HARU_HARNESS.md:58-65`).

---

## 8. Test plan

### 8.1 Tenants and the order

One test plugin (`radtest_haru_lm`, arch id `"haru"`, quant `""`) with **one** duplex tenant.
A program can declare only one duplex tenant: `Builder::declare_duplex_tenant` refuses a second
(`core/build/rad_builder.cpp:2072-2083`), and the engine registers one backend per declared name.
`max_lanes = 1`.

Declaration values for the test tenant: `max_prompt = 1` (or 64 for the later chunked check),
`max_ref_frames = 0` (no references; `ref_rate`/`ref_hop` may then be 0,
`core/build/rad_builder.cpp:2136-2142`), `n_audio_in = 0`, `text = 0` (so no `text_out` buffer),
`sample_rate = 24000`, `samples_per_frame = 3200`, `frame_rate_num/den = 15/2`, `latent_dim = 64`,
`channels = 1`. The probe declares `gen_duplex = 1`, `gen_rows = 64` (the largest pass; it also
sizes `max_tok`) and `gen_ctx = 128` (the lane's bound; the test uses 61 rows). These are the
`tests/duplex_plugin/radtest_duplex.cpp:22-44, 135-155` shapes with different numbers.

The one reply carries everything inside the 3200-float tunnel, with the primary comparison first:

| reply floats | content | goldens |
|---|---|---|
| `[0, 64)` | final normed hidden of this row | `prompt.hidden_last`, `frame1..24.hidden` (and `frame0.hidden`, which equals `prompt.hidden_last` — verified, max diff 0.0) |
| `[64, 320)` | the 4 layer taps, layer 0 first | `prompt.lm_layer0..3` on the last prompt step; `frame1..3.lm_layer0..3` on the first three frame steps |
| `[320, 832)` | 512 logits | `frame0..24.logits` |
| `[832, 834)` | f32 PAD/EOP logits | `frame*.logits_fp32_pad_eop` |

The test reads the first 64 floats on every step (the "64 out" of the brief); the tap/logit regions
are read only on the steps that have goldens. If the reviewer wants the wire to be strictly 64-out,
the same sources can be compiled as a second `archtest` target with a debug macro that suppresses
regions 2-4; that is a build choice, not an ABI one.

The driver requires at least one PROMPT row (`core/gen/duplex_driver.cpp:411-416`) and starts STEP
positions at the prompt length (`:966-972`), so the row order must be:

1. **PROMPT: exactly one row, golden prompt row 0.** Its embedding is the token embedding of
   `prompt.tokens[0] = 482`, which is `prompt.embed[0]` exactly (verified). Get token 482 through
   the template: `tmpl = "<|im_start|>{card}"`, empty card; the server tokenises with special
   tokens parsed (`core/server/duplex.cpp:262-275`) and the tiny tokenizer has 482 =
   `<|im_start|>` (read from `tests/data/haru/haru_tiny/tokenizer.json`). If that rendering is
   not one id, fill a constant i32 482 into a persistent buffer and `embed_lookup` from it instead.
2. **36 STEPs, golden prompt rows 1..36**, each with `prompt.embed[i]` in `audio_in[0][0..63]`.
   Step i runs at position i, so the last prompt row runs at position 36, as in the reference.
3. **24 STEPs, frames t = 1..24**, each with `frame{t}.embed` as 64 floats. Step 37 is position 37,
   which is the reference's position for frame 1 (`haru/session.py:131-143`).

Total 60 STEPs; the reply to STEP 36 is `prompt.hidden_last`, and reply 36+t is `frame{t}.hidden`.
No row is ever in a multi-row pass, matching the brief's "37 prompt rows as 37 single-row steps"
with the one PROMPT row counted as row 0.

*Deviation to approve:* the literal reading "37 STEPs" would need the driver to start STEP
positions at 0 after a zero-row prompt (open refuses T < 1), or a plugin-side position offset that
would desynchronise rope from the KV slots. I recommend the 1 + 36 + 24 order above and flag it as
question 2.

### 8.2 Per-step packing

Each `Lane::step` sends 3200 f32 (`tests/HARU_HARNESS.md:58-63`); the test builds
`in[0..63] = embed`, zeros after. The plugin reads the 64 from
`bcol_at(audio_in, lane, samples_per_frame, 0, 64, 1)` and `cast`s them into `b_x`. The reply is
3200 f32; its first 64 are the hidden and the rest are the debug regions of 8.1. One run of the 60
steps supplies every golden, so all comparisons share one lane and one deterministic sequence.

The engine is started the way `tests/duplex_plugin_test.cpp:109-117` starts it: the real
`radiance` binary, `--kernels libref`, `--debug-accept-reference-kernels` (libref is a reference
library, `docs/PLUGIN.md:70`), and `--http-threads 4` (the engine's minimum for that flag). The
harness API owns this; H4 must not start its own.

### 8.3 Which goldens

- `prompt.tokens[0] = 482`, `prompt.embed[0]` (input sanity; both read from the golden set).
- `prompt.hidden_last`, `prompt.lm_layer0..3` at STEP 36.
- `frame{t}.hidden` for t = 1..24 (t = 0 from `prompt.hidden_last`).
- `frame{1,2,3}.lm_layer0..3`.
- Optionally `frame{t}.logits` (512) and `frame{t}.logits_fp32_pad_eop` (2) from the logits
  tenant; the README's "LM trunk" row asks for per-layer outputs and the final hidden
  (`tests/data/haru/README.md:227-231`), so logits are a bonus, not a gate.
- The per-layer taps are defined as the residual stream after the whole layer (norm + mixer or
  GDN + residual, then norm + MLP + residual; `gen_haru_tiny.py:459-463` hooks
  `model.lm.layers[i]`, whose return is that stream — `modeling_qwen3_5.py:899-909`).

### 8.4 Tolerance

Start from `Tol{1e-4, 1e-4}` as the harness requires (`tests/HARU_HARNESS.md:81-82`) and record the
measured maximum in the test. Reasoning, all on f32:
- the embedding input is a bit-exact copy;
- norms, elementwise ops and the gate are f32 with the same formulas;
- `gemm_nt` accumulates f32 but in a different order from torch's reduction; at K = 64/128/256 the
  per-result difference is a few ulps (~1e-7 relative);
- softmax/attention is f32 in both, over the same keys in the same order (`libref/ref_attn.cpp:146-151`);
- the GDN chunked/algebraic path is the largest unknown: the reference uses
  `torch_chunk_gated_delta_rule` on CPU (`modeling_qwen3_5.py:618-643`), libref its own f32
  formulation (`libref/ref_gdn.cpp:385-665`), and H4 runs the prompt row-by-row
  (`gdn_conv_update` + `gdn_recurrent_update`, `:308-384, 666-917`) instead of one 37-row chunk.
  Mathematically the two forms agree; numerically they will not be bit-identical.
- four layers of such composition normally stay near 1e-6 relative on f32, which is two orders
  below 1e-4. If the row-by-row prefill is the dominant error, 1e-4 may need evidence-based
  loosening — but only after the reviewer measures. **I did not measure anything.**

### 8.5 Chunked-prefill equivalence, later

Add after the row-by-row test is green, with the same goldens:
- **Preferred:** make the duplex PROMPT carry per-row embeddings, so the block runs a true 37-row
  prefill (gdn_conv_prep/kkt/scan with T=37, attn_paged with q_len=37 — the reference's path). The
  ABI already has `RadBatch::mm_rows/mm_embd` (`abi/rad_runtime.h:229-242`), but the duplex driver
  does not stage them and `mm_embd` is bf16; this needs a core change (and an f32 form for the
  tiny), so it is a follow-up.
- **Cheap:** run the same 37 rows through a second lane with a 37-row PROMPT whose handler reads a
  test-only embedding matrix from a plugin-private buffer filled by the first 37 STEPs of an
  "upload" lane. This compares C++ chunked vs C++ recurrent without a new ABI, but uses cross-lane
  state, which the wire contract discourages (`docs/DUPLEX-ENGINE.md:105-113`).
- **Minimal:** compare the row-by-row run against a locally computed 37-row run of the same block
  and report the maximum difference (no golden needed), to know whether the reference comparison
  is limited by the path or by the C++ arithmetic.

---

## 9. Gaps, risks, size, order, questions

### 9.1 Missing ops or semantics

No op is missing for the trunk. Missing **semantics/plumbing**:

1. Duplex PROMPT has no per-row embedding channel (5.1 and 8.5).
2. The engine's metadata reader has no `lm.*` alias (5.1); without it neither the tiny test nor a
   converted real checkpoint loads.
3. `RadBatch::token_ids` is written 0 on every STEP (`core/gen/duplex_driver.cpp:844`); a code
   path that falls back to a token lookup there would silently use token 0.
4. The duplex driver never sets `n_out`/`out_ids`; logits must be issued for the pass's own rows.
5. `rad_declare_logits` is incompatible with a duplex tenant (section 0, item 3); use a plain f32 buffer.
6. The shared bf16 blocks cannot declare f32 weights (1.3).
7. `audio_out` is only read when `status == 0`; the tenant must fill `status` every step
   (`core/gen/duplex_driver.cpp:1000-1024`).

### 9.2 Size

- `arch/haru_common/haru_lm.h`: the struct, options, the declare/step interface; ~80-120 lines.
- `arch/haru_common/haru_lm.cpp`: geometry/metadata, maps and weights, groups/bindings, buffers,
  ops, the mirrored block issue code, taps/logits; ~350-450 lines.
- `tests/haru_plugins/lm/`: CMakeLists (kind `archtest`, like `tests/duplex_plugin/CMakeLists.txt`)
  and the test plugin (~200 lines).
- `tests/haru_lm_test.cpp`: ~250 lines; depends on the harness API landing.
Roughly 1.5-2 days for someone who can build and run, plus review iterations.

### 9.3 Risks

1. **Row-by-row vs chunked prefill** (8.4) may make `prompt.hidden_last`/`lm_layer*` miss 1e-4.
   This is the first thing the reviewer should measure; if it misses, the primary test must use a
   true 37-row PROMPT and 8.5 becomes blocking.
2. **Loadability** (5.1) blocks every run until the alias or a prepared config exists.
3. **The f32 path is new.** No shipped plugin runs f32 activations through these ops; libref's
   dtype-generic kernels say it works, but the selector and the builder's dtype checks are
   untested here.
4. **Numerical details of GDN** (decay, kkt, l2 norm) against FLA's CPU path; the reference's
   `torch_chunk_gated_delta_rule` may not be the same algebraic variant libref implements.
5. **Memory** on the real model is dominated by the full-attention KV (≈422 MiB/lane at 30
   minutes); this is a deployment risk, not an H4 one.
6. **bf16 production logits** differ from the reference's bf16-rounded matmul (the reference does
   `(hidden @ w.T).float()`; libref writes f32 from an f32 accumulator). The PAD/EOP path is the
   one the reference deliberately recomputes in f32.

### 9.4 Proposed order of work (each independently checkable)

1. **Loadability** (one-line alias, or the harness's prepared config) with a test that starts the
   engine on the tiny LM and prints the dimensions. No block code yet.
2. **`haru_lm` declare only**: geometry + maps + groups + buffers + weights + ops, plus a test
   plugin that declares a duplex tenant and returns `ready`; check `/duplex/v1/info`, the graph
   dump (`--debug-graph`) and that declare does not refuse the f32 checkpoint.
3. **STEP only, one token row**: the test's 1-row PROMPT + 36 prompt STEPs; compare
   `prompt.hidden_last` first (the path-risk measurement of 8.4).
4. **All prompt/frame hidden**: add the 24 frame steps and `frame{t}.hidden`.
5. **Taps and logits**: the 320/514 tenants and their goldens.
6. **Chunked-prefill check** (8.5) and the real/bf16 path.

Steps 1-2 are small enough to review without running the model; 3 is the first real numeric gate.

### 9.5 Questions to answer before coding

1. Does the reviewer accept the `lm.` metadata alias in `core/format/checkpoint.cpp` (5.1), or
   should the harness prepare a promoted-config copy? Who owns that change?
2. Is 8.1's order (1 PROMPT row = golden row 0, then 36 + 24 STEPs) the intended reading of "37
   prompt rows as 37 single-row steps"? If not, which position-offset or driver change is
   preferred?
3. Is the constant `prompt.tokens[0] = 482` / the `<|im_start|>` template acceptable in a test
   plugin, or must the first row come from the card text only?
4. Is the one-tenant reply layout of 8.1 (hidden at `[0,64)`, taps/logits after it) acceptable,
   or must the tunnelled reply be strictly 64 floats and the debug outputs come from a second
   `archtest` target built from the same sources?
5. For the f32 tiny path, is copying the blocks into `haru_lm` acceptable (1.3), or should the
   shared headers gain a dtype knob?
6. Are the real deployment's `max_prompt`, `gen_ctx` and `max_lanes` known, or may H4 declare
   placeholder values (e.g. 1400 / 13,500 / 1) and leave the production choice to the assembled
   plugin?
7. Is comparing logits in H4 in scope, or is hidden + layer taps enough (the README's "LM trunk"
   row names only those)?
8. Which harness owner guarantees `tests/haru_harness.h`? H4's test is written against the API in
   `tests/HARU_HARNESS.md`, which does not exist as code yet in this worktree.

### 9.6 What I could not verify

- Anything that needs building or running: the graph declares and resolves, the f32 selector bands,
  every numeric comparison, and the KV manager's exact rounding.
- The row-by-row-prompt numerical equivalence (the central unverified claim).
- That the engine tokenises `<|im_start|>` to exactly `[482]` through the C++ tokenizer and the
  server's `parse_special` path (I read the tokenizer data and the bridge, not a run).
- That `weight_codes_dtype` reports F32 before a weight declaration, and that one weight handle
  can feed both `embed_lookup` and `logits_gemm`.
- That a safetensors checkpoint served directly tolerates unclaimed tensors (the tiny checkpoint
  also contains codec/head tensors the LM plugin does not map).
- The exact real-checkpoint conversion path (`rad-convert` on a Haru directory) and how the
  `real/config.json` geometry behaves across all 32 layers.
- Whether the reference's FLA chunked GDN and libref's chunked/recurrent forms agree in f32 to any
  stated bound.
