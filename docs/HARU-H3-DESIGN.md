# Haru H3: checkpoint name-map coverage and the bf16 recipe

Status: design note for step 1 of H3. **No code is written in this step.** The blocks already in
`arch/haru_common/` are `haru_head`, `haru_proj`, `haru_lm` and `haru_codec`; this note states how
their declarations will be checked against the two checkpoints and what the conversion recipe
will contain. I cannot build or run the C++ tree here, so nothing below is a test result.

The manifests used are `tests/data/haru/haru_tiny/model.safetensors` (261 F32 tensors, the test
checkpoint) and `.haru-ref/header.json` + `.haru-ref/config.json` (the real 1022-tensor BF16
checkpoint header and config). H3 will copy those two small real-header files to
`tests/data/haru/real/{header,config}.json` (about 112 KB) so the checker is reproducible without
the untracked `.haru-ref/` directory.

## 1. How the declared names, shapes and dtypes are extracted

The checker is a stdlib-only Python script, `tests/data/haru/check_haru_names.py`; it does not
import torch, safetensors or the C++ tree. It reads the four block sources and parses the
declaration forms they actually use:

| block | form | extra data it needs |
|---|---|---|
| `haru_head.cpp` | `plain_weight(b, "head....", { ... }, &w)` | none; names are already checkpoint names |
| `haru_proj.cpp` | `plain_weight(...)` plus the `audio(prefix)` helper and its `"%s.fc1.weight"`-style `snprintf`s | the two literal prefixes `model_audio_proj`, `user_audio_proj` |
| `haru_lm.cpp` | `map_copy(b, nm.f(F), nm.ckpt(C))`, `map_concat(b, nm.f(F), { C1, C2, ... }, 0)`, `map_rows(...)`, and the matching `decl_w(b, nm.f(F), dtype, shape, ...)` | per-layer names with `%d`; `lm.num_hidden_layers`; `lm.layer_types` |
| `haru_codec.cpp` | `declare_weight(b, name, ...)` where `name` comes from `nm.f("%s.weight", prefix)`; block prefixes and literal `codec.encoder/decoder.stem/head/down.*/up.*` | `codec.base_channels`, `codec.encoder_depths`, `codec.decoder_depths`; seven fixed stages |

The parser is deliberately fail-closed: an unrecognised `plain_weight`, `map_copy`, `map_concat`,
`map_rows` or `decl_w`/`declare_weight` call, an unexpanded format specifier, or a `map_*` without
a declaration (and vice versa) is an error, not a silently skipped line. The generated names are
expanded with the config: LM layers `0..n_layer-1` with each layer's type from `layer_types`
(`self_attn` for full attention, `linear_attn` for GDN, always `input_layernorm`,
`post_attention_layernorm` and the MLP); head layers `0..head.num_layers-1`; codec stages
`0..6` with `encoder_depths[s]`/`decoder_depths[s]` blocks. I chose source parsing over adding a
new C++ export because the export would duplicate the declaration lists and the blocks have no
such ABI today; the parser's fail-closed rule is what keeps the two from drifting silently.

For each declaration the script records the **checkpoint** name(s), not only the internal
`nm.f(...)` name. `map_copy` contributes its `nm.ckpt(...)` argument, `map_concat` contributes
every component (the logical weight is their row-concatenation), `plain_weight` and codec
declarations contribute the name itself, and `map_rows` contributes the row source used for the
tied PAD/EOP view. That source-name set is what is compared with the checkpoint headers.

Shapes are recomputed from the config, not copied from the header. The formulas are the ones the
blocks use: LM `q_dim = n_head*head_dim`, `qg_dim = 2*q_dim`, `kv_dim = n_kv*head_dim`,
`k_dim = n_k_head*head_k`, `v_dim = n_v_head*head_v`, `conv_dim = 2*k_dim + v_dim`,
`in_dim = conv_dim + 2*n_v_head`; GDN `in_proj_qkv = [conv_dim, hidden]`,
`in_proj_a/b = [n_v_head, hidden]`, `in_proj_z = [v_dim, hidden]`,
`conv1d = [conv_dim, 1, conv_kernel]`, `A_log`/`dt_bias = [n_v_head]`,
`norm = [head_v]`, `out_proj = [hidden, v_dim]`; attention
`q_proj = [qg_dim, hidden]`, `k/v_proj = [kv_dim, hidden]`, `o_proj = [hidden, q_dim]`,
`q_norm/k_norm = [head_dim]`; MLP `gate/up = [intermediate, hidden]`,
`down = [hidden, intermediate]`. Head and codec use the formulas in `haru_head.cpp:82-110` and
`haru_codec.cpp:95-181` (for the codec, channel width `base << stage` in the encoder and
`base << (6-stage)` in the decoder). A `map_concat` logical row count must equal the sum of its
component row counts. The tiny checkpoint checks the tiny config; the real header checks the real
config.

The readiness rules are:

* **Names.** The declared source-name union equals the checkpoint's tensor-name set. Tiny has 261
  and real has 1022. Both directions are reported: declared-but-absent and present-but-undeclared.
  Two entries are intentional exceptions, whitelisted by name and documented in the script:
  `lm.embed_tokens.weight` is the tied token table, declared once and reused as the output head,
  so there is no separate `lm_head` tensor; and `lm.pad_eop.weight` is a virtual F32 weight made
  by `map_rows` from two rows of that table, so it is declared but never a checkpoint tensor.
* **Shapes.** Every source tensor's header shape equals the formula's expected shape (for a
  concatenated weight, per component and in sum). A mismatch is printed with both shapes.
* **Dtypes.** Tiny is all F32, real is all BF16. Every declaration is F32 or BF16. Equal dtypes are
  exact; BF16 → F32 is the one allowed widening (`core/build/rad_builder.cpp:207-227`), and the
  script lists every widened declaration. F32 → BF16 is reported as a narrowing error. The codec's
  dtype is taken from the config's `dtype` (`bfloat16` real, `float32` tiny); the LM's is the
  token-table dtype resolved from the checkpoint, with F32/BF16 only.

The script exits non-zero on any mismatch and prints one line per finding. It is registered in
`tests/CMakeLists.txt` inside the existing `find_program(RAD_PYTHON3 ...)` block
(`tests/CMakeLists.txt:857-866`) as `haru_names_check`, with labels `unit;fast`, so ctest runs the
same static check the reviewer can run by hand.

## 2. The BF16 recipe

`data/recipes/haru-bf16.recipe` follows the other recipes' style: a comment naming the model and
the exact `rad-convert` command, then one rule.

```
*                                    cast dtype=bf16
```

The real checkpoint is already BF16 for all 1022 tensors, so the rule is an exact no-op there; it
makes "Haru is served at BF16" explicit and canonical. The blocks hard-code F32 for the LM norms,
the head, the projectors and the latent scalars; with a BF16 source those declarations are the
exact BF16 → F32 widening the builder already performs at load, so no digits move. The codec
declares from the config's `bfloat16` and stays BF16. If someone converts a repacked F32 model the
wildcard rounds once to BF16 before those F32 declarations widen, which is the intended BF16
serving precision rather than an accidental F32 container. No per-tensor rule is needed: there is
no tensor that the blocks declare at a third precision, no transposed layout, and no folded norm.

## 3. Tied head, latent_scale and latent_bias

These are settled here and will be written up in `docs/HARU-H3-NOTES.md`:

* **Tied head.** `lm.embed_tokens.weight` is one checkpoint tensor. `haru_lm` declares it once as
  `token_embd.weight` and points `embed_lookup`, the tied logits GEMM and `map_rows` at that one
  weight; `haru_proj` receives the handle rather than declaring its own table. The checker
  therefore expects the tensor exactly once, not a `lm_head.weight`.
* **`latent_scale` / `latent_bias`.** Both are real checkpoint tensors, declared in
  `haru_proj.cpp:56-57` as ordinary F32 weights of shape `[1]`, and applied as
  `(raw + latent_bias) * latent_scale` (`haru_proj.h:6-9`). They are not folded into the
  projectors at convert time; the checker requires both, at the header's `[1]` shape.
* **PAD/EOP.** `pad_eop.weight` is the one virtual declared weight: two rows copied out of the
  tied table at load and widened to F32 for the reference's exact F32 PAD/EOP logits. It is
  whitelisted from the name comparison and its source rows are checked.

## 4. What cannot be verified here

I cannot compile or run C++, run `rad-convert`, execute the Python reference, or read tensor
values: the real input is a header (name, shape, dtype, byte offsets) plus config, not the weights.
So this work can prove static name/shape/dtype coverage against the two manifests and the source
text; it cannot prove that the builder accepts the maps at declare time, that the recipe converts
to a byte-identical container for every tensor, or that any kernel/numeric path is correct. Those
need the reviewer's Linux build, a real conversion and the existing golden tests. The checker is
also not a substitute for a compile-time manifest: a declaration form I failed to recognise is
made a hard error, but a new form still has to be taught to the parser when a block changes.
