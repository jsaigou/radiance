# Haru H3: checkpoint coverage, tied head and latent scalars

This note records the decisions behind `tests/data/haru/check_haru_names.py`,
`data/recipes/haru-bf16.recipe` and the `haru_names_check` ctest. It is a companion to
`docs/HARU-H3-DESIGN.md`; the design note describes the checker, this note fixes the model-side
choices.

## 1. Coverage

The checker parses the declarations in `arch/haru_common/{haru_head,haru_proj,haru_lm,haru_codec}.cpp`
(the `plain_weight`, `map_copy`, `map_concat`, `map_rows`, `declare_weight` and `decl_w` call
forms), expands the layer and stage loops from each config, and compares the checkpoint-side names,
shapes and dtypes with both manifests. Current result:

| checkpoint | header tensors | declared source tensors | logical weights |
|---|---|---|---|
| tiny (`tests/data/haru/haru_tiny/`) | 261 | 261 | 252 |
| real (`tests/data/haru/real/header.json`) | 1022 | 1022 | 943 |

The difference between logical weights and source tensors is intentional:

* `haru_lm` joins q/k/v/a/b into one `ssm_in` weight (`map_concat`), and gate/up into one
  `ffn_gate_up` weight; there are 426 LM source tensors but 347 LM logical declarations on the
  real config.
* `pad_eop.weight` is a logical declaration with no checkpoint tensor of its own (section 3).

Counts per block are printed on every run (`lm`, `head`, `proj`, `codec`). The parser fails closed
on an unknown declaration form and asserts each block's count against a config-derived expected
count, so a parser that finds zero (or a stale count) fails rather than passing silently.

Two shape rules need an explicit note, because the checkpoint layout and the declared layout differ
in element order but not in element count:

* `latent_scale` and `latent_bias` are shape `[]` in the real header and `[1]` in
  `haru_proj.cpp:56-57`.
* the LM's GDN `conv1d.weight` is `[C, 1, K]` in the checkpoint and is declared `[C, K]`
  (`haru_lm.cpp:481`).

The builder allows a same-element-count reshape of one tiled scalar plane
(`core/build/rad_builder.cpp:244-257`), so both are accepted and reported as reshapes. All other
source shapes are compared exactly.

## 2. Tied head

`lm.embed_tokens.weight` is one checkpoint tensor and `config.tie_word_embeddings` is true.
`haru_lm` declares it once as `token_embd.weight` (`haru_lm.cpp:207-208`) and points the embedding
lookup, the tied logits GEMM and the PAD/EOP row map at that one weight. `haru_proj` takes the
declared handle as an argument and never declares a second table (`haru_proj.cpp:37-43`). There is
no `lm_head.weight` in the checkpoint and the checker does not expect one. The real header's 1022
tensors therefore include exactly one token table.

## 3. `latent_scale`, `latent_bias` and PAD/EOP

`latent_scale` and `latent_bias` are real checkpoint tensors, stored as BF16 scalars in the real
model (shape `[]`). They are declared as ordinary F32 weights of shape `[1]` and applied in the
reference order

    scale(raw) = (raw + latent_bias) * latent_scale

(`haru_proj.h:6-9`, `haru_proj.cpp:56-57,100-110`). They are not folded into the projectors or the
codec at conversion; the checker requires both at their real names. With a BF16 source the loader
widens them exactly to F32.

`pad_eop.weight` is not a checkpoint tensor. `haru_lm` makes it with `map_rows` from the `pad_id`
and `eop_id` rows of `lm.embed_tokens.weight` (`haru_lm.cpp:224-228`), declares it F32, and uses it
for the reference's F32 PAD/EOP logits. The checker verifies that the source table is declared and
present, reports the virtual weight as `[2, hidden]` F32, and excludes it from the name comparison.

## 4. Precision and the recipe

The real checkpoint is BF16 in all 1022 tensors. The blocks declare:

* `haru_lm`: linears and the token table at the token table's dtype (BF16 real); norms, `A_log`,
  `dt_bias`, `output_norm` and `pad_eop` at F32;
* `haru_head`: every weight F32;
* `haru_proj`: every weight and both latent scalars F32;
* `haru_codec`: every weight at the config's `dtype` (BF16 real).

`core/build/rad_builder.cpp:207-227` accepts an exact BF16/F16 -> F32 widening, which covers all
of the F32 declarations above. The checker reports 197 such widenings on the real header and none
on tiny. No declaration narrows F32 to BF16 on either checkpoint, so the checker treats such a
narrowing as an error.

`data/recipes/haru-bf16.recipe` contains one rule, `* cast dtype=bf16`. On the real checkpoint it
is an exact no-op; on an F32 repack it rounds once to BF16 before the F32 declarations widen. The
rule uses the same wildcard and `cast dtype=` syntax as `hunyuan3d-2-bf16.recipe`,
`minimax-music3-bf16.recipe` and `anima-bf16.recipe`.

## 5. How to run it

    python3 tests/data/haru/check_haru_names.py
    python3 tests/data/haru/check_haru_names.py --self-test

`--self-test` builds the manifest for the real header (or tiny if the real files are absent),
renames one tensor, changes one shape and changes one BF16 dtype in an in-memory copy, and fails
unless the checker reports each mutation. ctest registers both commands as `haru_names_check` and
`haru_names_check_selftest` inside the existing `RAD_PYTHON3` block in `tests/CMakeLists.txt`.

## 6. What this does not establish

This is a static check: it cannot prove that the C++ declarations compile, that the builder accepts
every name at declare time, that `rad-convert` produces a valid container from the recipe, or that
any kernel is numerically correct. It reads a header (names, shapes, dtypes, offsets), not tensor
values, and it does not run the Python reference. Those remain the reviewer's Linux build,
`rad-convert --plan-only`/conversion, and the existing golden tests.
