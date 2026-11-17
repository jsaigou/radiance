# Q6_K dense: architecture declarations still needed (not committed)

Branch: `kernel-q6k-dense` (jsaigou/radiance-strix, PUBLIC fork). Kernel/shims are in
`libhalo/halo_gemm_q6ka16.hip`, the stored form in `halo_q6k.*`. Nothing in `arch/` was changed.

## Names the architecture must declare

- dtype string `q6k`, `group` 256.
- Stored layouts: `halo.q6k.w.tile16x64` (codes i6, operand 2 = `b`),
  `halo.q6k.s.tile16x16` (scale i8 per 16, operand 3 = `b_scale`),
  `halo.q6k.d.tile16x256` (scale.1 f16 per 256, operand 6 = `b_ref`).
  The third plane rides the schema's existing `b_ref` weight aux; there is no new operand.
- Rows: `gemm_q6ka16_nt_m8` (M <= 8) and `gemm_q6ka16_nt_wmma` (any M), both op `gemm_nt_q`,
  constraint K % 256 == 0, N % 16 == 0, group 256, dtype `q6k`. The activation is bf16, read
  directly; there is no activation quantiser.

## Trunk linears (`LinearFP8`, arch/common/rad_fp8.h)

Both architectures use `LinearFP8` for the dense trunk:

- `arch/qwen4exp_fp8/qwen4exp_fp8.cpp` (Qwen3.8-Flash-Next) and
  `arch/qwen35_fp8/qwen35_fp8.cpp` as built by `arch/qwen35moe_fp8/` (Qwen3.6 MoE trunk).

Needed: a Q6_K branch in `LinearFP8::probe` and `LinearFP8::declare`, chosen from the weight's
encoding (`affine`, codes i6 [1x1], scale i8 [1x16], scale.1 f16 [1x256]):

- `reads_x = true` (the kernel takes the bf16 plane; skip `quant_act_fp8`/`quant_act_i8g`).
- Declare `gemm_nt_q` with `group=256`, `dtype="q6k"`, and three views of the weight:
  `codes` -> `b`, `scale` -> `b_scale`, `scale.1` -> `b_ref` (`decl_view` each, row-sharded with
  the weight as usual; `b_ref` must be declared as a weight aux so the shim receives operand 6).
- Row slicing is fine: both rows read whole 256-K blocks and 16-row tiles, so a row shard must
  keep N a multiple of 16 and a column shard K a multiple of 256 (refuse otherwise in declare).
- The fused `gate|up` / `qkv|z` projections use the same row with the whole fused weight; the
  `pb1`/`pb2` parts are not read by this row (single activation, no copy).

## lm_head

`output.weight` is declared through `logits_gemm` (dtype `g.dtype`), not `gemm_nt_q`, in both
architectures. The dense Q6_K rows only serve `gemm_nt_q`, so an lm_head at Q6_K needs either:

- a new `logits_gemm` row over the same three planes (a thin wrapper over
  `halo_gemm_q6ka16_nt_mv/_wmma` writing f32 logits), or
- the trunk to declare `gemm_nt_q` at `q6k` into `b_hout` and a separate logits op over it.

Until one lands, a Flash-Next shard whose `output.weight` is Q6_K (shard 2) cannot bind its
lm_head; the trunk linears can.

## format_of / kServed (arch/common/rad_block_moe_fp8.h)

None. `MoeFP8::format_of` and `kServed` name the *grouped* MoE GEMM's formats, and the task says
not to change the grouped kernel. Adding `q6k` there would advertise a format no grouped kernel
can read. Q6_K experts (the Qwen3.6 UD down experts) stay on their current format; serving them
needs a separate grouped Q6_K kernel and then `format_of` index 10 + `kServed[10] = "q6k"`.

## Tuning observation

At M = 1..8 the WMMA row is often faster than the matvec on these shapes (lm_head M=8:
2.94 ms vs 5.01 ms), because one wave still reads a 48-byte row and the activation is reused.
The planner is free to pick `gemm_q6ka16_nt_wmma` for M <= 8; the matvec row remains the
bit-exact-reference form.

## Dense wiring landed (branch `kernel-dense-wire`, 2026-11-15)

`LinearFP8` (arch/common/rad_fp8.h) now reads both ggml dense forms from the weight's encoding,
so a Flash-Next container keeps the UD map's Q6_K/Q8_0 trunk planes:

- **Q6_K** (`i6*i8[1x16]*f16[1x256]`): `gemm_nt_q`, `dtype=q6k`, `group=256`, over the input's
  bf16 plane. Views are `codes` -> `b`, `scale` -> `b_scale`, `scale.1` -> `b_ref` (operand 6);
  no activation quantiser. Row parts stay whole 16-row tiles; a column shard must be whole
  256-value blocks. Covers the 70 source tensors fused into 35 `ssm_inz`, 12 each of
  `attn_qg`/`attn_k`/`attn_v`/`attn_output`, the shared expert's 47 `ffn_gate_up_shexp`, and
  `output.weight` (mapped, below).
- **Q8_0** (`i8*f16[1x32]`): `gemm_nt_q`, `dtype=q80a16`, `group=32`, the two planes as `b` and
  `b_scale`. Covers the 36 `ssm_out`, the 48 `ffn_down_shexp`, the PLE layer's
  `ple_key`/`ple_value` projections, and layer 2's whole block (the map's only Q8_0
  `ssm_inz` + `ffn_gate_up_shexp`). `PleLayer` also declares those projections at Q6_K (or bf16)
  when the container holds those planes. The 96 + 1 connection `*_hc_down/up` matrices stay on
  `hc_read`, which does not read Q8_0 (mapped, below).
- The grouped MoE declarations are unchanged: `format_of`/`kServed` still name only the grouped
  kernels. Q6_K/Q8_0 *expert* planes remain unserved.

### lm_head: mapped, not a new logits row

`logits_gemm` carries one weight operand, so Q6_K's three planes cannot be bound there (and the
MTP head reads the same `w_lm_head` through `logits_gemm`/`logit_rerank`). The arch now refuses a
Q6_K/Q8_0 `output.weight` at declare instead of loading an unservable op. `tools/ud_map`'s mirror
recipe maps it to the served int8 head (`rtn codes=i8 group=128 scale=bf16`), the same form the
production `qwen4exp-w4nl64-i8-hc8m.recipe` uses:

- output.weight Q6_K 497 MiB -> i8 615 MiB, **+118 MiB** (+0.115 GiB on the 89.00 GiB container);
- the head is then exact at its int8 plane values, but not UD's Q6_K codes. A faithful Q6_K head
  needs a new `logits_gemm` layout that packs the three planes into one row (or f32-output
  `gemm_nt_q` rows plus a logits op); not built here because the map's head would cost the MTP
  path the same work.

### Q8_0 connection mixing is mapped, not wired

`hc_read` serves bf16 or E4M3 rows, not Q8_0; the 96 + 1 connection's `*_hc_down/up` Q8_0 planes
are mapped by `ud_map` to the served `fp8_e4m3` + f32-per-group rows (down group 128, up group 80)
— the production recipe's form, essentially the same bytes. Q8_0 stays exact for the dense GEMMs;
the connection rows lose their 8-bit codes to E4M3.

### Issue-time wiring (branch `kernel-q6k-shared`, 2026-11-15)

The declaration above was right, but `LinearFP8::declare` kept the branch in `const bool` locals
that shadowed its `q6k`/`q80` members, so `step` never saw them: the plan declared `gemm_nt_q` with
`codes`, `scale` and `scale.1` (operand 6, `b_ref`) while the issue fell through to the fp8 form and
passed no `b_ref`. The Q6_K row checks all five operands (`q6k_args` reads `a`, `b`, `b_scale`, `y`
and `b_ref`), so the first Q6_K GEMM of the step -- the shared expert's fused `ffn_gate_up_shexp` at
`N=16384 K=2560`, M=101 -- was refused with `RAD_E_INVAL` ("invalid argument"). The shared arm was
already `LinearFP8` (`sh_gate_up`, `sh_down` in `rad_block_moe_fp8.h`), so no separate shared row
existed to fix; the trunk, the shared gate/up and the shared down were all affected.

`declare` now stores the flags on the members. The arch test issues one step and checks every Q6_K
issue carries all three weight planes with `scale.1` on operand 6 (and Q8_0 none), including the
shared expert's own issue at M=101.

### Out of scope (not in this map)

Q8_0 `token_embd` does not occur (the map's embedding is IQ4_XS, L2) and the MTP layer is kept from
bf16, not Q8_0, so no Q8_0 embedding gather or grouped Q8_0 MTP kernel is needed here. The 2-bit
`mtp.draft_head` and the MTP layer's formats are the recipe's, not this lane's.

### Verification (Tohil, `~/radiance-test/src-dense-wire`)

- arch_test **316,183 checks / 223 cases**, 0 fail (adds the two Q6_K/Q8_0 declaration tests);
  ref_test **155,196 / 303**; host_experts_test **216 / 1**; relayout_parity **1,173 / 22**.
- Q6_K bench: layout **22/0**, check **142/0**, real **30/0**; Q8_0: layout **24/0**, check
  **148/0**, real **40/0** (matvec bitwise vs ggml dequantisation, WMMA within ulp).
- `rad-convert --plan-only` on the bf16 Flash-Next source with a recipe that keeps the map's dense
  planes: `ssm_inz`/`attn_qg` land on `i6*i8[1x16]*f16[1x256]`, `ssm_out`/`ffn_down_shexp`/the PLE
  projections on `i8*f16[1x32]`, and `output.weight` on `i8*bf16[1x128]`; 51,488 weights planned,
  nothing written.
