# Q6_K dense: architecture declarations still needed (not committed)

Branch: `kernel-q6k-dense` (jsaigou/radiance, PUBLIC fork). Kernel/shims are in
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
