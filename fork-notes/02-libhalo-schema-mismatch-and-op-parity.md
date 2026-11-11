# libhalo is refused beside libavx (schema mismatches) and has 34 fewer ops than libr4d

*Not fixed in this unofficial fork. Not reported to, or confirmed by, the original author.*

## Summary

At commit `451b136` on gfx1151 the default kernel set does not start: the loader (correctly, since 1.16.0) refuses `libhalo` because two ops it
declares disagree with the schema `libavx` fixed first. Selecting `RADIANCE_KERNELS=libhalo` works around it, but then `libhalo` has no host
fallback for the ops it lacks.

## Reproduce

```
export RADIANCE_HOME=<build>/radiance_home        # libhalo, libavx, libr3d, libref, spechub all present
rad-info --plugins
E op 'mla_attn' has two different schemas: 'libavx' fixed it first, 'libhalo' disagrees (8 parameters vs 7) -- refusing 'libhalo'
```

After working around the first, the next one appears:

```
E op 'sample_logprobs' has two different schemas: 'libavx' fixed it first, 'libhalo' disagrees
  (operand 1 ('params' / 'params') has a different role) -- refusing 'libhalo'
```

## The two mismatches

1. `mla_attn`: `libr4d` and `libavx` gained an optional parameter `P_STRO("kv_dtype")` (E4M3 latent cache, 2026-10-28);
   `libhalo/halo_rows.cpp` `pMlaAttn` still has seven parameters.
2. `sample_logprobs`: `libr4d` and `libavx` declare operand `params` as optional (`RAD_OPD_IN, 1 /* absent: every row */`);
   `libhalo` declares it required. Flipping the declaration alone is not enough: `libhalo`'s implementation rejects an absent `params`
   (`params_ok` returns `RAD_E_INVAL` for null, the launcher refuses `!params`, and `logprobs_kernel` dereferences `sp[m]` unconditionally).
   The matching change is the one `libr4d` already has (`r4d_rad_sample_logprobs`): `if (pr) HALO_NEED(params_ok(pr, M));`, drop `!params` from the
   launcher's null check, and `if (sp && !(sp[m].flags & SP_LOGPROBS)) return;` in the kernel. Not applied or tested here. For `mla_attn`, a declared
   `kv_dtype` must also be honoured or refused by `libhalo`'s launcher, otherwise an E4M3 latent cache would be read as bf16.

There may be more behind these two: the loader reports only the first mismatch.

## Op parity

`libhalo` declares 246 schemas, `libr4d` 280. Missing from `libhalo` (34):

```
ar_rmsnorm_quant_i8g attn_paged_ahead attn_paged_anchor_window attn_paged_qrow_ahead attn_paged_qrow_tree
attn_paged_sink cond_aug dflash_markov_greedy dflash_markov_greedy_f32 dflash_tree gate_had_quant_i8g
gdn_conv_recurrent_update_had gelu_mul_quant_i8g gemm_nt_q_gated_wide gemm_nt_q_short grid_embed_cpe lili_attn
lili_cand lili_factor lili_walk logits_gemm_vocab mla_attn_paged ple_conv_bias rmsnorm_quant_i8g rmsnorm_rows
rope_rows router_sigmoid_group_topk seg_mean_bf16_sum seg_softmax_pool ssd_update swiglu_step vtile_crop
vtile_join vtile_plan
```

`libhalo/halo_stubs.cpp` still has 91 weak stub bodies (its header calls the line count "the parity scoreboard"). That list covers at least DFlash tree
drafters, attention sinks, LiLiCorr, the vision tile ops and the Gemma-4 per-layer-embedding ops, so those model families presumably cannot run on Strix
Halo yet. Qwen3.6-35B-A3B FP8 with MTP runs fine.

## Suggestion for the original project

Bring the two schemas in line, and add a check comparing every plugin's schemas pairwise, which would list all mismatches at once; or have the loader
print every mismatch instead of stopping at the first.
