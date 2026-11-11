# libhalo: MTP depth 3 aborts with "scratch buffer absent or too small" (q_len=4 decode, gqa 8)

*Fixed on branch `unofficial-fixes` of this unofficial fork. Not reported to, or confirmed by, the original author.*

## Summary

On Strix Halo (gfx1151, `libhalo`), serving a gqa-8 / head_dim-256 model with `--num-speculative-tokens 3` aborts on the first
speculative step. Depths 0, 1, 2, 4, 5, 6, 7 and 8 all work. The declare-time scratch size is computed for the band's upper bound
(`q_len = 5`) only, while the split law is not monotone in `q_len`, so a 4-row verify step (`q_len = 4`) needs a larger partial buffer
than was reserved. A 14-line fix is below.

## Environment

- radiance commit `451b136` (2026-11-05; `v1.17.0` plus 58 commits)
- AMD Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151, ROCm 10.1.0 (HIP 7.16.26323, clang 23), Linux 7.2.11, `amdgpu.cwsr_enable=1`
- `RADIANCE_KERNELS=libhalo` (see note 02 for why `libhalo` has to be selected on its own)
- Model: `qwen3.6-35b-a3b-fp8.rad`

## Reproduce

```
radiance --model qwen3.6-35b-a3b-fp8.rad --tp 1 --max-num-seqs 2 --max-model-len 32768 \
         --kv-cache-dtype fp8 --num-speculative-tokens 3          # also fails with --kv-cache-dtype bf16
# then any chat completion
```

```
E kernel attn_decode_h256_gqa8_fp8kv (libhalo, splits=0, device domain) refused op 'attn_paged' at q_len=4
  [band q_len<=5] geometry {head_dim=256 gqa=8 causal=1 window=0 n_head=16 block_size=16 max_seqs=2
  max_ctx=32768 q_dtype=bf16 kv_dtype=fp8_e4m3 q_len=5 splits=0}: scratch buffer absent or too small;
  the step is aborted (a kernel that refuses at issue was selected wrongly)
E step 1, issued ahead, failed: scratch buffer absent or too small -- the engine cannot continue
E stopping: the step loop failed with scratch buffer absent or too small
```

The engine then stops serving (later requests return 500). The same failure occurs with `attn_decode_h256_gqa8_bf16kv`.

## Cause

`halo_decode_splits` (`libhalo/halo_attn_decode_h256_gqa6.hip`) targets 192 workgroups while `q_len * gqa <= 32` and 128 above that. For
gqa 8 the discontinuity sits between `q_len = 4` (32 rows, target 192) and `q_len = 5` (40 rows, target 128).

`decode_scratch` sizes the buffer once at declare time from the band's upper bound (`q_len = 5`), sweeping the sequence count but not
`q_len`. The partial buffer scales with `q_len x splits`:

| q_len | rows (q_len x 8) | workgroup target | q_len x target |
|---|---|---|---|
| 1 | 8 | 192 | 192 |
| 2 | 16 | 192 | 384 |
| 3 | 24 | 192 | 576 |
| **4** | 32 | 192 | **768** |
| 5 (declared) | 40 | 128 | 640 |

Only `q_len = 4` needs more than what was reserved (768 > 640). A depth-`d` MTP verify step has `q_len = d + 1`, so depth 3 is the only
depth that hits it, which matches what we observed.

## Fix

Sweep `q_len` at declare time as well as the sequence count. The law is a closed form, so this is cheap:

```diff
+        const int q_hi = r.q_len > 0 ? r.q_len : 1;
         int64_t worst = 0;
-        for (long long n = 1; n <= max_seqs; ++n) {
-            r.num_seqs = (int)n;
-            const int64_t need = (int64_t)decode_scratch_bytes<OPT, WIN>(&r);
-            if (need > worst) worst = need;
+        for (int q = 1; q <= q_hi; ++q) {
+            r.q_len = q;
+            for (long long n = 1; n <= max_seqs; ++n) {
+                r.num_seqs = (int)n;
+                const int64_t need = (int64_t)decode_scratch_bytes<OPT, WIN>(&r);
+                if (need > worst) worst = need;
+            }
         }
```

The function is in the shared decode body, so it applies to every geometry that includes it (gqa 6, 8, 12, ...). Other geometries may have the same
latent issue at their own `q_len * gqa == 32` boundary; we only exercised gqa 8.

## Verification

- Depth 3 serves on both `--kv-cache-dtype fp8` and `bf16`.
- Depths 2, 4, 5, 6, 7 and 8 unchanged (identical decode tok/s before and after, three prompt types).
- Text equivalence with depth 0 at temperature 0 (250 tokens, three prompts). With bf16 KV, depth 3 is byte-identical on two prompts and diverges at
  character 582 of 620 on the third (a near-tie in an invented tag list). With FP8 KV all depths differ from each other by synonym-level wording after
  300-590 characters, including the unpatched depth 4, so this is not introduced by the fix.
