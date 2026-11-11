# libhalo: the prefill split-KV merge overwrites the decode rows of a mixed step

*Fixed on branch `unofficial-fixes` of this unofficial fork. Not reported to, or confirmed by, the original author. This is the cause of the symptom described in note 04.*

## Summary

On Strix Halo (`libhalo`), with two or more sequences in the engine, a decoding sequence's output was sometimes silently replaced by degenerate text (`MX!lylyly...`,
`MXp`, truncations). No error was logged. The cause is `halo_attn_prefill_merge`: in a **mixed step** (decode rows plus two or more prefill chunks) it merges
every token row of the batch, including the decode rows that the prefill launch never computed, and overwrites their attention output with stale scratch.

## Cause

A mixed step issues attention twice (`arch/common/rad_block_attn_gated_fp8.h`, around lines 811-836): first the decode half, then the prefill half with `cu_seqlens + D`
(D = number of decode rows). When the prefill half uses split-KV (`splits > 1`), `halo_prefill_launch` runs the merge kernel over `halo_q_total(r) * q_heads` rows, which is
all token rows. The prefill attention kernels only write partial results (and `m`, `l`) for rows inside `[cu_q[0], cu_q[num_seqs])`. For the first D rows, the partials hold
whatever the shared scratch held before, so the merge writes garbage into `out` for the decode rows, after the decode half had written them correctly.

It is intermittent because it needs the prefill half to split, which depends on chunk size, context length and whether the scratch fits.

## Fix

The merge kernel takes `cu_q`, `num_seqs` and `q_heads`, and returns early for any row whose token index is outside `[cu_q[0], cu_q[num_seqs])`. One file, 15 changed lines plus the licence change notice.

## How it was found (about 8K-token prompts, 4 concurrent requests, no other process on the GPU, wrong answers per 48 requests)

| Experiment | Wrong |
|---|---|
| unmodified | 5, 5 (and 6 earlier) |
| no step issued ahead (`pipelining_` off) | 3, 3 |
| recorded-pass hazard marker never marks dispatches unordered | 4, 4 |
| full block-table restage every build | 2, 6 |
| fused conv+recurrent disabled in mixed steps | 1, 7 |
| decode rows never share a step with prefill rows | 0, 0 |
| one attention launch in mixed steps | 0, 0 |
| prefill half issued before the decode half | 0, 0 |
| **the fix**, speculation off / depth 4 | **0 of 144 / 0 of 144** (experiment tree), then 0 of 96 / 0 of 96 (fix-only build) |

Step-shape logging showed every failure followed a mixed step with two or more prefill rows.

## Verification

- Fix-only build (no experiment code): 0 wrong in 96 requests with speculation off and 0 in 96 at depth 4 (about 8K tokens, four concurrent, no other process).
- Depths 2 and 3 serve and gave 0 of 48 each.
- Single-sequence greedy output is byte-identical to the build without the fix. Decode tok/s is unchanged.
