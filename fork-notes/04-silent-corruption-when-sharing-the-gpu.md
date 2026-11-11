# Silent output corruption with 2+ long sequences while another process uses the GPU

*Not fixed in this unofficial fork. Root cause not found: this is a reproducible characterisation. Not reported to, or confirmed by, the original author.*

## Summary

On Strix Halo (gfx1151, `libhalo`), Radiance returned corrupted output, with no error logged, when **two or more long (~60K-token) sequences were in the engine at once while another process
(llama.cpp) was doing a long prefill on the same GPU**. With one sequence at a time it was clean (0 of 28). The corrupted answers are degenerate text, not wrong guesses: `RC!lylylyly...` repeating
to the token limit, `RC!RCRC...`, `RC神秘`, `RCtekst`, truncations such as `RC-1`. llama.cpp's answers on the same GPU at the same time were always correct.

## Environment

radiance commit `451b136` plus the depth-3 scratch fix (note 01); AMD Ryzen AI MAX+ 395 (gfx1151), ROCm 10.1.0, Linux 7.2.11, `amdgpu.cwsr_enable=1`; `RADIANCE_KERNELS=libhalo`;
`qwen3.6-35b-a3b-fp8.rad`, `--tp 1 --max-model-len 131072 --kv-cache-dtype fp8`, MTP depth 4 unless stated. The second process was upstream llama.cpp (ROCm) serving the same model family.

## Reproduce

Each round: four prompts of about 60,000 tokens (3,500 random 12-word sentences; one sentence in the middle reads "The secret code is RC-<round>-<n>."), each ending with "What is the secret code
mentioned above? Answer with only the code." Temperature 0, thinking off, `max_tokens 20`. Send the four to Radiance at once while a fifth, similar prompt goes to llama.cpp at the same moment. A response is
wrong if it does not contain the code. The wrong answers appeared in about four of every five rounds with four sequences.

## Results (wrong answers; "other process" = a simultaneous ~60K-token llama.cpp prefill)

| Radiance sequences in the engine | other process on GPU | wrong |
|---|---|---|
| 4 (default settings) | yes | 6 / 20 |
| 4, `--busy-prefill-chunk off` | yes | 7 / 16 |
| 4, `--prefill-interleave 0` | yes | 5 / 16 |
| 4, `--no-prefix-cache` | yes | 6 / 16 |
| 4, `--deterministic` | yes | 5 / 16 |
| 4, MTP depth 2 | yes | 4 / 16 |
| 4, speculation off | yes | 5 / 48 |
| 2 (`--max-num-seqs 2`) | yes | 7 / 16 |
| 1 (`--max-num-seqs 1`, four clients queued) | yes | **0 / 16** |
| 1 (a single client) | yes | **0 / 12** |
| 4 | none | 0 / 8 concurrent (and 0 of 8 one at a time) |

Pooled, 27% (45 of 164) of requests with two or more sequences and the other process were corrupted, against 0 of 28 with one sequence (Fisher exact p = 0.0005).

**Not the cause:** MTP speculation raises the rate (33% with it, 5 of 48 = 10% without; p = 0.003) but the defect exists with it off. Prefill chunk capping (disabled in one run, no change), prefill interleaving, the
prefix cache and `--deterministic` do not change the rate. The engine's counters show no preemptions. An earlier partial result (1 of 16 with speculation off) was too optimistic; the larger sample above supersedes it.

**Weakest part of the evidence:** that the other process is required. Without it, four concurrent sequences were clean in the 8 requests tried, which cannot exclude a rate as high as 10%.

## Possibly related

With MTP on and 2 or more concurrent streams, the first draft position's acceptance intermittently collapses (note 03). There the damaged value is checked by verification, so it only costs speed; in the long-prefill
case the damage reaches the output. A cross-sequence hazard that is only exposed when something else perturbs GPU timing would fit both, but this has not been verified.

## Not known

Whether it occurs at shorter contexts (everything above used ~60K-token prompts); with other kinds of GPU load (decode, embeddings, speech); or at all on the upstream project's supported hardware.

## What would help

A debug build that checksums KV blocks and per-sequence scratch before and after each step would show which buffer is damaged. The practical workaround here is `--max-num-seqs 1` while the GPU is shared.
