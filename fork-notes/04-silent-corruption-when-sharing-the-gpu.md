# Silent output corruption with 2+ concurrent sequences (cause and fix: note 05)

*Fixed by note 05 on branch `unofficial-fixes` of this unofficial fork. Not reported to, or confirmed by, the original author. This note keeps the symptom description and the corrections made on the way.*

## Summary

On Strix Halo (gfx1151, `libhalo`), Radiance returned corrupted output, with no error logged, when two or more sequences were in the engine at once. The answers were degenerate
text, not wrong guesses: `RC!lylyly...` repeating to the token limit, `RC!RCRC...`, `RC神秘`, truncations such as `RC-1`. With one sequence at a time it was clean (0 of 28).
llama.cpp's answers on the same GPU were always correct. **Root cause and fix: [note 05](05-libhalo-prefill-merge-corrupts-decode-rows.md).**

## Environment

radiance commit `451b136` plus the depth-3 scratch fix (note 01); AMD Ryzen AI MAX+ 395 (gfx1151), ROCm 10.1.0, Linux 7.2.11, `amdgpu.cwsr_enable=1`; `RADIANCE_KERNELS=libhalo`;
`qwen3.6-35b-a3b-fp8.rad`, `--tp 1 --kv-cache-dtype fp8 --max-num-seqs 4`, MTP depth 4 unless stated.

## Reproduce

Each round: four prompts of N random 12-word sentences (470 is about 8K tokens, 1200 about 20K, 3500 about 60K), one of which reads "The secret code is MX-<round>-<n>.", each ending
"What is the secret code mentioned above? Answer with only the code." Temperature 0, thinking off, `max_tokens 20`. Send the four at once. A response is wrong if it lacks the code.

## Rates before the fix (wrong answers)

| Prompt size | no other process | llama.cpp prefill | llama.cpp decode | embedding load | TTS |
|---|---|---|---|---|---|
| ~8K tokens (48 each) | 6 | 3 | 3 | 3 | 3 |
| ~20K tokens (32 each) | 8 | 7 | 4 | 6 | 8 |
| ~60K tokens, llama.cpp prefill (12) | | 3 | | | |

With `--max-num-seqs 1` it was 0 of 28. Speculation made it likelier (33% with, 10% without) but did not cause it.

## Corrections to earlier versions of this note

- An earlier version concluded that another process using the GPU was required, and recommended `--max-num-seqs 1` "while the GPU is shared". Both were wrong: the first runs
  without another process were too few (8 requests). The matrix above shows the same rate with nothing else on the GPU. It also grows with context length.
- An earlier partial result (1 of 16 with speculation off) was too optimistic.

## Not the cause (each tested at about 8K tokens)

Step issued ahead, the recorded-pass hazard marker, block-table delta staging, the fused conv+recurrent kernel, the prefix cache, prefill interleaving and chunk capping, `--deterministic`.
