# MTP speculation intermittently gives no speedup with 2+ concurrent streams

*Not fixed in this unofficial fork. Root cause not found: this is a characterisation. Not reported to, or confirmed by, the original author.*

## Summary

With MTP speculation on and 2-8 concurrent requests, some runs fall into a "slow mode" where the engine emits exactly one token per sequence per
step (no speculative gain) and aggregate throughput roughly halves. The same workload run again is often fast. With a single stream it never
happened (0 of 6 runs); at 2 streams 1 of 8; at 4 streams about 6 of 10. With speculation off throughput is smooth and monotonic. Nothing is logged.

## Environment

Commit `451b136` plus the fix from note 01 (the behaviour is the same at depths 2 and 4 without it), Strix Halo gfx1151, `RADIANCE_KERNELS=libhalo`,
ROCm 10.1.0, `qwen3.6-35b-a3b-fp8.rad`, `--tp 1 --max-num-seqs 8`, nothing else on the GPU.

## Reproduce

Server: `--num-speculative-tokens 4 --kv-cache-dtype fp8 --max-num-seqs 8 --max-model-len 65536`.
Client: N threads, each `POST /v1/chat/completions` with a ~33-token prompt ("Variant i.rep: write a complete Python module implementing a binary
heap priority queue with tests and docstrings."), `max_tokens 300`, `temperature 0`, thinking off. Repeat the batch 10 times and take the difference
of the `/metrics` counters (`radiance:engine_steps_decode_total`, `...decode_tokens_total`, `...draft_tokens_total`, `...draft_accepted_total`) and of
`/stats` `draft_positions` around each batch.

## Observations (4 streams, 10 batches, same server)

| run | wall | decode steps | tokens/step | overall accept | position-1 accept |
|---|---|---|---|---|---|
| fast | 7.1-7.9 s | 116-126 | 7.1-10.4 | 0.41-0.45 | 0.74-0.78 |
| slow | 13.1-15.1 s | about 298 | **4.03** (= 4 rows x 1.0) | 0.05-0.23 | 0.08-0.41 |

- Steps in slow mode are not slower (42-50 ms vs 55-61 ms); they just emit one token per row.
- **Positions 2-4 stay healthy (about 0.6) in slow runs; only position 1 collapses.** Position-1 acceptance lands on discrete levels: about 0.75,
  0.40, 0.20, 0.12, 0.08.
- In some slow runs position-1 acceptance is 0.4 yet tokens/step is still 4.03: accepted drafts do not turn into extra emitted tokens.
- `engine_steps_ahead_total` is about equal to `engine_steps_decode_total` in both modes; held, mixed and unspeculated rows and refills are equal across modes.
- The mode is stable for a whole run, not per step.

## Ruled out (10 runs of 4 streams each)

| change | slow runs |
|---|---|
| baseline: fp8 KV, depth 4 | 7 of 10 |
| bf16 KV | 6 of 10 |
| `--draft-window adaptive` | 7 of 10 (keeps overall acceptance higher, wall time still slow) |
| depth 2 | 6 of 10 |
| `--no-prefix-cache` | 7 of 10 |
| speculation off | 0 of 10 (stable: 40 / 71 / 98 / 113 / 174 tok/s at 1 / 2 / 3 / 4 / 8 streams) |

## Hypotheses (unverified)

- A hazard or ordering issue between the verify step's output (accepted token and last hidden state) and the first MTP draft pass of the next step,
  under the step-issued-ahead pipeline. It would explain why only position 1 is affected, why more concurrent rows make it likelier, and why one stream
  is unaffected. Position-1 acceptance of about 0.2 looks like a draft made without a correct hidden-state input.
- A per-row mapping problem in the first draft pass when more than one row is active.

## Impact

At 4 streams the speculation-on mean (about 105 tok/s over 10 runs) is no better than speculation off (113 tok/s), with 2x run-to-run variance. At 1 stream
speculation is +80 percent (72 vs 40 tok/s). Until fixed, a multi-client deployment is better served with speculation off or capped to low concurrency.

## What would help

A debug counter or log line per step for the first draft pass's input row and hidden-state source would make this directly observable.
