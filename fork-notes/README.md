# Findings from running Radiance on AMD Strix Halo

These notes come from evaluating Radiance 1.17.0 (commit `451b136`) on an AMD Ryzen AI MAX+ 395 (gfx1151) with ROCm 10.1.0. They were
written as bug reports for the original project at <https://codeberg.org/StillDeadcode/radiance>. **We did not file them there**
because we did not want to create a Codeberg account; they are published here instead. Nothing in these notes has been confirmed by the
original author. This is an unofficial fork: see [`../UNOFFICIAL-FORK.md`](../UNOFFICIAL-FORK.md).

| # | Finding | Status in this fork |
|---|---|---|
| 01 | MTP speculation depth 3 aborts with "scratch buffer absent or too small" (`libhalo`, gqa 8) | **fixed** on branch `unofficial-fixes` |
| 02 | `libhalo` is refused beside `libavx` (schema mismatches) and has 34 fewer ops than `libr4d` | not fixed; workaround `RADIANCE_KERNELS=libhalo` |
| 03 | MTP speculation intermittently gave no speedup with 2+ concurrent streams | **not reproducible after the note-05 fix** (about 50 runs); cause never found; kept as a record |
| 04 | Silent output corruption with 2+ concurrent sequences (earlier thought to need another process on the GPU: it does not) | **fixed** by note 05; this note records the symptom and how it was narrowed |
| 05 | `libhalo` prefill split-KV merge overwrites decode rows in a mixed step (the cause of 04) | **fixed** on branch `unofficial-fixes` |

Unrelated observation: with the FP8 KV cache, speculative decoding at temperature 0 is not bit-identical to non-speculative decoding
(synonym-level divergence after a few hundred characters). `GUIDE.md` says greedy text matches; this may just be summation order across
different split counts.
