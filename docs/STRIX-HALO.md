# Running Radiance on AMD Strix Halo (gfx1151)

> **Unofficial fork.** Everything here was measured on one machine by a third party, not by the original author. Numbers are
> evidence from that machine, not guarantees. See [`../UNOFFICIAL-FORK.md`](../UNOFFICIAL-FORK.md) for what this fork is and
> [`FORK-KERNELS.md`](FORK-KERNELS.md) and [`RAD-FILES.md`](RAD-FILES.md) for the kernel and container work.

Strix Halo is an APU: the GPU and CPU share one pool of LPDDR5X. That changes how an inference engine behaves in ways a discrete GPU
does not, and most of what this fork changed follows from it.

## The platform we tested

| | |
|---|---|
| APU | AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151) |
| Memory | 128 GB unified (123.5 GiB visible to Linux) |
| OS / kernel | Linux 7.2.x |
| Toolchain | ROCm 10.1.0 (TheRock build, HIP 7.16, clang 23). The ROCm 7.1 packages did not compile `libr3d`. |
| Required kernel parameter | `amdgpu.cwsr_enable=1` (the engine refuses to start without it) |
| Kernel selection | `RADIANCE_KERNELS=libhalo` (see fork-notes/02 for why) |

What we also ran with (not requirements, recorded so results can be compared): a GTT pool of 120 GiB (`ttm.pages_limit=31457280`,
4 KiB pages), the engine behind a systemd unit with a 300 s stop timeout (a 90 GiB model takes minutes to unload), and Radiance's plugin
directory (`RADIANCE_HOME`) writable only by its owner, because the engine `dlopen`s whatever it finds there.

## Memory on a unified-memory machine

GTT pages are ordinary system RAM. Three consequences we measured:

1. **The GTT counter is not the footprint.** The same 84 GiB container, launched twice under the same settings, put 88 GiB in GTT and
   4 GiB in anonymous process memory on one launch, and 66 GiB in GTT and 31 GiB in anonymous memory on the next. The total (about 93 GiB
   with three 131K-token sequences) was identical. Budget by the **sum** of GTT and resident set, never by `mem_info_gtt_used` alone.
2. **The host, not the GTT ceiling, becomes the limit.** With a container of this size the 120 GiB GTT ceiling is not what runs out;
   the machine's 123.5 GiB is, once the OS and any always-on services are counted. Leave a reserve (we use 8 GiB) and watch `MemAvailable`.
3. **Swap does not help.** GTT pages are not swappable. If swap is zram it is itself RAM; ours held scratch data at 1.05:1 compression.

## Bugs found and fixed

Two real bugs in `libhalo` on this hardware are fixed here; the rest are written up. Details, reproduction and verification are in
[`../fork-notes/`](../fork-notes/):

| Note | Finding | State in this fork |
|---|---|---|
| 01 | MTP speculation depth 3 aborts: "scratch buffer absent or too small" (`halo_attn_decode_h256_gqa6.hip`) | fixed |
| 02 | `libhalo` is refused beside `libavx` (schema mismatches) and has 34 fewer ops than `libr4d` | not fixed; use `RADIANCE_KERNELS=libhalo` |
| 03 | Speculation sometimes gave no speedup with 2+ concurrent streams | not reproducible after note 05 |
| 04 | Silent output corruption with 2+ concurrent sequences | fixed by note 05 |
| 05 | Prefill split-KV merge overwrote decode rows in a mixed step (the cause of 04) | fixed (`halo_attn_prefill_h256_gqa6.hip`) |
| 06 | Unrotated-MoE kernel bench and its logs | record |

## Measured results

All on the machine above, `libhalo`, one process on the GPU unless stated. "Direct" means the engine was run by hand, without a scheduler.

### Qwen3.6-35B-A3B (the model upstream tested)

| Container | Size | Decode (tuned MTP depth) | Notes |
|---|---:|---:|---|
| FP8 `.rad` (reference) | 34.63 GiB | 58.6 tok/s | |
| W4/W5 experts, RTN (`q36-w4w5-rtn.recipe`) | 21.01 GiB | 63.7 tok/s | smaller than Unsloth's UD GGUF (21.65 GiB); matched the FP8 container within noise on an 835-prompt English and a 570-prompt Japanese set |

The W4/W5 container uses the format pair `w4nl64a8h` (gate/up) and `w5nl64a8h` (down) with the dense trunk kept as block FP8. A GPTQ
variant exists (`q36-w4w5-gptq.recipe`). The tuned MTP depth was 2; the FP8 container also does better at 2 than at the default of 4.

### Qwen3.8-27B (dense), FP8 `.rad`

Single stream, MTP depth 7: Radiance 16.4 tok/s against llama.cpp UD-Q8_K_XL 7.2 and UD-Q5_K_M 10.7. An 8K-token prompt prefills in 13.6 s
(592 tok/s) against 21.4 s and 23.8 s. The decode advantage comes from speculation: our estimate from the measured step time is that
without it Radiance would be slower than both (about 6.2 tok/s). Prefill and concurrency are real wins independent of speculation.

### Qwen3.8-Flash-Next (about 90 GiB, 262K-context MoE)

Three containers built from the Unsloth UD v3.1 map; see [`RAD-FILES.md`](RAD-FILES.md) for how they differ. All score 59/60 on a 60-item
eval. On an 835-prompt English set: exact 777 (93.1%), fast 781 (93.5%), size 784 (93.9%); the differences are inside noise
(McNemar p = 1.0). Throughput, direct, three sequences of 131K tokens, speculation 2: **91.2 tok/s** for the size container
(114 GiB used of 123.5 GiB host RAM with the OS and always-on services counted).

## Limits and what was not tested

- One machine, one operator. No second Strix Halo box, no other gfx11 part.
- Quality evidence is English-heavy. Japanese, thinking-mode and an absolute FP8/bf16 baseline for Flash-Next were not run (the official
  FP8 weights are about 173 GiB and were not loaded).
- Speculation is the dominant decode lever for dense models and varies with the prompt; single-stream prose is the low-acceptance case.
- The bug fixes are small, targeted and ours; none has been reviewed by the original author.
