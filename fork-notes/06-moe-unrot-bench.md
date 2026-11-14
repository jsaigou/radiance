# libhalo: the unrotated expert forms against the rotated ones (speed)

Benchmark of the three GGUF-faithful expert forms added at `kernel-full` `2a3200f` — `iq4nl32a16`
(W4 6), `q50a16` (7) and `q20a16` (8) — against the rotated codebook forms already served —
`w4nl64a8h` (2), `w5nl64a8h` (4) and `w3nl64a8h` (5). Kernel `moe_gemm_w4a8` (two parity classes),
same `libhalo.so`, same process, same device state.

Standalone bench (not CMake): `libhalo/halo_moe_unrot_bench.hip`. Run as

```sh
hipcc -x hip --offload-arch=gfx1151 -O2 -std=gnu++17 -ffp-contract=off -I abi -I libhalo \
    -I build/rad_include libhalo/halo_moe_unrot_bench.hip -o halo_moe_unrot_bench -ldl -lamdhip64
flock ~/radiance-test/gpu.lock ./halo_moe_unrot_bench <libhalo.so> layout|check|bench [reps [trials]]
```

Full logs: `bench-logs/moe-unrot-layout.log`, `bench-logs/moe-unrot-check.log`,
`bench-logs/moe-unrot-bench-tp2-tp1.log`.

## Verification

- `layout` 111 checks, 0 failures: the hooks accept whole 128s of K (the GEMM's `group`), refuse the
  rest, keep writes inside the plane, and relayout→unrelayout restores the canonical plane for every
  form. The unrotated scale view (and Q2_0's canonical u2 code plane) are stored as they are.
- `check` 288 checks, 0 failures: all six forms, gate_up and down, T = 2/8/64, over the launcher's
  four forms. Every launch form agrees with the first **bit for bit**, and the unrotated forms match
  a host decode within 0.4% + 1e-4 (the same bound the selftest uses). The rotated forms also match
  a host decode of the same planes; their host reference uses the canonical codebook, the E4M3 scale
  per 64 under `kW4G64Scale`, and the activation scale per 128.
- Built from this worktree on Tohil (`~/radiance-test/src-moe-bench`, never `/opt/radiance`);
  slots a1–a4 were down/idle (`requests_processing` absent) before and after; `flock` held.

## Geometry

512 experts, top_k 10, one expert's stored plane repeated into 512 distinct addresses (real read
traffic; values do not change the kernel's arithmetic). Two shape families:

- **two-rank production slice** (the brief's numbers): gate_up N 640, K 5120; down N 5120, K 1280.
- **single-rank whole expert**: gate_up N 1280, K 2560; down N 2560, K 640 (the whole expert as
  `qwen4exp_fp8` declares it and `halo_selftest`'s tp1 case uses). The brief's 5120/1280 are double
  this family's K; both are reported so the integration can use whichever it meant.

T = M·top_k routed rows. The rule takes the decode grid while `2T ≤ 3·ne` (T ≤ 768), else a prefill
grid. M = 1/4/32 are decode band. `spread` puts 1–2 rows on **every** one of the 512 experts at
T 760 (decode band). M = 128 and 512 are prefill (T 1280/5120). Streaming-read ceiling measured in
the same run: **204.8 GB/s**.

## Two-rank production slice, µs and GB/s

Decode M = 1 (10 experts), and prefill M = 512 (all 512 active):

| form | gate_up M=1 | gate_up M=512 | gate_up GB/s | down M=1 | down M=512 | down GB/s |
|---|---:|---:|---:|---:|---:|---:|
| `iq4nl32a16` W4 6 | 355.4 | 8775.8 | 107.5 | 694.9 | 20513.4 | 92.0 |
| `q50a16` W4 7 | 394.6 | 10159.4 | 113.5 | 731.9 | 23335.0 | 98.9 |
| `q20a16` W4 8 | 689.4 | 17208.4 | 27.4 | 844.2 | 34553.7 | 27.3 |
| `w4nl64a8h` W4 2 | 432.0 | 7447.3 | 116.2 | 503.5 | 17399.9 | 99.4 |
| `w5nl64a8h` W4 4 | 582.2 | 8158.7 | 131.7 | 749.4 | 18424.1 | 116.7 |
| `w3nl64a8h` W4 5 | 499.7 | 7539.1 | 86.9 | 522.0 | 17130.0 | 76.5 |

Against `w4nl64a8h` at prefill: gate_up `iq4nl32a16` +17.8%, `q50a16` +36.4%, `q20a16` +131.1%;
down +17.9%, +34.1%, +98.6%. At decode M = 1 gate_up is *better* for the A16 forms (−17.7% and
−8.7%) but down is worse (+38.0%, +45.4%, +67.7%); the down decode kernel (`moe_down_dec_kernel`)
is a `!kA16` path, so the A16 forms take the (tile, run) grid there.

Spread (T 760, all 512 experts, decode band), gate_up / down µs:

| form | gate_up | GB/s | down | GB/s |
|---|---:|---:|---:|---:|
| `iq4nl32a16` | 7256.1 | 130.1 | 15011.1 | 125.7 |
| `q50a16` | 8369.8 | 137.8 | 17392.3 | 132.6 |
| `q20a16` | 14277.9 | 33.0 | 29170.0 | 32.4 |
| `w4nl64a8h` | 7142.6 | 121.1 | 16437.7 | 105.3 |
| `w5nl64a8h` | 10005.9 | 107.4 | 17767.7 | 121.0 |
| `w3nl64a8h` | 7420.9 | 88.3 | 16279.6 | 80.5 |

The single-rank whole-expert family shows the same ranking (see the log): at M = 512, `iq4nl32a16`
is +24.4% over `w4nl64a8h` on gate_up and +15.3% on down; `q50a16` +41.2% / +30.9%; `q20a16`
+125.5% / +72.8%.

## What limits the A16 forms

**Not DRAM, and not the table decode.** At prefill no form is near the 204.8 GB/s ceiling; the best
(`w5nl64a8h`, `q50a16`) reach 132–139 GB/s, about two-thirds of it. Apparent GB/s tracks **bytes
read per f16 element staged into LDS**, not bytes available: every form expands each code to a
16-bit LDS value, so the 5-bit forms (0.625 B a weight) reach the highest GB/s, then the 4-bit
(0.5 B), then the 3-bit (0.375 B), and the 2-bit (0.25 B) last. The table decode is not the
limiter: `q50a16`'s 32-entry table plus sign plane is the **fastest** A16 form, and the rotated
`w5nl64a8h` (also 32 levels) is the fastest overall; a 16-entry `iq4nl32a16` is not faster than
either.

**The bf16→f16 activation staging costs a real, secondary amount.** `q50a16` and `w5nl64a8h` have
the same 5-bit weight bytes and codebook; the A16 form reads bf16 activations (twice the bytes) and
converts them (`moeq_bf16x2_f16x2`) where the rotated form reads E4M3. At M = 512 gate_up,
8.16 ms → 10.16 ms (+24.5%); the activations are only ~4% of the traffic, so most of that is the
conversion VALU, not the extra bytes. At M = 1 the A16 forms are not slower on gate_up (iq4 355 µs
vs w4 432 µs), so the staging is not a decode penalty.

**Q2_0 is decode-bound.** `q20a16` is the outlier: 27–33 GB/s, 2.0–2.3× slower than `w3nl64a8h`
at prefill although it reads a third fewer bytes, and ~4× the time per weight byte of
`iq4nl32a16`. The cause is `moeq_stage_q20`'s per-code select chain (four branches per code byte);
it is the one A16 form whose time does not fall as its bytes fall. A permute/table form of the same
decode, or a 4-entry LUT, is the obvious experiment if Q2_0 is to be served at speed.

**At M = 1 everything is launch/parallelism bound, not bandwidth bound** (10 experts, a small grid):
the absolute times, not GB/s, are the decode number.

## Commits

- Bench: `libhalo/halo_moe_unrot_bench.hip`, this note and the three logs — see the commit with
  subject `libhalo: unrotated MoE form speed bench` on branch `kernel-moe-bench` (base
  `kernel-full` `2a3200f`).
- No core, kernel, arch or CMake changes.
