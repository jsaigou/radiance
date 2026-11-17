# Kernel and format work in this fork

> **Unofficial fork.** This is work by a third party on top of Radiance `v1.17.0` (+58 commits); it has not been reviewed by the original
> author. See [`../UNOFFICIAL-FORK.md`](../UNOFFICIAL-FORK.md). Hardware, toolchain and results are in [`STRIX-HALO.md`](STRIX-HALO.md); how the
> containers are built is in [`RAD-FILES.md`](RAD-FILES.md). Upstream's own kernel notes are [`KERNELS.md`](KERNELS.md).

## Why

On Strix Halo, decode is bandwidth-bound and the model that matters most to us (Qwen3.8-Flash-Next, about 90 GiB) does not fit comfortably
at the precision upstream ships. Unsloth's Dynamic GGUF quantisations show that most of a MoE's bytes (the routed experts) tolerate 2-4
bits while the small, sensitive tensors stay near 8. Radiance had the 4- and 5-bit expert forms (`w4nl64a8h`, `w5nl64a8h`) but not the
formats a mixed-precision file actually uses. This fork adds them, makes the decode kernel faster, and adds the tooling to turn an Unsloth
GGUF into a `.rad`. Everything is behind the existing plugin and layout interfaces; no upstream ABI was changed except where noted.

## Formats added

"Rotated" formats use Radiance's Hadamard-rotated codebooks. "Unrotated" formats are llama.cpp's own block formats, imported as planes so the
decoded values are bit-identical to `dequantize_row_<type>`.

| Format | Used for | Bits/weight | Where |
|---|---|---:|---|
| `w3nl64a8h` (new) | routed experts, rotated | 3.125 | `libquant` codebook `w3nl` (sign plane + 2-bit magnitude plane, table {3.25, 10, 18, 28}, E4M3 scale per 64 of K); `libhalo` grouped MoE GEMM; `arch` declaration |
| `w4nl64a8h`, `w5nl64a8h` (upstream) | routed experts, rotated | 4.125 / 5.125 | decode kernel retuned (below) |
| IQ3_XXS, IQ2_S | routed experts, unrotated | 3.06 / 2.56 | `libhalo/halo_moe_grids.h` + unrotated expert GEMMs |
| IQ4_NL, IQ4_XS | routed experts, unrotated | 4.5 / 4.25 | unrotated expert GEMMs (IQ4_XS needs a third table) |
| Q5_0 | routed experts, unrotated | 5.5 | unrotated expert GEMM |
| Q2_0 (ggml type 42) | routed experts, unrotated | 2.25 | `libquant` encoder and planes, unrotated expert GEMM |
| Q8_0, Q6_K | dense trunk, shared experts, `lm_head` | 8.5 / 6.56 | `halo.q80.tile16x32` and `halo.q6k.*` layouts; matvec and WMMA kernels |
| IQ4_XS | token embedding | 4.25 | `embed_lookup_iq4xs` gather |
| IQ4_NL | per-layer n-gram (PLE) table | 4.5 | `embed_lookup_q` gather |

Every new device kernel has a host reference (`libavx`, `libref`) and a bit-for-bit comparison in `halo_selftest` or a dedicated bench
(`halo_w3_bench`, `halo_moe_unrot_bench`, `q6k_bench`).

**Not implemented:** the K-quants Q4_K and Q5_K. Containers for Qwen3.6 use the nearest fast rotated formats instead of the K-quants.

## Decode kernel

The routed-expert decode GEMM (`libhalo/halo_moe_gemm_q.hip`) ran far below memory bandwidth. Measured on the machine in
[`STRIX-HALO.md`](STRIX-HALO.md), shape gate_up N=1024 K=2048 with 256 experts, a bare-read ceiling of about 218 GB/s:

- It was **not** waiting on DRAM: re-reading the same weight bytes every pass (so they hit cache) changed nothing, and removing dequantisation
  only saved about a quarter of the time. The skeleton (activation staging, barriers, WMMA, fold) dominated, as a serial chain per block.
- It was limited by **LDS occupancy**, not registers: the original 64-column form used 43.5 KB of LDS, three blocks per WGP.
- Deeper prefetch did not help (under 3% either way).

Two changes, both bit-identical to the previous forms (all launch forms are compared in the selftest):

1. **32-column two-wave blocks, one 128-wide K block a pass** for `w3nl64a8h` and `w4nl64a8h` gate/up (14 KB of LDS), and the same
   (tile, run) kernel for the down projection below 200 rows. Microbenchmark, microseconds per launch, E experts active:

   | | E=8 | E=16 | E=32 | E=64 |
   |---|---|---|---|---|
   | gate_up w3 | 71.9 → 53.5 (1.34x) | 121.0 → 92.1 (1.31x) | 222.6 → 171.1 (1.30x) | 431 → 329 (1.31x) |
   | gate_up w4 | 75.8 → 54.1 (1.40x) | 128.0 → 98.4 (1.30x) | 234.8 → 175.8 (1.34x) | 454 → 334 (1.36x) |
   | down w4 | 35.6 → 28.2 (1.26x) | 58.1 → 50.0 (1.16x) | 112.2 → 95.2 (1.18x) | 211 → 186 (1.14x) |

2. **Register-direct dequantisation with a staged activation** (now the default rule): each lane dequantises its own weight row straight into
   the WMMA B fragment, with no LDS weight tile and no per-pass block barrier, and the first two activation rows are staged to f16. Over the
   32-column form: gate_up 1.30x / 1.30x / 1.26x / 1.24x for w3 at T = 8 / 16 / 32 / 64 and 1.17x to 1.00x for w4; down 1.10x to 1.20x. It is
   bounded by `kRegKMax = 2048` K elements of LDS rows; above that the rule falls back to the weight-only register path.

These are expert-kernel figures. End to end the speed-up is bounded by the dense trunk (under 10% of a token at one token, our estimate),
and was not isolated in a served model. The 5-bit and `w4nl32` forms and the down projection at T >= 200 were not retuned.

Correctness evidence at the commit that made this the rule: `halo_w3_bench check` 504 ok / 0 fail across the three formats (>= 99.99%
bit-exact against the host model, the remainder within one bf16 ulp plus f32 noise); `halo_selftest` 2266 checks, 0 fail.

## Prefill fix (affects every model)

`libhalo`'s prefill split-KV merge overwrote decode rows when a step mixed decoding rows with two or more prefill chunks, giving degenerate
answers in roughly 10-25% of concurrent requests. Fixed in `halo_attn_prefill_h256_gqa6.hip`; see
[`../fork-notes/05-libhalo-prefill-merge-corrupts-decode-rows.md`](../fork-notes/05-libhalo-prefill-merge-corrupts-decode-rows.md).

## Architecture plugins

- **`qwen4exp_fp8`** (Qwen3.8-Flash-Next): each expert projection declares its own format; an expert encoding the build cannot serve is refused by
  name; and the plugin keeps a configuration **per trunk layer**, so one container can mix native unrotated layers with rotated ones. Before the
  per-layer change a container mixing the two failed at the first speculative step ("rotated experts need the `hr` pair") because the
  architecture probed the format of layer 0 only.
- **`qwen35moe_gguf`** (Qwen3.6): imports an Unsloth UD GGUF losslessly (planes are byte copies; the container side is exact).

## Tools

| Tool | Purpose |
|---|---|
| `tools/ud_repack/` | CPU-only lossless repack of Unsloth Dynamic GGUF blocks into planes, with a dequant-equality check against llama.cpp's own routines |
| `tools/ud_map/` | Plans a `.rad` from the bf16 weights using an Unsloth GGUF as the *map* of which format each tensor gets; calibrate, build, verify, score |
| `tools/q6k_bench/` | Q6_K bench/check and a GGUF sample extractor |
| `libhalo/halo_w3_bench`, `halo_moe_unrot_bench` | Host-checked speed and correctness benches for the grouped MoE GEMMs |

## Known limits

- A lossy container (anything but the exact-UD map) adds quantisation error relative to Unsloth's file. For Flash-Next the KL against the exact
  container has a long tail (mean 0.21-0.22, median 0.055-0.059 over 1.58 M positions); no paired task loss was seen in English.
- The offline verifier applies one generic RMSE gate that Unsloth's own IQ3_XXS tensors fail (0.283 against bf16), so it reports FAIL on the
  exact container. That is a verifier-policy issue, not a decoder failure; per-format tolerances are not yet written.
- Everything was measured on one gfx1151 machine.
