---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
tags:
- radiance
- rad
- moe
- unsloth-dynamic
- amd
- strix-halo
- gfx1151
library_name: radiance
---
# Qwen3.6-35B-A3B, 4/5-bit experts (GPTQ), Radiance `.rad`

Qwen3.6-35B-A3B quantised to 21.01 GiB for the Radiance engine on AMD Strix Halo, with **GPTQ** calibration of the experts. This is the variant we serve ourselves.

> **Unofficial.** This is a `.rad` container for the **Radiance** inference engine, built and tested by a third party. It is **not** a transformers, GGUF or llama.cpp
> model and will not load in them. It needs the engine fork described below. The original Radiance project is at <https://codeberg.org/StillDeadcode/radiance>.

## What is in this repository

| File | Size | SHA-256 |
|---|---:|---|
| `qwen3.6-35b-a3b-w4w5-gptq.rad` | 22564196352 bytes | `86a820e904650a552e18cfffbf411792d2d4fcd22876f9b95266e6cb030fd955` |

## How it was made

Source: the official bf16 checkpoint [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) at revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`.
Routed experts are stored at 4 bits (gate and up, `w4nl64a8h`) and 5 bits (down, `w5nl64a8h`), a rotated-codebook format; the dense trunk is stored as block FP8 exactly as in
the published FP8 container; the MTP draft head is derived as in that container. The choice of formats follows the **Unsloth Dynamic** Q4_K_XL map (Q4_K gate/up and Q5_K/Q6_K down
mapped to the nearest formats the engine serves fast), with credit to Unsloth ([unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)). The file is smaller than Unsloth's own UD GGUF (21.01 GiB against 21.65 GiB).

Rounding: GPTQ (Hessian-weighted, with error compensation), calibrated on Project Gutenberg prose plus Radiance source text; the evaluation text was disjoint from the calibration text. Built from `data/recipes/q36-w4w5-gptq.recipe` in the fork. The embedded recipe's calibration path was replaced with a neutral placeholder (`/calibration/hessians`); the weights and every other byte are unchanged, and the scrubbed file produced identical output to the original on our test prompts.

Tooling, recipes and the measurements behind this card are public in the fork: [`docs/RAD-FILES.md`](https://github.com/jsaigou/radiance/blob/unofficial-fixes/docs/RAD-FILES.md) and
[`docs/FORK-KERNELS.md`](https://github.com/jsaigou/radiance/blob/unofficial-fixes/docs/FORK-KERNELS.md).

## Quality (measured by us, on our own sets)

Against the published FP8 container over 1,575,352 text positions (disjoint from the calibration text): mean KL divergence 0.0567 (median 0.0089), perplexity ratio 1.0105, top-1 agreement 91.28%. On task sets it is statistically indistinguishable from FP8: 92.1% against 92.2% on 835 English prompts and 83.7% against 84.0% on 570 Japanese prompts.

## Speed (one AMD Ryzen AI MAX+ 395, gfx1151, 128 GB unified memory)

Measured end to end through a router at speculation depth 2: single-stream decode about 76 (prose) to 78 (code) tokens a second; a 38K-token prompt prefills in 13.5 s (llama.cpp with Unsloth's UD GGUF takes 29.6 s on the same machine); three concurrent streams give 171 tokens a second in aggregate (69 for llama.cpp). Single-stream decode is within 3 to 8% of llama.cpp. The published FP8 container decodes about 9% slower than this format at the same depth in our earlier pilot.

## Running it

Requires the unofficial Radiance fork **https://github.com/jsaigou/radiance**, tag **`strix-halo-r2`** (commit `a16657e`) or a later commit on the same line, built for your GPU (instructions in the fork's
`UNOFFICIAL-FORK.md`). Upstream Radiance 2.x does not serve this container (it replaced the formats this one uses). Tested only on gfx1151 with ROCm 10.1.

```
RADIANCE_HOME=<build>/radiance_home RADIANCE_KERNELS=libhalo \
  radiance --model qwen3.6-35b-a3b-w4w5-gptq.rad --tp 1 --max-model-len 262144 --max-num-seqs 4 --num-speculative-tokens 2 --kv-cache-dtype fp8
```

About 34 GiB of memory for 4 sequences of up to 262,144 tokens with an FP8 KV cache.

## Provenance

- Container built 2026-11-14 with `rad-convert` and GPTQ calibration from the fork; verified to load and answer correctly on `strix-halo-r2` (2026-11-17).
- Base model revision: `345d467b117d1fd5eda9b879a5988de9ae5f47a7`.
- Format map: Unsloth Dynamic Q4_K_XL (credited above).

## License and credits

Apache License 2.0, as the base model [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE). The `LICENSE` file in this repository is that license.
Model credit: Qwen. Format-selection credit: Unsloth (Dynamic quantisation).


Quantisation, container building and testing: jsaigou. Engine: Radiance by Deadcode (Apache-2.0); this container needs the unofficial fork linked above.
No warranty; use at your own risk.
