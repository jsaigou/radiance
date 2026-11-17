---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
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
# Qwen3.8-Flash-Next, Unsloth-Dynamic size policy (84.37 GiB), Radiance `.rad`

Qwen3.8-Flash-Next, a 256K-context mixture-of-experts model, reduced to 84.37 GiB for the Radiance engine on a 128 GB AMD Strix Halo machine, using Unsloth's Dynamic v3.1 map to decide the precision of each tensor.

> **Unofficial.** This is a `.rad` container for the **Radiance** inference engine, built and tested by a third party. It is **not** a transformers, GGUF or llama.cpp
> model and will not load in them. It needs the engine fork described below. The original Radiance project is at <https://codeberg.org/StillDeadcode/radiance>.

## Quick start (read this first)

**This file does not run in llama.cpp, Ollama, vLLM or transformers.** It is a Radiance container: it needs the Radiance engine, in our unofficial fork, built for your GPU.

**You need**
- An AMD Strix Halo machine (Ryzen AI MAX+ 395, Radeon 8060S, gfx1151) with **128 GB** of memory. About 93 GiB is used for the container plus three 131,072-token sequences, and the machine ran at 114 of 123.5 GiB with the system and other services. This is **not** for smaller machines, and it is the only hardware we have tested.
- Linux with ROCm 10.1 (the TheRock build) and the kernel parameter `amdgpu.cwsr_enable=1` (the engine refuses to start without it; changing it needs a reboot).
- `git`, `cmake`, `ninja` and a C++ compiler.

**1. Build the engine (once, roughly 10 to 20 minutes)**
```
git clone --branch strix-halo-r2 https://github.com/jsaigou/radiance.git && cd radiance
export ROCM_PATH=/path/to/rocm-10.1 PATH=$ROCM_PATH/bin:$PATH LD_LIBRARY_PATH=$ROCM_PATH/lib
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1151 -DCMAKE_PREFIX_PATH=$ROCM_PATH
cmake --build build -j8
```

**2. Download the container**
```
hf download jsaigou/Qwen3.8-Flash-Next-UD-Size-radiance flash-size.rad --local-dir .
```

**3. Run it**
```
RADIANCE_HOME=$PWD/build/radiance_home RADIANCE_KERNELS=libhalo \
  build/bin/radiance --model flash-size.rad --served-model-name qwen38-flash-next \
  --host 127.0.0.1 --port 8080 --tp 1 --max-model-len 131072 --max-num-seqs 3 --num-speculative-tokens 2 --kv-cache-dtype k6_v4
```

**4. Talk to it (OpenAI-compatible)**
```
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen38-flash-next", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 200}'
```

The flags set three sequences of up to 131,072 tokens each, speculation depth 2 and a `k6_v4` KV cache. Fewer or shorter sequences use less memory, but we have only measured the settings shown here.

If it will not start: check `cat /sys/module/amdgpu/parameters/cwsr_enable` prints `1`, that the build used `RAD_GPU_TARGETS=gfx1151`, and that you built tag `strix-halo-r2` (or later on the same line), not upstream's 2.x.

## What is in this repository

| File | Size | SHA-256 |
|---|---:|---|
| `flash-size.rad` | 90593767424 bytes | `6b0887a7b25a5c93999669c3ae8d3c7cceecd1754036343ec8f2913826620cb3` |

## How it was made

Source: the official bf16 checkpoint [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision `812d96e10a0dd82b69d749e68693698cf6c36daa`, with the **Unsloth Dynamic v3.1** quantisation
([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), `UD-Q3_K_XL`, downloaded 2026-11-05; the exact revision hash was not recorded) used as the *map* of which tensors take which precision. Credit to Unsloth for the map and
the importance matrix.

This is the per-layer **size policy**, a different quantisation from Unsloth's file and not a lossless copy: IQ3_XXS tensors become `w3nl64a8h`, IQ4_NL/IQ4_XS become `w4nl64a8h`, Q5_0 becomes `w5nl64a8h`, native IQ2_S/Q2_0 are kept on layers 0, 8, 9 and 10
(44 rotated layers and 4 native), the token embedding is IQ4_XS, the MTP layer `w4nl64`, and the vision tower bf16. The result is 84.37 GiB.

Tooling, recipes and the measurements behind this card are public in the fork: [`docs/RAD-FILES.md`](https://github.com/jsaigou/radiance/blob/unofficial-fixes/docs/RAD-FILES.md) and
[`docs/FORK-KERNELS.md`](https://github.com/jsaigou/radiance/blob/unofficial-fixes/docs/FORK-KERNELS.md).

## Quality (measured by us, on our own sets)

Against Unsloth's UD container (the exact-format build of the same map) over 1,575,352 positions at 8K context: mean KL divergence 0.2090 (median 0.0547, p99 2.15), perplexity ratio 0.949, top-1 agreement 83.53%. The mean and the tail are large; the median is close to the Qwen3.6 result.
On 835 English prompts (ARC, GSM8K, MMLU, tool-calling): 784 correct (93.9%) against 777 (93.1%) for the exact-UD container; the difference is within noise (McNemar p = 1.0), with no request errors. On a 60-item set both score 59/60. **No FP8 or bf16 reference was run**
(the official FP8 weights are about 173 GiB), so this card does not claim equivalence to FP8; the defensible claim is parity with the exact-UD container on that set, with an unmeasured tail. Japanese and thinking-mode sets were not run.

## Speed (one AMD Ryzen AI MAX+ 395, gfx1151, 128 GB unified memory)

Direct runs of the engine, three simultaneous sequences of 131,072 tokens each, speculation depth 2, `k6_v4` KV cache: **91.2 tokens a second** in aggregate. Other measured points on the same container: 2 x 131K 66.2 tok/s, 2 x 262K 55.1, 3 x 262K 76.5.
A request through our router answered correctly in 1.4 s for a short prompt and in 20 s for a 10K-token one (including thinking).

## Notes on running

**Memory:** about 93 GiB of unified memory for the container plus three 131K sequences; with the operating system and other services the machine used 114 of 123.5 GiB. It needs a 128 GB class machine and is not meant for smaller ones.

The engine is the unofficial fork **https://github.com/jsaigou/radiance** at tag **`strix-halo-r2`** (commit `a16657e`) or a later commit on the same line. Upstream Radiance 2.x does not serve this container (it replaced the formats this one uses). Tested only on gfx1151 with ROCm 10.1.

## Provenance

- Container built 2026-11-16 with `rad-convert` and `tools/ud_map` (`--policy size`) from the fork. It mixes native and rotated expert layers, which needs the plugin fix in `strix-halo-r1` or later; it was verified to load and answer correctly on `strix-halo-r2` (2026-11-17).
- Base model revision: `812d96e10a0dd82b69d749e68693698cf6c36daa`.
- Format map and importance matrix: Unsloth, downloaded 2026-11-05.

## License and credits

**Qwen Community License 1.0**, as the base model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). The `LICENSE` file in this repository is that license, which **must stay with every copy**.
Among its terms: a commercial product or service with more than 100,000,000 monthly active users or US$20,000,000 monthly revenue must prominently display the model name; and a licensee that runs a Model-as-a-Service or AI Work Assistant business needs a separate license
from Qwen for commercial use (internal use is exempt). Read the license; this card is a summary and not legal advice. Model credit: Qwen. Format map and importance matrix credit: Unsloth.


Quantisation, container building and testing: jsaigou. Engine: Radiance by Deadcode (Apache-2.0); this container needs the unofficial fork linked above.
No warranty; use at your own risk.
