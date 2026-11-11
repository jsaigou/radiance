# About this unofficial fork

**This is an unofficial fork of Radiance. The original is at <https://codeberg.org/StillDeadcode/radiance>**
(author: Deadcode / StillDeadcode; project site <https://radiance.stilldeadco.de/>). Use the original for releases,
documentation, support and bug reports. This fork is not affiliated with or endorsed by the original author.

All credit for Radiance belongs to Deadcode and the radiance contributors listed in [`NOTICE`](NOTICE).

## Why this fork exists

We run Radiance on AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S, gfx1151) and found some bugs while evaluating it.
We fixed one and wrote up the others. We did not want to create an account on Codeberg to file them, so instead of keeping
the fixes private we publish them here. Nothing here is meant to compete with or replace the original; if the original author
wants any of it, they are welcome to take it (it is Apache-2.0), and this fork can then be dropped.

## What is different from the original

| Branch | Contents |
|---|---|
| `main` | An exact mirror of the original's `main`, including all release tags. Never committed to. |
| `unofficial-fixes` (default) | The original at commit `451b136` (2026-11-05, which is `v1.17.0` plus 58 later commits) **plus** the changes below. |

Changes on `unofficial-fixes`, relative to the original:

1. **Bug fix:** `libhalo/halo_attn_decode_h256_gqa6.hip` — MTP speculation at depth 3 aborted on Strix Halo with
   "scratch buffer absent or too small". The declare-time scratch sizing now also sweeps `q_len`. Details and verification:
   [`fork-notes/01-libhalo-mtp-depth3-scratch-undersized.md`](fork-notes/01-libhalo-mtp-depth3-scratch-undersized.md).
2. This notice (`README.md` banner, this file, and the `fork-notes/` folder). No other files are changed.

Bugs we found but did **not** fix (descriptions only): [`fork-notes/`](fork-notes/).

## What was tested, and what was not

- Hardware: AMD Ryzen AI MAX+ 395 (Radeon 8060S, gfx1151), 123 GB unified memory, Linux 7.2.11, `amdgpu.cwsr_enable=1`.
- Toolchain: ROCm 10.1.0 (HIP 7.16, clang 23). The older ROCm 7.1 packages did not compile `libr3d`.
- Model: one only, `qwen3.6-35b-a3b-fp8.rad` (Qwen3.6-35B-A3B, FP8, MTP). Run with `RADIANCE_KERNELS=libhalo` (see fork-notes/02).
- The fix was checked at speculation depths 2-8 on that model, with FP8 and bf16 KV caches.
- Not tested: other models, other GPUs, multi-GPU, the `v1.17.0` tag itself, long soaks, image/video/audio models.

No warranty, as in the Apache-2.0 license. For anything beyond the changes above, the original project is the reference.

## Building (Strix Halo / gfx1151)

```
export ROCM_PATH=/path/to/rocm-10.1 PATH=$ROCM_PATH/bin:$PATH LD_LIBRARY_PATH=$ROCM_PATH/lib
cmake -S . -B build -G Ninja -DRAD_GPU_TARGETS=gfx1151 -DCMAKE_PREFIX_PATH=$ROCM_PATH && cmake --build build -j8
RADIANCE_HOME=$PWD/build/radiance_home RADIANCE_KERNELS=libhalo build/bin/radiance --model <file.rad> ...
```

The engine refuses to start unless `amdgpu.cwsr_enable=1`. The original's own README has the full instructions.

## Licensing

Apache-2.0, same as the original. [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) are unchanged. As in the original, two parts carry their
own terms: `core/server/systemone.cpp` includes text under the LFM Open License v1.0 (`core/server/LICENSE.lfm`, which limits commercial
use) and `data/vectors/` carries the Qwen community license. Model weights keep the license of the model they were converted from.
The one modified source file carries a notice of change as Apache-2.0 section 4(b) requires.

## Contact

Issues with this fork's changes: the Issues tab of this repository. Anything about Radiance itself: the original project on Codeberg.
