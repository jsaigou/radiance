# Upstream 2.x review: what we took, what we learned, what we did not take

Reviewed 2026-11-17 against `codeberg.org/StillDeadcode/radiance` `main` (`38c182a`; 63 commits after our base `451b136`, including the `v2.0.0`
and `v2.1.0` releases). This is an unofficial fork; see [`../UNOFFICIAL-FORK.md`](../UNOFFICIAL-FORK.md).

## Where we stay, and why

This fork stays on the **1.17.0 base**. Upstream 2.0.0 is a redesign we cannot adopt without rebuilding what we run:

- whole-model formats move from FP8 and MXFP4 to **IQ8R** and **IQR-V4**, and the KV cache from FP8 to **int8** (`kv-cache-dtype: i8`); containers whose weights
  are block FP8 are refused at startup by the name of the first such weight. That includes the stock FP8 containers and our Qwen3.6 W4/W5 containers (block-FP8 trunk);
- the command-line flags are removed in favour of one YAML file (`radiance --config FILE`), which breaks launchers that build an argument vector;
- the plugin ABI is bumped (39 in 2.0.0, 40 in 2.1.0) and out-of-tree plugins must be rebuilt;
- a trial merge conflicts in 11 files, including the core of our work (`rad_block_moe_fp8.h`, `qwen4exp_fp8.cpp`, `checkpoint.cpp`, `ref_shapes.cpp`).

**Compatibility policy.** Our additions are additive. A stock `.rad` built for upstream 1.17.0 (FP8 trunk and experts, the published Qwen3.6 and
Qwen3.8 containers) must keep loading and serving on this fork, and no on-disk layout of an existing dtype is changed; new formats get new names
(`w3nl64a8h`, the `halo.q80.*` / `halo.q6k.*` layouts, and so on). A change that would break a stock container is a regression here. Upstream 2.x containers
(IQ8R, IQR-V4) are not served by this base.

**Neither of our two fixes is upstream.** Upstream's `halo_attn_decode_h256_gqa6.hip` and `halo_attn_prefill_h256_gqa6.hip` are byte-identical to `451b136`,
so notes 01 and 05 still describe live bugs there.

## The twelve fixes we took (`git cherry-pick -x`, applied cleanly)

Each commit carries its upstream hash. What each one is, and what it teaches us:

| Upstream fix | What it is | Lesson for this fork |
|---|---|---|
| sched: an over-cap request would stall every queue behind it | A plugin that declares a fixed per-sequence carrier (`plugin_words`) had no admission check; the build stage refused an over-cap sequence and restaged forever. Now refused at the gate (400). | **A bound the batch builder assumes must be enforced where requests enter.** Our plugins do not declare `plugin_words` (only the upstream `clef` plugins do), so no present exposure; any new plugin that fixes a carrier size needs the check. |
| sched: the preemption gate credits only what the recompute adopts | The gate that readmits a preempted request subtracted the raw attention prefix hit; under checkpoints the linear layers replay from the last checkpoint, so the adopted prefix is smaller. The request was admitted against memory it could not get and preempted again, in a loop. | **Directly relevant.** Qwen3.5/3.6/3.8 are hybrid (linear + attention layers) and we run them with checkpoints at three long sequences. The open item is a live 3 × ~129K stress that exercises preemption. |
| sched: the bidirectional plane trusts its runs no further than it checks them | An overlapping-run layout could bill one token twice and push an index plane out of range. Validated at the consumer. | Vision-only today. The principle (check a documented invariant where it is consumed) applies to our new gathers; see the follow-up audit below. |
| sched: an anchor row with no anchor is row zero, not row minus one | A `-1` "no value" sentinel in an index plane read one element before a device allocation. | **Never use `-1` as a placeholder in a plane a kernel uses as an index.** Audit target: our `embed_lookup_iq4xs` and `embed_lookup_q` gathers. |
| server: a wrong api key closes the connection it guessed on | A 401 kept the connection alive, so one keep-alive could try keys at full speed. | Cheap hardening; our engine is bound to loopback behind a router, so the exposure is low. |
| server: `/props` keeps the model path to the operator; the model card hides the expert-profile path | Two responses told any client where model files live on the host. | Same class: a path on the machine is information. Taken. |
| server: what a client put in a request never reaches the log raw; the worker-start failure logs its reason sanitized | A client-supplied model name or message tail could forge log lines or inject terminal escapes. | Anything client-controlled goes through `log_sanitize` at the print. |
| format: the index does not get to reach outside the model dir | A shard index naming `../../etc/passwd` was opened as the serving user. | A manifest that names files must be screened by name before the first open. Our `ud_map` and `ud_repack` take file lists from the command line, not from inside a model, so no exposure; keep it in mind for any importer that reads names from a file. |
| hf: a lone container does not outrank weights beside it | A decoy `.rad` beside a real safetensors set was served in place of the model named. | Selection by rank can be gamed; resolve ties explicitly. |
| kvstream: a stage run past the host layer says so, once | A silent clamp is memory-safe but gives a fluent wrong answer with no fault. | **Matches our own rule:** refuse by name instead of clamping quietly (`moe: refuse an unserved expert encoding by name`; the `w3nl64a8h` layout refuses a K that is not a whole 128). |

## Six upstream fixes we did not take

These depend on 2.x code (the bounded-read utility, the bailing-layout checks, the new vendored HTTP library, the front-end wake path) and do not apply to
a 1.17 base: `cpp-httplib 0.66.0`, the bounded `util` read and its callers, the two bailing-layout bounds, and `front: a refused request does not wake a model`.
The scheduler's `stuck-exit` watchdog and its load fix are 2.1.0 features and are not in this base.

## Follow-ups that fall out of the review

1. A live stress of three ~129K sequences that forces preemption on a hybrid model with checkpoints (the scenario the preemption-gate fix is about).
2. Audit our new gather kernels (`embed_lookup_iq4xs`, `embed_lookup_q`) for token ids outside the vocabulary, since the gather validates an index plane
   by shape in the style the anchor-row fix describes.
3. Re-test a stock FP8 container after any further merge from upstream, as the compatibility check above requires.
