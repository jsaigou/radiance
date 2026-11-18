# Haru H6: the production host source

Status: design note for STEP 1. **No code is written in this step.** This note specifies the
production implementation of `gen::DuplexHostSource` for the Haru duplex tenant, the host noise
recipe, where the sampling parameters come from, how the reply text and audio reach the wire, how
the source is attached for `haru` only, and the test plan. Every claim about the reference cites
the read-only copy under `.haru-ref/`; every claim about this tree cites a file and line.

Trees and contracts:

* reference: `.haru-ref/{session,sampling,config,vibevoice_codec}.py`, `.haru-ref/real-config.json`;
* ABI 39 seam: `core/gen/duplex_driver.h`, `core/gen/duplex_driver.cpp`;
* host sampler: `core/gen/duplex_sampler.{h,cpp}`;
* wire/server: `core/server/duplex.cpp`, `core/server/duplex_engine.{h,cpp}`,
  `docs/DUPLEX-ENGINE.md`;
* ABI: `abi/rad_builder.h:1619-1678`, `abi/rad_runtime.h:651-705`;
* plugin: `arch/haru/haru.cpp`, `arch/haru_common/haru_codec.h`.

---

## 0. Summary

The engine can already run a Haru lane with a teacher feed: the plugin reads `text_in`,
`codec_noise` and `head_noise` from ABI 39 buffers, writes `logits_out`/`pad_eop_out`, and
`tests/haru_assembly_test.cpp` replays the goldens through `gen::DuplexDriver`. What is missing is
the production occupant of that seam:

1. **No source is attached.** With `host_sampling = 1` and `host_source_ == nullptr`, the driver
   zeroes the two noise buffers (`core/gen/duplex_driver.cpp:844-860`, `:966-984`) and never
   writes `text_in` (`core/gen/duplex_driver.cpp:1101-1122`). The plugin then embeds token 0 on
   every step (`arch/haru/haru.cpp:349`) and samples from wrong audio. A real session cannot run.
2. **Text is decoded one token at a time by the server.** `EngineDuplexLane::map` calls the
   tokenizer for each reply token (`core/server/duplex.cpp:177-182`) instead of a per-lane
   `gen::StreamDetokenizer`, so PAD/EOP runs and partial UTF-8 are wrong
   (`.haru-ref/sampling.py:103-128`; `docs/HARU-H5-DESIGN.md:431-437`).
3. **Reply audio is never clamped.** The plugin's codec decode writes `audio_out` directly
   (`arch/haru/haru.cpp:371-374`) and there is no constant-bound clamp op; the reference clamps
   each frame to `[-1, 1]` (`.haru-ref/session.py:163`). The host must clamp at the one conversion
   from engine reply to wire reply: `EngineDuplexLane::map` (`core/server/duplex.cpp:167-184`).

The plan: one `HostSampleSource` per program, attached in `Engine::run_server` only when the
tenant declares `host_sampling = 1`; per-lane `LaneSampler`, noise PRNG and `StreamDetokenizer`;
noise constants declared by the plugin; text and clamp applied in the server adapter.

---

## 1. What exists, and what the source must do

### 1.1 The seam today

`DuplexDriver` calls the source in exactly three places:

* `noise()` before a pass, from `build_prompt` (`core/gen/duplex_driver.cpp:851-860`) and
  `build_step` (`:974-983`), while `next()` holds the driver mutex
  (`core/gen/duplex_driver.cpp:1030-1067`);
* `tokens()` after the pass's readbacks, from `done()`, also under the mutex
  (`core/gen/duplex_driver.cpp:1101-1122`).

`DuplexHostPass` gives `kind`, `n_lane`, `lane[]`, `frame[]`, `n_tok`, `latent_dim`, `n_vocab`,
`pad_id`/`eop_id`/`eos_id`, and the readback rows (`core/gen/duplex_driver.h:107-118`;
`core/gen/duplex_driver.cpp:751-767`). It does **not** give the lane seed, the PROMPT row's
`ref_frame`, or a text channel.

Two indexing conventions matter:

* **Noise is entry-indexed.** The plugin reads `codec_noise + i * latent_dim` and
  `head_noise + i * latent_dim` with `lane = lane_h[i]` (`arch/haru/haru.cpp:336-347`,
  `:358-370`), so the source writes by pass entry `i`, not lane.
* **Logits and PAD/EOP are lane-indexed.** The readbacks target
  `hptr(R_LOG) + lane*n_vocab` and `hptr(R_PE) + lane*2`
  (`core/gen/duplex_driver.cpp:892-899`, `:1013-1023`). The current header comment
  "`logits(): [n_lane, n_vocab]`" (`core/gen/duplex_driver.h:116-117`) is wrong for a multi-lane
  pass; the production source must read `pass.logits + lane*n_vocab` using `pass.lane[i]`. The
  teacher source hides this because its lane is re-used as lane 0
  (`tests/haru_assembly_test.cpp:83-130`).

### 1.2 What the source must produce

| pass | `noise()` | `tokens()` |
|---|---|---|
| PROMPT | reference-frame codec noise for every row with `ref_frame[t] >= 0`; frame 0 head noise; card rows zero | sample token 0, write it to `text_in`; advance DRY and the detokenizer; discard frame 0's text |
| STEP | user-frame codec noise and head noise for each entry `i`, lane `lane[i]` | sample token `s+1`; write it to `text_in`; return the token and keep the detokenizer's delta for the reply |

Frame 0 is computed but dropped (`docs/HARU-H5-DESIGN.md:86-97`, `:439-440`); the first wire reply
is model frame 1. The first step's input token is token 0 (`text_in` written by the PROMPT
sampling).

---

## 2. (a) `HostSampleSource : DuplexHostSource`

Suggested files: `core/server/duplex_host.h`, `core/server/duplex_host.cpp`. They may include
`gen/duplex_driver.h`, `gen/duplex_sampler.h` and `text/tokenizer.h`; they do not touch a device.

### 2.1 Shape

```cpp
namespace rad { namespace server {

class HostSampleSource final : public gen::DuplexHostSource {
public:
    struct Options {
        gen::SamplerConfig sampler;          /* defaults from .haru-ref, server flags override */
        double  codec_noise_scale = 0.0;     /* declared by the plugin: latent_std / 0.8 */
        int32_t pad_id = -1, eop_id = -1, eos_id = -1;
        int64_t n_vocab = 0, latent_dim = 0, max_lanes = 0;
        /* HF-equivalent decode: .haru-ref/sampling.py:124 uses skip_special_tokens=False. */
        std::function<std::string(const std::vector<int>&)> decode;
        /* Test seam only: null uses the internal splitmix64/Box-Muller stream. A test supplies
         * the golden normal draws in the same order to check the combination, without making
         * the production path depend on the injected values. */
        std::function<double()> normal;
    };
    explicit HostSampleSource(Options o);

    /* new seam notifications, section 2.2 */
    void lane_opened(int32_t lane, uint64_t seed) override;
    void lane_closed(int32_t lane) override;

    int noise(const gen::DuplexHostPass& p, float* codec_noise, float* head_noise,
              std::string* why) override;
    int tokens(const gen::DuplexHostPass& p, int32_t* out, std::string* why) override;
    int reply_text(int32_t lane, std::string* out, std::string* why) override;

private:
    struct Lane {
        gen::LaneSampler            sampler;   /* DRY + forced-text + own splitmix64 */
        NoisePrng                   noise;     /* own splitmix64 + normal method */
        gen::StreamDetokenizer      detok;     /* PAD/EOP gap, partial UTF-8 */
        std::string                 last_text; /* delta from the last STEP sample */
        int64_t                     next_frame = 0;
    };
    Options                          o_;
    std::vector<std::unique_ptr<Lane>> lanes_;  /* sized o_.max_lanes at construction */
    std::vector<float>               logits_;   /* n_vocab scratch, reused under mu_ */
    std::mutex                       mu_;
};

}}  /* namespace rad::server */
```

`LaneSampler` and `StreamDetokenizer` are exactly the S2 implementation
(`core/gen/duplex_sampler.h:73-133`); this class is wiring, not a second sampler. `LaneSampler`
already performs DRY, PAD/EOP replacement, `eop_bias`, the forced-text ban, temperature/top-k/top-p
and its own RNG (`core/gen/duplex_sampler.cpp:113-225`). `StreamDetokenizer` already implements the
silent-run space and the U+FFFD hold (`core/gen/duplex_sampler.cpp:209-224`;
`.haru-ref/sampling.py:116-128`).

### 2.2 Learning the seed, the reference rows and the text stream

`lane_opened`, `lane_closed`, `reply_text` and `DuplexHostPass::ref_frame` must be added to the
seam. ABI 39 is unreleased; this is an internal C++ seam, not a struct in the ABI.

```cpp
/* core/gen/duplex_driver.h */
struct DuplexHostPass {
    /* ... existing fields ... */
    const int32_t* ref_frame = nullptr;  /* PROMPT: n_tok entries, null otherwise */
};

class DuplexHostSource {
public:
    /* One conversation's lifetime. Called by open() after the lane index is assigned and before
     * any PROMPT pass can be staged; lane_closed() when the driver releases or retires the lane. */
    virtual void lane_opened(int32_t lane, uint64_t seed) {}
    virtual void lane_closed(int32_t lane) {}

    /* The STEP reply's detokenized delta. RAD_OK means this lane's text is a stream delta;
     * RAD_E_UNSUPPORTED (the default) leaves the reply on the plugin-sampling path. */
    virtual int reply_text(int32_t lane, std::string* out, std::string* why) {
        (void)lane; (void)out; (void)why;
        return RAD_E_UNSUPPORTED;
    }
    /* ... existing noise()/tokens() ... */
};

struct DuplexReply {
    /* ... existing fields ... */
    std::string text;       /* the host stream's delta, when host_text */
    bool        host_text = false;  /* the text came from StreamDetokenizer, not text_out */
};
```

Driver wiring:

* **seed.** `DuplexRequest::seed` (`core/gen/duplex_driver.h:98-104`) is copied to `Job::seed`
  when the lane is admitted (`core/gen/duplex_driver.cpp:506`). Call
  `host_source_->lane_opened(lane, j->seed)` after `t->set_lane(j->lane)`
  (`core/gen/duplex_driver.cpp:500-507`). The notification is the seed's only channel; a per-entry
  field on the pass would be ambiguous on a multi-lane STEP and is not needed.
* **reference rows.** Set `hp.ref_frame = h_ref_frame_.data()` on PROMPT. The driver already has
  the vector for the plugin (`core/gen/duplex_driver.cpp:888-889`); the server fills one entry per
  prompt row (`core/server/duplex.cpp:273-278`).
* **close.** Call `host_source_->lane_closed(lane)` in `release_locked` before the lane index is
  returned to `free_lanes_` (`core/gen/duplex_driver.cpp:373-385`). This runs under the driver
  mutex from `close`, `shutdown` and `retire_locked`; a subsequent `open` cannot interleave.
* **reply text.** In `done()`'s STEP branch, after `tokens()`, call
  `reply_text(lane, &s.r.text, ...)` for each entry with a ticket; set
  `s.r.host_text = true` only when it returns `RAD_OK` (`core/gen/duplex_driver.cpp:1153-1167`).
  The default `RAD_E_UNSUPPORTED` leaves `host_text` false, so the teacher source and every
  plugin-sampling tenant keep the old per-token path. No `reply_text` on PROMPT.
* **teacher source.** Give the three new virtuals empty defaults so `tests/haru_assembly_test.cpp`
  compiles unchanged; fix its lane indexing in the same change (section 1.1).

Why explicit open/close rather than inferring from PROMPT: lane indices are reused
(`core/gen/duplex_driver.cpp:342-343`, `:380-382`). A reopened lane must start with a fresh DRY
history, RNG and detokenizer, and a closed lane must release its text. PROMPT is the model's reset
point (`docs/HARU-H5-DESIGN.md:525-531`) but a map keyed only by lane index cannot tell a new
conversation from a re-PROMPT, nor can it know a seed that has not been staged yet. If the
notifications are rejected, the fallback is to reset on every PROMPT and read the seed from the
pass, but then the map lives for `max_lanes` entries and a closed lane's state lingers.

### 2.3 Thread safety

* The driver calls `noise()`, `tokens()` and `reply_text()` on the engine thread while holding its
  `mu_` (`core/gen/duplex_driver.cpp:1032`, `:1097`), and `lane_opened`/`lane_closed` on the server
  session thread inside `open()`/`close()` while holding the same `mu_`
  (`core/gen/duplex_driver.cpp:480-510`, `:373-385`). In the current code every seam call is
  therefore serialized by the driver lock.
* The interface does not promise that, so `HostSampleSource` keeps its own `mu_` around all lane
  state. It must never call back into `DuplexDriver`, take its lock, or block on I/O; a source lock
  order is driver `mu_` -> source `mu_` only.
* `LaneSampler` and `StreamDetokenizer` are not thread-safe (`core/gen/duplex_sampler.h:126-127`).
  One instance per lane, touched only under the source lock.
* `noise()` and `tokens()` are O(latent_dim), O(n_tok) and O(n_vocab); they run inside the pass
  staging and `done()` paths. The logits copy (section 5.1) is the only sizeable work; the driver
  already copied the same row back from the card.
* The decode callback runs on the engine thread under the lock. Use a dedicated
  `core::Tokenizer` instance owned by the source, or a callback onto a const decode; do not share
  the server's `TokenizerBridge` if it is also used concurrently by HTTP. `Tokenizer::decode` is
  const (`core/text/tokenizer.cpp:1656-1672`), but the source should not rely on an unstated
  thread-safety claim for a shared instance.
* Do not log per frame; a log line per 133 ms per lane is a throughput and privacy hazard.

---

## 3. (b) Noise: recipe, declaration, PRNG order

### 3.1 The reference recipe

`VibeVoiceCodec.sample` draws one scale **per batch row** and a `[B, T, D]` raw noise:

```python
scale = torch.randn(mean.shape[0], 1, 1, generator=generator)   # .haru-ref/vibevoice_codec.py:198
noise = torch.randn(mean.shape, generator=generator)            # .haru-ref/vibevoice_codec.py:199
return mean + scale * (latent_std / 0.8) * noise                # .haru-ref/vibevoice_codec.py:200
```

* `latent_std` is `0.5` in the config (`.haru-ref/config.py:26`, `.haru-ref/real-config.json:115`);
  the `0.8` is a fixed reference constant. The combined factor is
  `codec_noise_scale = latent_std / 0.8 = 0.625`.
* The user frame is one row (`[1, 1, D]`), so each step has one scale and `D` raw normals
  (`.haru-ref/session.py:140`).
* The whole reference clip is one batch row: the golden schema has
  `prompt.ref_noise_scale [1]`, `prompt.ref_noise_raw [14, 64]`
  (`tests/data/haru/README.md:175-177`). Therefore the reference uses **one** scale for all 14
  frames and `F*D` raw normals in frame-then-dimension order, not one scale per frame. The server
  stages one clip (`ref_cu = {0, ref_audio.size()}`, `core/server/duplex.cpp:248-266`); if several
  clips ever become several batch rows, each row would get its own scale, but that is not the
  current request shape.
* The head starts from a raw normal of width `latent_dim` (`.haru-ref/session.py:159-161`;
  `tests/data/haru/README.md:201`).

The host writes the **combined** term `scale * codec_noise_scale * raw` into `codec_noise`, because
the plugin's `codec_sample` adds that buffer directly
(`arch/haru_common/haru_codec.h:140-148`; `arch/haru/haru.cpp:284-292`, `:344-347`). It writes raw
normals into `head_noise` (`.haru-ref/session.py:159-162`; `arch/haru/haru.cpp:367-369`).

### 3.2 What the plugin declares

The plugin knows both the config value and the fixed `0.8`; the host knows neither. Append one
field to `RadDuplexDecl`, after `text_in` (`abi/rad_builder.h:1673-1676`):

```c
    /* ---- ABI 39: the codec-noise coefficient the host multiplies its normal draws by.
     * .haru-ref/vibevoice_codec.py:196-200: combined = scale * (latent_std / 0.8) * raw.
     * The host does not read the checkpoint; the plugin declares latent_std / 0.8 here.
     * 0 when host_sampling is 0. */
    double      codec_noise_scale;
```

The plugin already has the value: `haru::codec_geometry_from_meta` fills `cg.latent_std`
(`arch/haru_common/haru_codec.cpp:356`, default `arch/haru_common/haru_codec.h:48`), and
`haru::codec_noise_factor` documents the exact term (`arch/haru_common/haru_codec.h:145-148`). In
`arch/haru/haru.cpp:231-238` set `d.codec_noise_scale = cg.latent_std / 0.8;` beside the existing
`host_sampling` fields.

Validation and layout:

* `Builder::declare_duplex_tenant` adds to the host-sampling block
  (`core/build/rad_builder.cpp:2196-2216`): `host_sampling = 1` requires
  `codec_noise_scale > 0` and finite; a non-finite or non-positive value is `RAD_E_INVAL`.
* `DuplexDriver::init` may repeat the check before attaching a source
  (`core/gen/duplex_driver.cpp:281-290`); the engine also refuses to attach a source when the
  value is not positive.
* Append the field to `abi/rad_abi_layout_rows.h` and regenerate `tests/abi_layout_table.h` on the
  Linux build; `RAD_ABI_VERSION` is already 39 (`abi/rad_abi.h:22`), and the layout fingerprint
  changes with the field (`docs/HARU-H5-DESIGN.md:465-476`).
* Declaring the combined coefficient, not `latent_std`, keeps the reference's fixed `0.8` out of
  the host and leaves the plugin the only reader of the checkpoint.

### 3.3 PRNG and stream order

Each lane gets its own noise PRNG so the draw sequence is independent of which other lanes share a
pass and of the pass's entry order. This also keeps the teacher replay order clear.

* State: 64-bit `splitmix64`, initialized at `lane_opened` to
  `seed ^ 0xD1B54A32D192ED03ULL` (a domain separation from the text sampler, which starts
  `LaneSampler::rng_state_` at `seed`; `core/gen/duplex_sampler.cpp:44-57`). The two streams are
  deliberately independent: the reference interleaves token, head and codec draws from one torch
  generator, which the host cannot reproduce, and separating them keeps token choices from
  perturbing audio noise. Production is deterministic, not torch-bit-identical.
* Uniform: the same splitmix64 step as S2: `z = (state += 0x9E3779B97F4A7C15); ...;
  u = (z >> 11) * 2^-53` (`core/gen/duplex_sampler.cpp:50-57`). This is documented so a replay
  script can reproduce it.
* Normal: Box-Muller, two uniforms per pair. `u1` is clamped up to `2^-53`; `r = sqrt(-2 ln u1)`,
  `a = 2*pi*u2`; the pair is `(r cos a, r sin a)`. Draw in pairs and emit in order; if an odd count
  is requested, the unused sine is discarded (the state still advanced). `latent_dim = 64` is
  even, so this only matters for a future odd head width.
* Every normal, including a scale, consumes the pair; the scale is one normal draw.

Draw order per lane, consumed sequentially across the lane's life:

1. **PROMPT.** For the reference clip: one scale normal, then `F * latent_dim` raw normals in
   frame-major, then dimension order. Write `scale * codec_noise_scale * raw` at
   `codec_noise[t * latent_dim + d]` for the row `t` whose `ref_frame[t] == f`; the server appends
   reference rows in frame order (`core/server/duplex.cpp:273-278`), but the source orders by
   `ref_frame`, not by row position. Card rows (`ref_frame[t] == -1`) stay zero.
2. **PROMPT, after the reference noise.** `latent_dim` raw normals for frame 0's head, into
   `head_noise[0 * latent_dim ...]` (one lane).
3. **Each accepted STEP**, for the lane: one scale normal and `latent_dim` raw normals for the
   user frame; place the combined term at entry `i` (`codec_noise[i * latent_dim ...]`), where
   `p.lane[i]` is this lane. Then `latent_dim` raw normals for the head, at
   `head_noise[i * latent_dim ...]`.
4. **Flush** consumes no noise; the driver stages no pass (`core/gen/duplex_driver.cpp:559-593`).

The order is per lane, not global: if lane 1 steps five times before lane 0 steps once, lane 0's
first draw is still its reference/head/first-user sequence. The source may assert
`p.frame[i] == lane.next_frame` and increment; it must not index the PRNG by frame number.

Edge cases:

* A reference frame appearing on two prompt rows is a server/prompt bug; the source refuses it
  with a reason instead of silently writing two different noise rows for one frame.
* A PROMPT with no references: the reference noise loop draws nothing; only frame 0's head noise
  is drawn.
* The driver zeroes both noise buffers before calling `noise()` (`core/gen/duplex_driver.cpp:847-850`,
  `:969-972`), so a source that writes only reference rows still leaves card rows zero. The source
  should still write every row it is responsible for, to make its own behaviour explicit.

Reproducibility is within one binary and machine: splitmix64 is exact, but `log`/`cos`/`sin`/`sqrt`
come from the platform libm. If cross-build replay is required, replace Box-Muller with a fixed
integer-to-normal table or a ziggurat with integer-only state; that is a source implementation
choice, not a seam change.

---

## 4. (c) Sampling parameters

### 4.1 What the wire and Forge send today

`docs/DUPLEX-ENGINE.md:48-65` defines `start` as `type`, `contract`, `session_id`, `tenant`,
`text`, `max_minutes`, `card` and `reference`. There is no temperature, top-k, top-p, DRY,
`eop_bias` or `speak_first` field. The in-repo client sends exactly those keys
(`tests/haru_harness.h:417-430`). Forge is external and follows the same contract; I cannot inspect
its source here, so I record the contract's field set, not a guess about Forge's internals. The
only per-session inputs that affect sampling are `session_id` (the seed) and `text` (whether text
is emitted); `card` is prompt text, never parsed for sampling.

The contract allows additive fields and new `flags` bits (`docs/DUPLEX-ENGINE.md:120-123`). An
optional `start.sampling` object could be added later without a contract bump, but it is **not**
sent today; the production defaults must not depend on it.

### 4.2 Where the values come from

Add duplex sampling fields to `Config` next to the existing duplex options
(`core/rad_core.h:1135-1154`), with parsing in `core/config.cpp:1063-1075` and help rows in
`:1527-1533`. `Engine::run_server` constructs the source from `cfg_` (section 6). Proposed flags
and defaults:

| flag | default | reference |
|---|---|---|
| `--duplex-temperature` | `0.7` | `.haru-ref/session.py:35` |
| `--duplex-top-k` | `25` | `.haru-ref/session.py:36` |
| `--duplex-top-p` | `1.0` | `.haru-ref/session.py:37` |
| `--duplex-eop-bias` | `0.0` | `.haru-ref/session.py:39` |
| `--duplex-speak-first` | off | `.haru-ref/session.py:40` |
| `--duplex-dry` | on | `.haru-ref/session.py:38` (`DryConfig` default) |
| `--duplex-dry-multiplier` | `0.8` | `.haru-ref/sampling.py:14` |
| `--duplex-dry-base` | `1.75` | `.haru-ref/sampling.py:15` |
| `--duplex-dry-allowed-length` | `1` | `.haru-ref/sampling.py:16` |
| `--duplex-dry-lookback` | `1024` | `.haru-ref/sampling.py:17` |
| `--duplex-dry-max-match` | `32` | `.haru-ref/sampling.py:18` |
| `--duplex-dry-max-penalty` | `32.0` | `.haru-ref/sampling.py:19` |

`gen::SamplerConfig` already carries exactly these values with the reference defaults
(`core/gen/duplex_sampler.h:73-80`); the source copies them into each lane's `LaneSampler`.

* `guidance_scale = 1.5` (`.haru-ref/session.py:34`) is **not** a host sampler field: the plugin
  declares it and fills its guidance buffer (`arch/haru/haru.cpp:197-198`).
* `seed` is `std::hash<std::string>{}(start.session_id)` at session open
  (`core/server/duplex.cpp:291`) and is handed to the driver in `DuplexRequest::seed`. It seeds
  both the text sampler and the noise PRNG (section 3.3). It is not a server flag. Note that
  `std::hash` is implementation-defined; the same server binary replays, a different standard
  library need not (section 8).
* `text`: the tenant capability is `RadDuplexDecl::text` (`abi/rad_builder.h:1647-1648`) and the
  session request is `start.text`; `EngineDuplexTenant::open` already combines them
  (`core/server/duplex.cpp:492-530`). **Host sampling runs regardless**, because the plugin needs
  `text_in` as its next input (`arch/haru/haru.cpp:349-353`); `text` only gates whether the
  detokenized delta is put on the wire. `EngineDuplexLane` gains a `want_text` flag from
  `start.text` (section 5.3).
* `speak_first`: the reference forces one non-silent token on frame 0 (`.haru-ref/session.py:112`,
  `:154-156`). Haru's deployment is false (`docs/HARU-H5-DESIGN.md:425-426`). With the one-ahead
  rule frame 0's text is not on the wire (`docs/HARU-H5-DESIGN.md:439-440`), so enabling
  `speak_first` would sample the forced text into token 0 and show it nowhere. Keep the flag off;
  a real opening turn needs the first reply to carry frame 0's delta or a separate frame-0 message.
  The design preserves `SamplerConfig::force_text_frames` so the default path stays identical.
* `max_frames = 27_000` (`.haru-ref/session.py:41`) is a deployment bound, not a sampler input: the
  driver bounds a lane at `--duplex-max-ctx` (`core/rad_core.h:1141-1143`,
  `core/engine.cpp:2276-2288`) and the server enforces `start.max_minutes`
  (`core/server/duplex.cpp:735-746`).

---

## 5. (d) Detokenizer, wire text/flags, clamp

### 5.1 Sampling one row

In `tokens()`:

1. For each entry `i`, take `lane = p.lane[i]` and copy the f32 row
   `p.logits + lane * p.n_vocab` into the source's `logits_` scratch. The driver's row is
   lane-indexed (section 1.1). Copying is necessary because `p.logits` is `const` and
   `LaneSampler::sample` applies DRY and the PAD/EOP replacement in place
   (`core/gen/duplex_sampler.h:89-96`).
2. `pad_eop = p.pad_eop + lane * 2` (`core/gen/duplex_driver.cpp:1013-1023`).
3. `token = lane.sampler.sample(logits_.data(), p.n_vocab, pad_eop)`; the sampler's order is DRY,
   PAD/EOP replacement, `eop_bias`, forced-text ban, sample, history update
   (`core/gen/duplex_sampler.h:89-92`, `:1-27`).
4. `out[i] = token`. On PROMPT, also `lane.detok.push(token)` and discard the returned text; the
   token is written to `text_in` by the driver, and frame 0 is dropped.
5. On STEP, `lane.last_text = lane.detok.push(token)`. Do not return it from `tokens()`; the driver
   asks for it with `reply_text(lane)` (section 2.2) so the seam's token array stays the token
   feed and the text stays the wire channel.

The logits scratch is one `n_vocab` vector in the source, reused under the lock; the real
vocabulary is 248,320 ids (`abi/rad_builder.h:1668`, `.haru-ref/real-config.json:11-15`), about
1 MB.

### 5.2 The detokenizer and the `flags` text bit

`StreamDetokenizer` is constructed per lane with `decode` and the silent set `{pad_id, eop_id}`
(`core/gen/duplex_sampler.h:128-133`; `.haru-ref/session.py:73`, `:93`). The decode callback must
match `.haru-ref/sampling.py:124`'s `skip_special_tokens=False`; `TokenizerBridge::decode` uses
`render_special=false` (`core/server/adapters.h:371-375`), so the production source should use a
dedicated `Tokenizer::decode(ids, /*render_special=*/true)` (`core/text/tokenizer.cpp:1656-1672`),
not that bridge, or the EOS/control-token text differs from the reference.

The wire's bit 1 is already computed from the reply text: `step_message` sets it iff `r.text` is
non-empty (`core/server/duplex.cpp:75-82`). No change is needed there. Consequences:

* A run of PAD/EOP yields one space on the first silent frame and `""` after it
  (`.haru-ref/sampling.py:116-121`), so that frame has bit 1 set with a single-space delta.
* An empty delta (a second silent frame, or a held partial UTF-8) leaves bit 1 clear.
* Every real step still has bit 0 set (audio present); status 1 is a silent frame
  (`core/server/duplex.cpp:170-175`).
* Token 0's delta is consumed at PROMPT and dropped with frame 0, so the first wire reply starts
  from token 1's delta, not a leading space from token 0.

### 5.3 Getting the text to the server, and the clamp

Path: source `reply_text()` -> `gen::DuplexReply::{text,host_text}`
(`core/gen/duplex_driver.h:51-60`) -> `EngineDuplexBridge` copies into
`server::DuplexGenReply::{text,host_text}` (`core/server/duplex_engine.cpp:55-57`, `:78-80`) ->
`EngineDuplexLane::map` builds `server::DuplexReply` (`core/server/duplex.cpp:167-184`) ->
`step_message` (`core/server/duplex.cpp:75-89`).

`EngineDuplexLane::map` is **the one engine-to-wire conversion**. Change it as follows:

```cpp
DuplexReply map(const DuplexGenReply& r) const {
    if (r.status < 0) throw ...;
    DuplexReply out;
    if (r.status == 0) {
        out.audio = r.audio;
        for (float& x : out.audio) x = std::clamp(x, -1.0f, 1.0f);  /* .haru-ref/session.py:163 */
    } else if (r.status == 1) {
        /* silent frame: flags = 0 */
    } else throw ...;
    if (r.host_text) {
        if (want_text) out.text = r.text;      /* already streamed; may be empty */
    } else if (r.token >= 0) {
        /* plugin-sampling tenants, unchanged */
        std::string text;
        if (!gen->decode_token(r.token, &text) && tok)
            text = tok->decode(std::vector<int32_t>{ r.token });
        out.text = std::move(text);
    }
    return out;
}
```

* `want_text` is a new `bool` on `EngineDuplexLane`, set from `start.text` and the tenant's `text`
  capability at open (`core/server/duplex.cpp:492-530`). Sampling still runs when it is false; only
  the delta is suppressed.
* The clamp must **not** move to `step_message`: the echo tenants return their input unchanged
  (`.haru-ref`/contract: `docs/DUPLEX-ENGINE.md:125-130`; `core/server/duplex.cpp:107-127`), and
  the contract says echo is bit-exact. Clamping in `map` covers `step` and `flush` (both call
  `map`, `core/server/duplex.cpp:186-206`) and leaves echo untouched.
* The clamp values are the frame contract's `[-1, 1]` (`docs/DUPLEX-ENGINE.md:67-84`,
  `.haru-ref/session.py:163`), not the declaration's geometry. Non-finite samples are a plugin bug;
  the source/test should assert finite values. If the wire must never carry NaN, map NaN to 0 and
  infinities to the endpoints after the clamp; that is a one-line decision in `map`, not an ABI
  change.
* Plugin-sampling tenants keep the old per-token decode because `r.host_text` is false; no echo or
  plugin code changes.

---

## 6. (e) Attaching the source for the `haru` tenant only

The only duplex program is the Haru tenant, and the gate is the declaration itself:

```cpp
/* core/engine.cpp, inside the `if (dup_gen_)` block at :3486-3516, after the bridge is built
 * and before the server can accept a session. */
if (dd.host_sampling == 1) {
    server::HostSampleSource::Options o;
    o.sampler = duplex_sampler_from_config(cfg_);   /* section 4, defaults from the reference */
    o.codec_noise_scale = dd.codec_noise_scale;
    o.pad_id = dd.pad_id; o.eop_id = dd.eop_id; o.eos_id = dd.eos_id;
    o.n_vocab = dd.n_vocab; o.latent_dim = dd.latent_dim; o.max_lanes = dd.max_lanes;
    o.decode = [tok = std::make_shared<Tokenizer>()](const std::vector<int>& ids) { ... };
    duplex_host_ = std::make_unique<server::HostSampleSource>(std::move(o));
    dup_gen_->set_host_source(duplex_host_.get());
}
```

* Store `std::unique_ptr<server::HostSampleSource> duplex_host_` on `Engine` so it outlives the
  server and the worker threads. `set_host_source` is a plain pointer store
  (`core/gen/duplex_driver.h:166`); it must run before any lane opens, which is true here because
  `run_server` builds the bridge before it starts accepting. `serve()` starts workers first
  (`core/engine.cpp:2326-2334`), but no lane exists until the listener runs. On teardown clear the
  pointer (`set_host_source(nullptr)`) before the source is destroyed, or declare the source
  before `gen_` so it outlives the driver; no pass runs after `shutdown()`.
* Gate on `dd.host_sampling == 1`, which is exactly the ABI 39 contract: a plugin-sampling tenant
  (`host_sampling == 0`) leaves `host_source_` null and keeps the old `text_out` path
  (`core/gen/duplex_driver.cpp:1159-1161`). Today only Haru declares 1
  (`arch/haru/haru.cpp:231-238`). If a second host-sampling tenant ever appears, the source is
  generic (constants from its declaration); the alternative is to add `dd.name == "haru"` to the
  gate, which is redundant today.
* The echo and echo1 tenants are registered separately by `Server::init_duplex_tenants` under
  `--duplex-test-tenants` and never use `dup_gen_` or the driver
  (`core/server/duplex.cpp:315-339`); they are unchanged. They are also the reason the clamp
  cannot live in `step_message`.
* Tests that drive `DuplexDriver` directly (`Engine::start_workers`, `engine.duplex_driver()`;
  `core/engine.h:90-95`) do not pass through `run_server`; the engine-level test attaches its
  source explicitly, as `tests/haru_assembly_test.cpp` does today (`:253`).

---

## 7. (f) Test plan

All tests are for the reviewer's Linux build; this host cannot compile or run them.

### 7.1 Unit tests with fakes: `tests/duplex_host_test.cpp`

A fake decode callback (`id -> "A"`, a two-id UTF-8 character, etc.) and synthetic logits; no
engine, no device, no plugin.

* **Lane lifetime.** `lane_opened(3, seed)` twice with `lane_closed(3)` between gives identical
  first draws; a close followed by a new seed gives a different stream. No state survives
  `lane_closed`; a lane index reused with a new seed has no DRY history or detokenizer gap.
* **Seed delivery.** The source uses the seed from `lane_opened`, not a global; two lanes with
  different seeds in one source have independent streams; the same lane's stream is unaffected by
  another lane opening, stepping or closing.
* **Reference noise recipe.** With an injected normal source (or a test accessor over the PRNG),
  feed the golden `prompt.ref_noise_scale` (`[1]`) and `prompt.ref_noise_raw` (`[14,64]`)
  (`tests/data/haru/README.md:175-177`) and assert
  `out[f,d] == scale * (latent_std/0.8) * raw[f,d]` matches `prompt.ref_noise` when
  `codec_noise_scale = 0.5/0.8`. Assert one scale is consumed for the clip, not one per frame.
  Assert card rows stay zero.
* **User/head noise order.** Script the PRNG (or inject normals) and assert a STEP consumes scale
  + 64 user normals, then 64 head normals; a PROMPT consumes reference scale + `F*64`, then 64
  head normals; a second step continues the same lane stream. Assert the noise lands at entry `i`
  while the lane is `p.lane[i]`, and that entry order does not affect each lane's draws.
* **Sampler wiring.** Given a row with a huge PAD logit, a huge EOP logit and a normal token,
  assert the token and that `pad_eop` replaces the two entries (not the raw logits), DRY is applied
  first, `eop_bias` moves EOP, and `temperature = 0` is greedy. Reuse the S2 golden vectors where
  possible (`tests/duplex_sampler_test.cpp` already covers the sampler itself).
* **Streamed text.** Script tokens `[PAD, EOP, PAD, "A", "B"]` (with a fake decode) and assert the
  deltas are `[" ", "", "", "A", "B"]`; PROMPT's token 0 delta is discarded; the first STEP's
  `reply_text` is token 1's delta; the cumulative text equals the detokenizer's stream from token
  1. A two-id UTF-8 character is held on the first id and emitted on the second.
* **Reply text plumbing.** `reply_text` returns each STEP delta exactly once; a later step or a
  flush does not repeat it. `tokens()` returns `-1` only on an error path; it returns a valid id for
  every lane otherwise.
* **Logits indexing.** A two-entry STEP pass with lanes `[1, 0]` returns tokens sampled from
  `logits + lane*n_vocab`, not from entry order. This catches the current teacher-source
  assumption (section 1.1).

### 7.2 Engine-level determinism: extend `tests/haru_assembly_test.cpp` or add `tests/haru_host_test.cpp`

Drive the real `DuplexDriver` with the real `haru` plugin and `tests/data/haru/haru_tiny`, as
`tests/haru_assembly_test.cpp:195-330` does, but attach the production `HostSampleSource` instead
of `TeacherSource`.

* **Same seed, same output.** Open a lane with `DuplexRequest.seed = 1234`, send a fixed sequence
  of user frames, collect `(token, audio)` per step; close; open a fresh lane with the same seed
  and frames; assert every token is equal and every audio sample is bit-equal (same build, same
  host). Also run with the same seed but a different lane index or a second concurrent lane and
  assert the same lane's stream is unchanged.
* **Different seed, different output.** The same frames with `seed = 5678` must differ; assert the
  audio vectors differ (tokens may coincidentally agree, audio with different codec/head noise
  should not; assert at least the full audio vector is not bit-equal, and prefer several steps).
* **Text for a scripted token sequence.** The real model's tokens are not scripted, so test the
  production source's text path in 7.1 and the wire path with a fake `IDuplexGen` in 7.3. A
  model-level variant may use `temperature = 0` and logits that force one token, but that is not
  required if 7.1 and 7.3 cover the sequence.
* **Clamp.** The tiny model normally stays in range, so the engine-level clamp is checked in 7.3
  with a fake reply; a model-level check asserts every reply sample is finite and in `[-1,1]`.

### 7.3 Wire-level: extend `tests/duplex_bridge_test.cpp`

The fake `IDuplexGen` already sits behind the real `EngineDuplexTenant` and WebSocket path
(`tests/duplex_bridge_test.cpp:170-230`, `:365-399`).

* **Host text and flags.** Set `DuplexGenReply.host_text = true` and `text = "A"` with a token id;
  assert the reply text is `"A"` (not the tokenizer's per-token decoding) and flags bit 1 is set.
  Set `text = ""`; assert flags bit 1 is clear while bit 0 stays set. Start with `text: false`;
  assert no text on the wire even though the reply carries it.
* **Clamp.** Return samples `2.0f`, `-3.0f`, `0.5f` from the fake and assert the wire samples are
  `1.0f`, `-1.0f`, `0.5f`. Also assert the echo tenant still returns a valid input frame
  bit-exactly (`.haru-ref` contract; `docs/DUPLEX-ENGINE.md:125-130`), proving the clamp is on the
  engine path only.
* **Silent frame.** Status 1 with a host text delta: audio absent, bit 0 clear; text still maps per
  `want_text`.
* **Plugin-sampling regression.** A fake with `host_text = false` and a token still decodes via
  `decode_token`/`IDuplexGen::decode_token` (`core/server/iface.h:816-822`), unchanged.

### 7.4 ABI and builder

* `tests/duplex_builder_test.cpp` / `tests/abi_layout_test.cpp`: a `host_sampling = 1` declaration
  with `codec_noise_scale <= 0` or non-finite is refused; the new field appears in the layout table
  after regeneration; the ABI version remains 39.
* A declaration with `host_sampling = 0` keeps working with the field zero.

### 7.5 Forge conformance suite (reviewer runs it)

Forge is the wire/lifecycle gate: info, start/reference, binary step, flags, seq echo, text delta,
flush, close, error codes, lane limit, no card/reference/audio in logs, and p95 latency. It is not
a bit-parity test: production sampling is stochastic. The tiny engine-level test above is the
numeric/determinism gate; Forge is the conformance gate. Add assertions to the Forge run that the
new behaviour holds end to end:

* every reply audio sample is finite and in `[-1, 1]` (the clamp);
* text deltas are valid UTF-8, a PAD/EOP run produces at most one space, and no PAD/EOP token
  string leaks;
* flags bit 1 agrees with a non-empty text delta; bit 0 agrees with the audio length;
* `start.text = false` yields no text bits while the lane still answers every step;
* same `session_id` and same input frames replay the same tokens/audio within the same server
  build (a strong determinism check; different `session_id` seeds differ).

I cannot run Forge here.

---

## 8. (g) What I cannot verify

* **No build or run on this host.** I cannot compile the proposed signatures, regenerate
  `tests/abi_layout_table.h`, link the new `core/server/duplex_host.cpp`, or run any test. Every
  line and every tolerance must come from the reviewer's Linux build.
* **No model run.** I have not measured the tiny-model determinism, the audio equality, the text
  deltas, the clamp, the p95 step time, or memory. The determinism claim assumes the plugin is
  deterministic given the same `text_in`, codec noise and head noise, which the H5 teacher parity
  supports but this host cannot confirm with the production source.
* **Forge is external.** I have not inspected its source or run its suite; the "what Forge sends"
  statement is the wire contract (`docs/DUPLEX-ENGINE.md:48-65`) plus the in-repo harness client
  (`tests/haru_harness.h:417-430`), not a reading of Forge.
* **Tokenizer semantics.** I cannot confirm that `core/text`'s decode with `render_special=true`
  matches the reference's HF `decode(..., skip_special_tokens=False)` for the real Haru vocabulary
  (byte fallback, U+FFFD suffix, EOS rendering). The tiny byte-level tokenizer may hide a real-BPE
  merge or decoder difference. The H5 note already flags prompt-tokenisation merges
  (`docs/HARU-H5-DESIGN.md:597-607`); the text stream needs the same real-tokenizer check.
* **Seed stability.** `std::hash<std::string>` (`core/server/duplex.cpp:291`) is
  implementation-defined. Replay is guaranteed only within one standard library/build; a
  cross-build replay needs a fixed hash (FNV-1a/SipHash with pinned constants) or a client-supplied
  seed. I have not changed the seed derivation here.
* **libm bit equality.** Box-Muller uses `log`, `cos`, `sin`, `sqrt`; different libm versions can
  differ in the last bits. Same build/machine replay is exact; cross-platform bit replay is not.
  A fixed table or integer normal is the remedy if that is required.
* **The reference frame-noise grouping.** The one-scale-per-clip claim comes from the golden schema
  (`prompt.ref_noise_scale [1]`, `tests/data/haru/README.md:175-177`) because the read-only copy
  does not include `prompt.py`. I cannot check the real `build_prompt` or a real-checkpoint prompt
  with multiple reference clips (the server currently stages one clip).
* **Real checkpoint declaration.** I cannot confirm that `cg.latent_std` is populated from the real
  container's `codec.latent_std`; the code path is `arch/haru_common/haru_codec.cpp:356`, but the
  real conversion/load is a reviewer check.
* **Multi-lane readback indexing.** I believe the readbacks are lane-indexed and the header
  comment is wrong (section 1.1), but I cannot run a two-lane pass to prove it. The design uses
  lane indexing and the test in 7.1 is the proof.
* **Thread safety under TSAN.** I have not run a race detector; the claim that all seam calls are
  under the driver mutex is from the current code paths (`core/gen/duplex_driver.cpp:1032`,
  `:1097`), not a measured result. A future driver change could invalidate it.
* **`speak_first` product behaviour.** The design keeps the flag off; I cannot decide from this
  tree whether the first reply should carry frame 0's text. That is a wire/product decision.

---

## 9. Files the next step will touch

* `docs/HARU-H6-DESIGN.md` (this note).
* `abi/rad_builder.h`: `RadDuplexDecl::codec_noise_scale` after `text_in`
  (`:1673-1676`), comment and validation hook.
* `abi/rad_abi_layout_rows.h`, `tests/abi_layout_table.h`, `tests/abi_layout_test.cpp`: the new
  field; regenerate the table on Linux; `RAD_ABI_VERSION` stays 39.
* `core/gen/duplex_driver.h`: `DuplexHostPass::ref_frame`; `lane_opened`/`lane_closed`/
  `reply_text`; `DuplexReply::text`/`host_text`; fix the `logits`/`pad_eop` lane-indexing comment
  (`:116-117`).
* `core/gen/duplex_driver.cpp`: call the lane notifications (`:500-510`, `:373-385`), pass
  `ref_frame` (`:751-767`), collect reply text and `host_text` (`:1153-1167`).
* `core/server/duplex_host.{h,cpp}`: `HostSampleSource`, its per-lane state, noise PRNG and text.
* `core/server/iface.h`: `DuplexGenReply::text`/`host_text` (`:792-798`).
* `core/server/duplex_engine.cpp`: copy them (`:55-57`, `:78-80`).
* `core/server/duplex.cpp`: `EngineDuplexLane::want_text`, `map` uses `host_text` and clamps audio
  (`:167-184`); set `want_text` at open (`:492-530`).
* `core/engine.h`/`core/engine.cpp`: own and attach `HostSampleSource` in `run_server`
  (`:3486-3516`) behind `dd.host_sampling == 1`.
* `core/rad_core.h`, `core/config.cpp`: the `--duplex-*` sampling flags
  (`:1135-1154`, `:1063-1075`, `:1527-1533`).
* `arch/haru/haru.cpp`: set `d.codec_noise_scale = cg.latent_std / 0.8`
  (`:231-238`; `cg` from `:116-122`).
* `tests/duplex_host_test.cpp`, `tests/duplex_bridge_test.cpp`, `tests/haru_assembly_test.cpp`
  (or `tests/haru_host_test.cpp`), `tests/duplex_builder_test.cpp`.
* `docs/PLUGIN.md`, `docs/DUPLEX-PLUGIN-ABI.md`: record the field and the production source.
* `DUPLEX-REPORT-H6.md` (untracked): the implementation report after the reviewer's build.
