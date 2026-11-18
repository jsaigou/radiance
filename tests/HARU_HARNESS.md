# Haru test harness: the shared API (tests/haru_harness.h)

Every Haru building block (codec, head and projectors, LM trunk, later the assembled plugin) is checked the same way: a
**test plugin** wraps the block, the **real `radiance` binary** serves it on the host backend (libref) with the tiny
checkpoint in `tests/data/haru/haru_tiny`, a test **drives a lane over WebSocket** and compares what comes back with the
**goldens** in `tests/data/haru/golden.safetensors` (schema: `tests/data/haru/README.md`). This file is the API all packages
code against. The harness itself is `tests/haru_harness.h` (header-only is fine; one `.cpp` if needed); a package must not
fork or copy it.

## Tensor tunnelling (the convention for test plugins)

A duplex step carries exactly 3200 float32 in and 3200 float32 out. A test plugin is a duplex tenant whose STEP computes the
block under test from `audio_in` and writes its result to `audio_out`. A tensor of N <= 3200 floats occupies the first N
samples; the rest are zero. A tensor longer than 3200 uses several steps (document the order in the test). The text channel
and status are unused. Per-lane state (a codec's history, a KV cache) lives in the plugin and is reset when a lane opens. A
plugin may declare several tenants (for example `encode` and `decode`); the test picks one in the start message. Integers
(token ids) are passed as floats and cast in the plugin (exact for ids < 2^24).

## API

```cpp
namespace haru_test {

// Golden tensors, loaded once from RAD_HARU_DATA "/golden.safetensors" (read with tiny::read_safetensors).
struct Golden {
    static const Golden& get();                       // loads on first use; aborts the test with a clear message on failure
    bool has(const std::string& name) const;
    std::vector<float>   f32(const std::string& name) const;   // float32 tensors, flattened row-major
    std::vector<int64_t> i64(const std::string& name) const;   // int64 tensors
    std::vector<int64_t> shape(const std::string& name) const;
};

// A tolerance: an element passes when |got - want| <= atol + rtol * |want|.
struct Tol { double atol, rtol; };

struct Diff {
    bool   ok;
    double max_abs, max_rel;      // over all elements
    size_t worst_index;           // index of the largest violation of the tolerance, or of max_abs when ok
    float  got, want;             // at worst_index
    std::string text() const;     // one line: "ok max_abs=... max_rel=..." or "FAIL idx=... got=... want=... max_abs=..."
};
Diff compare(const std::vector<float>& got, const std::vector<float>& want, Tol tol);   // sizes must match, else ok=false
// REQUIRE-style macro for the repo's test framework: prints the Diff text on failure.
#define HARU_REQUIRE_CLOSE(got, want, tol) /* uses rad_test.h's REQUIRE; message includes the label */

// The real engine, started once per test: RAD_E2E_SERVER, host backend, libref only, a temporary RADIANCE_HOME holding
// the plugin RAD_HARU_PLUGIN (as <home>/architectures/<name>.so) and libref, the model directory tests/data/haru/haru_tiny.
class Engine {
public:
    static std::unique_ptr<Engine> start(const std::vector<std::string>& extra_args, std::string* why);
    int port() const;
    std::string log() const;                           // the child's output so far, for failure messages
    ~Engine();                                         // stops the child; keeps its temp directory when RAD_DUPLEX_KEEP is set
};

// One duplex lane over WebSocket (the engine's /duplex/v1/session).
class Lane {
public:
    static std::unique_ptr<Lane> open(Engine& e, const std::string& tenant, std::string* why,
                                      const std::string& card = "", const std::vector<uint8_t>& reference = {});
    // One frame in (3200 floats; shorter input is zero padded), one reply out (3200 floats). false + why on any failure.
    bool step(const std::vector<float>& in, std::vector<float>* out, std::string* why);
    // Convenience: a tensor of any length <= 3200 in, the first `n_out` floats of the reply out.
    bool run(const std::vector<float>& tensor, size_t n_out, std::vector<float>* out, std::string* why);
    void close();
    ~Lane();
};

}  // namespace haru_test
```

## Rules

- Lane protocol: `docs/DUPLEX-ENGINE.md` (JSON `start`, binary steps, one reply per step). The harness implements the client side;
  use cpp-httplib's WebSocket client (vendored, `vendor/cpp-httplib/httplib.h`), no new dependency.
- CMake wiring is already in `tests/CMakeLists.txt`: for a block `<b>` a test `tests/haru_<b>_test.cpp` and a plugin target
  `radtest_haru_<b>` (built from `tests/haru_plugins/<b>/CMakeLists.txt`, kind `archtest`) get `RAD_E2E_SERVER`, `RAD_E2E_HOME`,
  `RAD_HARU_PLUGIN` and `RAD_HARU_DATA` defined, and the e2e label.
- The block's own sources live in `arch/haru_common/haru_<b>.{h,cpp}` and are compiled into the test plugin by path (as hyv3_common is).
- Tolerances are evidence-based. Start from `Tol{1e-4, 1e-4}` for float32 on the host reference unless the README's sensitivity
  section says otherwise, record the measured maximum error in the test's comment, and tighten or justify any loosening in the report.
