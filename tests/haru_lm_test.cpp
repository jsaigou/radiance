/* tests/haru_lm_test.cpp -- Haru's LM trunk against the committed goldens.
 *
 * The plugin radtest_haru_lm (tests/haru_plugins/lm/) serves one duplex tenant, "haru-lm", whose
 * reply packs, in the first 3200 floats:
 *
 *   [0, n_embd)                     the final normed hidden of the row this step fed
 *   [n_embd, n_embd + 4*n_embd)     the four layer taps (one-row passes only)
 *   [n_embd + 4*n_embd, +n_vocab)   the 512 logits
 *   [.., +2)                        the f32 PAD/EOP logits
 *
 * The row order is the one docs/HARU-H4-DESIGN.md section 8.1 fixes.  Opening the lane runs the
 * PROMPT pass over golden prompt row 0: the plugin's template "<|im_start|>{card}" with the
 * harness' empty card tokenises to the single special token 482, whose embedding is
 * `prompt.embed[0]`.  Then the remaining 36 prompt rows and the 24 recorded frames are one STEP
 * each; step i is at position i, so the last prompt row is position 36 and frame 1 is position 37,
 * as in the reference.
 *
 * What this proves: the GDN's recurrent form (one row a step, gdn_conv_update +
 * gdn_recurrent_update) and a one-row paged attention reproduce the reference state that its
 * *37-row chunked* prefill produced.  What it does not prove: the chunked scan's own numerics
 * (gdn_conv_prep/kkt_solve/chunk_scan); the two forms are the same recurrence algebraically, so
 * this is a meaningful but not bit-level cross-check (design note section 8.5).
 *
 * Tolerances: the recommended absolute values from tests/data/haru/README.md's "Numerical
 * sensitivity" table (float64 replay against the f32 goldens) with rtol 1e-4.  They are a starting
 * point measured from the reference's own operation order; expect_close prints every family's
 * max_abs/max_rel on each run, and the values from the first Linux run should be recorded here.
 */
#include "haru_harness.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

using haru_test::Diff;
using haru_test::Golden;
using haru_test::Lane;
using haru_test::Tol;

/* The reply layout, derived from the goldens so a regenerated tiny model cannot silently move a
 * field. */
struct Layout {
    int64_t ne = 0;        /* hidden width (64) */
    int64_t nl = 0;        /* trunk layers (4) */
    int64_t nv = 0;        /* vocabulary (512) */
    int64_t taps_off = 0;
    int64_t logits_off = 0;
    int64_t pad_off = 0;
    int64_t reply = 0;
};

Layout layout() {
    const Golden& g = Golden::get();
    Layout l;
    l.ne = g.shape("prompt.hidden_last")[0];
    while (g.has("prompt.lm_layer" + std::to_string(l.nl))) ++l.nl;
    l.nv = g.shape("frame0.logits")[0];
    l.taps_off = l.ne;
    l.logits_off = l.taps_off + l.nl * l.ne;
    l.pad_off = l.logits_off + l.nv;
    l.reply = l.pad_off + 2;
    return l;
}

std::vector<float> slice(const std::vector<float>& v, int64_t off, int64_t n) {
    return std::vector<float>(v.begin() + (std::ptrdiff_t)off,
                              v.begin() + (std::ptrdiff_t)(off + n));
}

/* A golden row of a [rows, n] f32 tensor. */
std::vector<float> row(const std::vector<float>& m, int64_t i, int64_t n) {
    return slice(m, i * n, n);
}

/* Print the Diff and record a CHECK-style failure (the case continues, so one bad step does not
 * hide the families after it). */
void expect_close(const char* what, const std::vector<float>& got, const std::vector<float>& want,
                  Tol tol) {
    const Diff d = haru_test::compare(got, want, tol);
    std::fprintf(stderr, "    %-38s %s\n", what, d.text().c_str());
    ++radtest::checks();
    if (!d.ok) radtest::fail(__FILE__, __LINE__, std::string(what) + ": " + d.text());
}

/* Recommended absolutes from the README table, at rtol 1e-4. */
constexpr Tol kHidden{ 2e-5, 1e-4 };    /* lm_hidden: 2.0e-5 */
constexpr Tol kLayer{ 1e-5, 1e-4 };     /* lm_layer0..3: 8-9e-6 */
constexpr Tol kLogits{ 2e-5, 1e-4 };    /* logits: 2.0e-5 */
constexpr Tol kPadEop{ 2e-5, 1e-4 };    /* the same rows, recomputed in f32 */

struct TestBed {
    std::unique_ptr<haru_test::Engine> engine;
    std::string why;

    TestBed() {
        /* The lane bound is the probe's 128 rows, not the engine's 13,500-row duplex default:
         * the tiny test drives 61 rows and should not reserve a 30-minute conversation. */
        engine = haru_test::Engine::start({ "--duplex-max-ctx", "128" }, &why);
        if (!engine) throw std::runtime_error("the engine did not start: " + why);
    }
};

std::unique_ptr<Lane> open_lane(haru_test::Engine& e, std::string* why) {
    auto lane = Lane::open(e, "haru-lm", why);
    if (!lane) std::fprintf(stderr, "    lane open failed: %s\n", why->c_str());
    return lane;
}

bool step(Lane& lane, const std::vector<float>& in, std::vector<float>* out, std::string* why) {
    if (!lane.step(in, out, why)) {
        std::fprintf(stderr, "    step failed: %s\n", why->c_str());
        return false;
    }
    return true;
}

/* A row's four families, checked against the named golden row.  `taps` may be empty when the
 * goldens do not keep that row's layers (frames after the third). */
void check_row(const char* prefix, const Layout& l, const std::vector<float>& out,
               const std::string& hidden, const std::string& taps,
               const std::string& logits, const std::string& pad) {
    const Golden& g = Golden::get();
    expect_close((prefix + std::string(" hidden")).c_str(), slice(out, 0, l.ne),
                 g.f32(hidden), kHidden);
    if (!taps.empty() && l.nl > 0)
        for (int64_t i = 0; i < l.nl; ++i) {
            const std::string name = taps + std::to_string(i);
            expect_close((std::string(prefix) + " lm_layer" + std::to_string(i)).c_str(),
                         slice(out, l.taps_off + i * l.ne, l.ne), g.f32(name), kLayer);
        }
    expect_close((prefix + std::string(" logits")).c_str(), slice(out, l.logits_off, l.nv),
                 g.f32(logits), kLogits);
    expect_close((prefix + std::string(" logits_pad_eop")).c_str(), slice(out, l.pad_off, 2),
                 g.f32(pad), kPadEop);
}

}  /* namespace */

TEST(the_trunk_matches_the_reference_over_the_prompt_and_every_frame) {
    TestBed t;
    const Layout l = layout();
    REQUIRE(l.ne > 0);
    REQUIRE(l.nl > 0);
    REQUIRE(l.nv > 0);
    REQUIRE(l.reply <= (int64_t)haru_test::Engine::kSamples);

    const Golden& g = Golden::get();
    const std::vector<float> pe = g.f32("prompt.embed");
    const int64_t np = g.shape("prompt.tokens")[0];
    REQUIRE_EQ(np, (int64_t)37);
    REQUIRE_EQ(pe.size(), (size_t)(np * l.ne));

    std::string why;
    std::unique_ptr<Lane> lane = open_lane(*t.engine, &why);
    REQUIRE(lane != nullptr);

    std::vector<float> out;
    for (int64_t i = 1; i < np; ++i) {
        if (!step(*lane, row(pe, i, l.ne), &out, &why)) REQUIRE(false);
        REQUIRE_EQ(out.size(), (size_t)haru_test::Engine::kSamples);
        if (i != np - 1) continue;

        /* The last prompt row is the position that produced frame 0. */
        check_row("prompt.last", l, out, "prompt.hidden_last", "prompt.lm_layer",
                  "frame0.logits", "frame0.logits_fp32_pad_eop");
        expect_close("frame0 hidden (redundant)", slice(out, 0, l.ne), g.f32("frame0.hidden"),
                     kHidden);
    }

    int64_t nf = 0;
    while (g.has("frame" + std::to_string(nf + 1) + ".hidden")) ++nf;
    REQUIRE_EQ(nf, (int64_t)24);

    for (int64_t f = 1; f <= nf; ++f) {
        const std::string pre = "frame" + std::to_string(f);
        if (!step(*lane, g.f32(pre + ".embed"), &out, &why)) REQUIRE(false);
        REQUIRE_EQ(out.size(), (size_t)haru_test::Engine::kSamples);
        /* The first three frames keep their layer taps. */
        const std::string taps = (f <= 3) ? pre + ".lm_layer" : std::string();
        check_row(pre.c_str(), l, out, pre + ".hidden", taps, pre + ".logits",
                  pre + ".logits_fp32_pad_eop");
    }

    lane->close();
    CHECK_EQ(t.engine->stop(), 0);
}

TEST(two_lanes_keep_independent_state) {
    TestBed t;
    const Layout l = layout();
    const Golden& g = Golden::get();
    const std::vector<float> pe = g.f32("prompt.embed");
    std::string why;

    std::unique_ptr<Lane> a = open_lane(*t.engine, &why);
    REQUIRE(a != nullptr);
    std::unique_ptr<Lane> b = open_lane(*t.engine, &why);
    REQUIRE(b != nullptr);

    /* Both lanes run the same four rows at the same time, so if the driver batches them the block
     * sees a two-row pass (no taps, two logits rows) — and each lane's bytes must still be its
     * own. */
    std::vector<std::vector<float>> outa(4), outb(4);
    std::atomic<int> ready{ 0 };
    std::atomic<bool> ok_a{ true }, ok_b{ true };
    std::string why_a, why_b;

    auto run = [&](Lane* lane, std::vector<std::vector<float>>* outs, std::string* w,
                   std::atomic<bool>* ok) {
        ready.fetch_add(1);
        while (ready.load() < 2) std::this_thread::yield();
        for (int i = 0; i < 4 && ok->load(); ++i) {
            if (!lane->step(row(pe, 1 + i, l.ne), &(*outs)[(size_t)i], w)) {
                ok->store(false);
                return;
            }
        }
    };
    std::thread ta(run, a.get(), &outa, &why_a, &ok_a);
    std::thread tb(run, b.get(), &outb, &why_b, &ok_b);
    ta.join();
    tb.join();
    if (!ok_a.load()) {
        std::fprintf(stderr, "    lane a failed: %s\n", why_a.c_str());
        REQUIRE(false);
    }
    if (!ok_b.load()) {
        std::fprintf(stderr, "    lane b failed: %s\n", why_b.c_str());
        REQUIRE(false);
    }

    for (int i = 0; i < 4; ++i) {
        REQUIRE_EQ(outa[(size_t)i].size(), (size_t)haru_test::Engine::kSamples);
        REQUIRE_EQ(outb[(size_t)i].size(), (size_t)haru_test::Engine::kSamples);
        const std::string pre = "row" + std::to_string(i + 1);
        expect_close((pre + " a/b hidden").c_str(), slice(outa[(size_t)i], 0, l.ne),
                     slice(outb[(size_t)i], 0, l.ne), kHidden);
        expect_close((pre + " a/b logits").c_str(), slice(outa[(size_t)i], l.logits_off, l.nv),
                     slice(outb[(size_t)i], l.logits_off, l.nv), kLogits);
    }
    a->close();
    b->close();
}

TEST(closing_and_reopening_a_lane_starts_from_a_fresh_state) {
    TestBed t;
    const Layout l = layout();
    const Golden& g = Golden::get();
    const std::vector<float> pe = g.f32("prompt.embed");
    std::string why;

    std::unique_ptr<Lane> a = open_lane(*t.engine, &why);
    REQUIRE(a != nullptr);

    std::vector<std::vector<float>> first(4);
    for (int i = 0; i < 4; ++i) {
        if (!step(*a, row(pe, 1 + i, l.ne), &first[(size_t)i], &why)) REQUIRE(false);
        REQUIRE_EQ(first[(size_t)i].size(), (size_t)haru_test::Engine::kSamples);
    }
    a->close();

    /* A new lane gets a new KV sequence and zeroed state; the rows must come out as they did on
     * the first lane, not as a continuation of it. */
    std::unique_ptr<Lane> b = open_lane(*t.engine, &why);
    REQUIRE(b != nullptr);
    for (int i = 0; i < 4; ++i) {
        std::vector<float> out;
        if (!step(*b, row(pe, 1 + i, l.ne), &out, &why)) REQUIRE(false);
        REQUIRE_EQ(out.size(), (size_t)haru_test::Engine::kSamples);
        const std::string pre = "reopen row" + std::to_string(i + 1);
        expect_close((pre + " hidden").c_str(), slice(out, 0, l.ne),
                     slice(first[(size_t)i], 0, l.ne), kHidden);
        expect_close((pre + " logits").c_str(), slice(out, l.logits_off, l.nv),
                     slice(first[(size_t)i], l.logits_off, l.nv), kLogits);
    }
    b->close();
}

RAD_TEST_MAIN()
