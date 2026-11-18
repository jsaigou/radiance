/* tests/haru_harness_test.cpp -- the shared harness' own test.
 *
 * The Haru block tests code against tests/haru_harness.h before anyone can build it, so this test
 * is the harness' contract: compare()'s tolerance arithmetic, the goldens, and a real engine and
 * lane against the existing model-free radtest-duplex plugin (a zero-delay echo whose STEP also
 * writes a text token, so the text channel cannot hide an audio offset bug).
 *
 * The checkpoint is the same tiny one tests/duplex_plugin_test.cpp writes: the three weights the
 * plugin's name map names and a config whose model_type selects radtest_duplex. It is not a
 * tiny::write_llama checkpoint because that config names the llama plugin, which this home does
 * not carry.
 */
#include "haru_harness.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

namespace {

/* The two f32 text tables of the test plugin, and the scalar that makes its echo a multiply by
 * one (exact, including -0). */
tiny::Tensor f32_table(const char* name, int32_t base, int64_t rows) {
    tiny::Tensor t{ name, "F32", { rows, 1 }, {} };
    t.bytes.resize((size_t)rows * 4);
    for (int64_t i = 0; i < rows; ++i) {
        const float v = (float)(base + (int32_t)i);
        std::memcpy(t.bytes.data() + (size_t)i * 4, &v, 4);
    }
    return t;
}

bool write_duplex_checkpoint(const std::string& dir) {
    tiny::Tensor one{ "duplex.one", "F32", { 1 }, {} };
    one.bytes.resize(4);
    const float f = 1.0f;
    std::memcpy(one.bytes.data(), &f, 4);

    nlohmann::ordered_json c;
    c["architectures"] = { "RadTestDuplexForCausalLM" };
    c["model_type"] = "radtest_duplex";
    c["num_hidden_layers"] = 1;
    c["hidden_size"] = 2;
    c["num_attention_heads"] = 1;
    c["num_key_value_heads"] = 1;
    c["head_dim"] = 1;
    c["intermediate_size"] = 4;
    c["vocab_size"] = 320;
    c["max_position_embeddings"] = 4096;
    c["rms_norm_eps"] = 1e-5;
    c["tie_word_embeddings"] = false;
    std::vector<tiny::Tensor> ts;
    ts.push_back(f32_table("duplex.text0", 0, 1024));
    ts.push_back(f32_table("duplex.text1", 1000, 1024));
    ts.push_back(one);
    return tiny::write_checkpoint(dir, c, ts);
}

/* A temporary checkpoint and a running engine, in one object whose constructor throws with the
 * engine's own log when the start fails. Tests use it and let rad_test report the exception. */
struct TestBed {
    tiny::TempDir dir;
    std::unique_ptr<haru_test::Engine> engine;
    std::string why;

    TestBed()
        : dir("haru_harness_e2e", std::getenv("RAD_DUPLEX_KEEP") != nullptr) {
        if (!dir.ok())
            throw std::runtime_error("could not make a temporary checkpoint directory");
        if (!write_duplex_checkpoint(dir.path))
            throw std::runtime_error("could not write the tiny duplex checkpoint under " +
                                     dir.path);
        engine = haru_test::Engine::start_with(RAD_HARU_PLUGIN, dir.path, {}, &why);
        if (!engine) throw std::runtime_error("the engine did not start: " + why);
    }
};

/* A frame whose samples are distinct and exactly representable, so a bit-exact comparison is the
 * only thing that can differ. */
std::vector<float> frame(int seed) {
    std::vector<float> f(3200);
    for (size_t i = 0; i < f.size(); ++i)
        f[i] = (float)(seed * 4096 + (int)i) - 20000.0f;
    return f;
}

std::unique_ptr<haru_test::Lane> open_lane(haru_test::Engine& e, const std::string& tenant,
                                           std::string* why) {
    auto lane = haru_test::Lane::open(e, tenant, why);
    if (!lane) fprintf(stderr, "    lane open failed: %s\n", why->c_str());
    return lane;
}

}  /* namespace */

TEST(compare_semantics) {
    using haru_test::Tol;
    using haru_test::compare;

    {
        std::vector<float> a = { 1.0f, 2.0f, 3.0f }, b = a;
        const auto d = compare(a, b, Tol{ 0, 0 });
        CHECK(d.ok);
        CHECK_EQ(d.max_abs, 0.0);
        CHECK_EQ(d.max_rel, 0.0);
        CHECK_EQ(d.worst_index, (size_t)0);
        CHECK_EQ(d.text().compare(0, 2, "ok"), 0);
    }
    {
        std::vector<float> a = { 1.00005f, 2.0f }, b = { 1.0f, 2.0f };
        CHECK(compare(a, b, Tol{ 1e-4, 0 }).ok);
    }
    {
        std::vector<float> a = { 105.0f }, b = { 100.0f };
        CHECK(compare(a, b, Tol{ 0, 0.1 }).ok);
        a[0] = 111.0f;
        CHECK(!compare(a, b, Tol{ 0, 0.1 }).ok);
    }
    {
        std::vector<float> a = { 1.0f, 1.0f, 2.0f, 1.0f }, b = { 1.0f, 1.0f, 1.0f, 1.0f };
        const auto d = compare(a, b, Tol{ 0, 0 });
        CHECK(!d.ok);
        CHECK_EQ(d.worst_index, (size_t)2);
        CHECK_EQ(d.got, 2.0f);
        CHECK_EQ(d.want, 1.0f);
        CHECK(has(d.text(), "idx=2"));
        CHECK(has(d.text(), "got=2"));
        CHECK(has(d.text(), "want=1"));
    }
    {
        std::vector<float> a = { 1.0f, 2.0f }, b = { 1.0f };
        CHECK(!compare(a, b, Tol{ 1, 1 }).ok);
    }
    {
        std::vector<float> a = { 1.0f, std::nanf("") }, b = { 1.0f, 1.0f };
        CHECK(!compare(a, b, Tol{ 1e9, 1e9 }).ok);
    }
    {
        const float w = 1.0f;
        std::vector<float> a = { w }, b = { w };
        CHECK(compare(a, b, Tol{ 0, 0 }).ok);
        a[0] = std::nextafter(w, 2.0f);
        CHECK(!compare(a, b, Tol{ 0, 0 }).ok);
    }
}

TEST(haru_require_close_accepts_a_matching_pair) {
    std::vector<float> a = { 1.0f, 2.0f, 3.0f }, b = { 1.0f, 2.0f, 3.0f };
    HARU_REQUIRE_CLOSE(a, b, haru_test::Tol{ 0, 0 });
}

TEST(golden_tensors_load) {
    const haru_test::Golden& g = haru_test::Golden::get();
    CHECK(g.has("prompt.tokens"));
    CHECK(g.has("frame1.audio_out"));
    CHECK(g.shape("prompt.tokens") == std::vector<int64_t>{ 37 });
    CHECK(g.shape("frame1.audio_out") == std::vector<int64_t>{ 3200 });
    CHECK_EQ(g.i64("prompt.tokens").size(), (size_t)37);
    CHECK_EQ(g.f32("frame1.audio_out").size(), (size_t)3200);
    CHECK(!g.has("no.such.tensor"));

    bool threw = false;
    std::string message;
    try {
        (void)g.f32("no.such.tensor");
    } catch (const std::exception& e) {
        threw = true;
        message = e.what();
    }
    CHECK(threw);
    CHECK(has(message, "no.such.tensor"));
}

TEST(engine_starts_and_stops_cleanly) {
    TestBed t;
    REQUIRE(t.engine != nullptr);
    CHECK(t.engine->port() > 0);
    CHECK(has(t.engine->log(), "listening on http://127.0.0.1:"));
    CHECK_EQ(t.engine->stop(), 0);
}

TEST(a_lane_echoes_six_distinct_frames_bit_exactly) {
    TestBed t;
    std::string why;
    std::unique_ptr<haru_test::Lane> lane = open_lane(*t.engine, "radtest-duplex", &why);
    REQUIRE(lane != nullptr);

    /* radtest-duplex declares text and writes a token every STEP, so the reply carries the text
     * field the harness has to skip before the audio. */
    for (int f = 0; f < 6; ++f) {
        const std::vector<float> in = frame(f + 1);
        std::vector<float> out;
        if (!lane->step(in, &out, &why)) {
            fprintf(stderr, "    step %d failed: %s\n", f, why.c_str());
            REQUIRE(false);
        }
        REQUIRE_EQ(out.size(), in.size());
        CHECK(std::memcmp(out.data(), in.data(), in.size() * sizeof(float)) == 0);
    }
    lane->close();
}

TEST(two_lanes_run_at_once_and_keep_their_own_data) {
    TestBed t;
    std::string why;
    std::unique_ptr<haru_test::Lane> a = open_lane(*t.engine, "radtest-duplex", &why);
    REQUIRE(a != nullptr);
    std::unique_ptr<haru_test::Lane> b = open_lane(*t.engine, "radtest-duplex", &why);
    REQUIRE(b != nullptr);

    for (int f = 0; f < 3; ++f) {
        const std::vector<float> fa = frame(100 + f), fb = frame(200 + f);
        std::vector<float> out;
        if (!a->step(fa, &out, &why)) {
            fprintf(stderr, "    lane a step %d failed: %s\n", f, why.c_str());
            REQUIRE(false);
        }
        REQUIRE_EQ(out.size(), fa.size());
        CHECK(std::memcmp(out.data(), fa.data(), fa.size() * sizeof(float)) == 0);
        if (!b->step(fb, &out, &why)) {
            fprintf(stderr, "    lane b step %d failed: %s\n", f, why.c_str());
            REQUIRE(false);
        }
        REQUIRE_EQ(out.size(), fb.size());
        CHECK(std::memcmp(out.data(), fb.data(), fb.size() * sizeof(float)) == 0);
    }
}

TEST(closing_a_lane_releases_it_and_a_step_after_close_fails) {
    TestBed t;
    std::string why;
    std::unique_ptr<haru_test::Lane> a = open_lane(*t.engine, "radtest-duplex", &why);
    REQUIRE(a != nullptr);
    std::unique_ptr<haru_test::Lane> b = open_lane(*t.engine, "radtest-duplex", &why);
    REQUIRE(b != nullptr);

    /* The tenant's limit is 2, so a third lane is refused while both are open. */
    std::string why3;
    std::unique_ptr<haru_test::Lane> refused =
        haru_test::Lane::open(*t.engine, "radtest-duplex", &why3);
    CHECK(refused == nullptr);
    CHECK(has(why3, "session_limit"));

    a->close();

    /* The lane that is still open is unaffected by the close. */
    const std::vector<float> fb = frame(300);
    std::vector<float> out;
    if (!b->step(fb, &out, &why)) {
        fprintf(stderr, "    lane b step failed after lane a closed: %s\n", why.c_str());
        REQUIRE(false);
    }
    REQUIRE_EQ(out.size(), fb.size());
    CHECK(std::memcmp(out.data(), fb.data(), fb.size() * sizeof(float)) == 0);

    /* The server releases the lane just after it answers the close, so give the release a moment
     * rather than assume the two orderings. */
    std::unique_ptr<haru_test::Lane> c;
    std::string whyc;
    for (int i = 0; i < 300 && !c; ++i) {
        c = haru_test::Lane::open(*t.engine, "radtest-duplex", &whyc);
        if (c) break;
        if (!has(whyc, "session_limit")) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!c) fprintf(stderr, "    third lane open failed: %s\n", whyc.c_str());
    CHECK(c != nullptr);

    std::vector<float> after;
    std::string why_closed;
    CHECK(!a->step(frame(400), &after, &why_closed));
    CHECK(has(why_closed, "closed"));
}

TEST(an_unknown_tenant_is_refused_with_the_engines_error) {
    TestBed t;
    std::string why;
    std::unique_ptr<haru_test::Lane> lane =
        haru_test::Lane::open(*t.engine, "no-such-tenant", &why);
    CHECK(lane == nullptr);
    CHECK(has(why, "tenant_unavailable"));
    CHECK(has(why, "no tenant by that name is registered"));
}

RAD_TEST_MAIN()
