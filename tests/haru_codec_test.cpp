/* tests/haru_codec_test.cpp -- the VibeVoice codec block against the H0 goldens.
 *
 * The plugin (tests/haru_plugins/codec) is one duplex tenant; the operation is selected by the
 * card's token count (tests/HARU_HARNESS.md): 1 encode, 2 decode, 3 debug stages, 4 the whole
 * reference clip, 6 sampling, 8 mode probe. The first case proves that mapping, because the rest
 * would silently test the wrong operation if the tokenizer merged the probe's letters.
 *
 * The tolerances are the per-family recommended absolute values from the "Numerical sensitivity"
 * table in tests/data/haru/README.md, with rtol 1e-4:
 *   codec_encode_mean 9e-6, codec_decode_audio 2e-6, user_latent 8e-6;
 *   enc_stage0..6 7e-7, 2e-6, 8e-7, 2e-6, 3e-6, 3e-6, 4e-6;
 *   dec_stage0..6 2e-6, 3e-6, 3e-6, 3e-6, 6e-6, 7e-6, 9e-6.
 * Every comparison prints its measured max_abs/max_rel, so the Linux run records the real maxima
 * next to these checks; anything that needs a looser bound is justified in the H1 report.
 */
#include "haru_harness.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char* kTenant = "haru-codec";

const char* kProbeCard = "abcdefgh";  /* 8 distinct ASCII letters -> mode 8 */
const char* kEncodeCard = "a";        /* mode 1 */
const char* kDecodeCard = "ab";       /* mode 2 */
const char* kDebugCard = "abc";       /* mode 3 */
const char* kClipCard = "abcd";       /* mode 4 */
const char* kSampleCard = "abcdef";   /* mode 6 */

const haru_test::Tol kEncMean{ 9e-6, 1e-4 };
const haru_test::Tol kDecAudio{ 2e-6, 1e-4 };
const haru_test::Tol kSampleTol{ 8e-6, 1e-4 };
const double kEncAtol[7] = { 7e-7, 2e-6, 8e-7, 2e-6, 3e-6, 3e-6, 4e-6 };
const double kDecAtol[7] = { 2e-6, 3e-6, 3e-6, 3e-6, 6e-6, 7e-6, 9e-6 };

std::unique_ptr<haru_test::Engine> start_engine(std::string* why) {
    auto engine = haru_test::Engine::start({}, why);
    if (!engine) std::fprintf(stderr, "    engine did not start: %s\n", why->c_str());
    return engine;
}

std::unique_ptr<haru_test::Lane> open_lane(haru_test::Engine& engine, const char* card,
                                           std::string* why) {
    /* A close answers before the server releases the lane, so a following open can briefly see
     * session_limit. Poll it; any other refusal is real. */
    std::unique_ptr<haru_test::Lane> lane;
    for (int i = 0; i < 300 && !lane; ++i) {
        lane = haru_test::Lane::open(engine, kTenant, why, card);
        if (lane) break;
        if (!has(*why, "session_limit")) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!lane)
        std::fprintf(stderr, "    lane open (card \"%s\") failed: %s\n", card, why->c_str());
    return lane;
}

bool close_enough(const std::string& label, const std::vector<float>& got,
                  const std::vector<float>& want, haru_test::Tol tol) {
    const haru_test::Diff d = haru_test::compare(got, want, tol);
    std::fprintf(stderr, "    %s: max_abs=%g max_rel=%g %s\n", label.c_str(), d.max_abs,
                 d.max_rel, d.ok ? "ok" : "FAIL");
    return d.ok;
}

bool run_step(haru_test::Lane& lane, const std::vector<float>& in, size_t n_out,
              std::vector<float>* out, std::string* why) {
    if (!lane.run(in, n_out, out, why)) {
        std::fprintf(stderr, "    step failed: %s\n", why->c_str());
        return false;
    }
    return true;
}

/* Two lanes stepped from two threads at the same moment, so a STEP pass may carry both. Correctness
 * must not depend on whether the driver batches them. */
bool step_two(haru_test::Lane& a, const std::vector<float>& ia, std::vector<float>* oa,
              std::string* wa, haru_test::Lane& b, const std::vector<float>& ib,
              std::vector<float>* ob, std::string* wb) {
    std::atomic<int> ready{ 0 };
    std::atomic<bool> go{ false };
    bool oka = false, okb = false;
    std::thread ta([&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        oka = a.step(ia, oa, wa);
    });
    std::thread tb([&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        okb = b.step(ib, ob, wb);
    });
    while (ready.load(std::memory_order_acquire) < 2) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    ta.join();
    tb.join();
    return oka && okb;
}

std::string frame_name(int t, const char* what) {
    return "frame" + std::to_string(t) + "." + what;
}

}  /* namespace */

TEST(mode_channel_self_check) {
    std::string why;
    auto engine = start_engine(&why);
    REQUIRE(engine != nullptr);
    auto lane = open_lane(*engine, kProbeCard, &why);
    REQUIRE(lane != nullptr);

    std::vector<float> out;
    if (!run_step(*lane, std::vector<float>(3200, 0.0f), 2, &out, &why)) {
        REQUIRE(false);
    }
    if (!(out.size() >= 2 && out[0] == 8.0f && out[1] == 8.0f)) {
        std::fprintf(stderr,
                     "    mode channel: card \"%s\" must tokenise to 8 tokens (mode 8); the reply "
                     "was [%g, %g]. The tokenizer's card mapping is not m letters -> m tokens, so "
                     "the mode channel cannot select the codec operations; see "
                     "tests/HARU_HARNESS.md.\n",
                     kProbeCard, out.size() > 0 ? out[0] : -1.0f,
                     out.size() > 1 ? out[1] : -1.0f);
        REQUIRE(false);
    }
    lane->close();
}

TEST(codec_encodes_the_user_frames) {
    const haru_test::Golden& g = haru_test::Golden::get();
    std::string why;
    auto engine = start_engine(&why);
    REQUIRE(engine != nullptr);
    auto lane = open_lane(*engine, kEncodeCard, &why);
    REQUIRE(lane != nullptr);

    for (int t = 1; t <= 24; ++t) {
        const auto want = g.f32(frame_name(t, "user_codec_mean"));
        std::vector<float> got;
        if (!run_step(*lane, g.f32(frame_name(t, "user_audio")), 64, &got, &why))
            REQUIRE(false);
        REQUIRE(close_enough(frame_name(t, "user_codec_mean"), got, want, kEncMean));
    }
    lane->close();
}

TEST(codec_encodes_the_whole_reference_clip) {
    const haru_test::Golden& g = haru_test::Golden::get();
    std::string why;
    auto engine = start_engine(&why);
    REQUIRE(engine != nullptr);
    auto lane = open_lane(*engine, kClipCard, &why);
    REQUIRE(lane != nullptr);

    const auto clip = g.f32("prompt.ref_audio");
    REQUIRE_EQ(clip.size(), (size_t)44000);
    std::vector<float> got;
    for (int f = 0; f < 14; ++f) {
        const int64_t start = (int64_t)f * 3200;
        const int64_t n = f == 13 ? 2400 : 3200;
        const std::vector<float> in(clip.begin() + (std::ptrdiff_t)start,
                                    clip.begin() + (std::ptrdiff_t)(start + n));
        std::vector<float> out;
        if (!run_step(*lane, in, 64, &out, &why)) REQUIRE(false);
        got.insert(got.end(), out.begin(), out.end());
    }
    const auto want = g.f32("prompt.ref_codec_mean");
    REQUIRE_EQ(got.size(), (size_t)14 * 64);
    REQUIRE(close_enough("prompt.ref_codec_mean", got, want, kEncMean));
    lane->close();
}

TEST(codec_decodes_every_model_latent) {
    const haru_test::Golden& g = haru_test::Golden::get();
    std::string why;
    auto engine = start_engine(&why);
    REQUIRE(engine != nullptr);
    auto lane = open_lane(*engine, kDecodeCard, &why);
    REQUIRE(lane != nullptr);

    for (int t = 0; t <= 24; ++t) {
        const auto want = g.f32(frame_name(t, "audio_out"));
        std::vector<float> got;
        if (!run_step(*lane, g.f32(frame_name(t, "latent_out")), 3200, &got, &why))
            REQUIRE(false);
        for (float& v : got) v = std::max(-1.0f, std::min(1.0f, v));
        REQUIRE(close_enough(frame_name(t, "audio_out"), got, want, kDecAudio));
    }
    lane->close();
}

TEST(codec_debug_stages) {
    const haru_test::Golden& g = haru_test::Golden::get();
    std::string why;
    auto engine = start_engine(&why);
    REQUIRE(engine != nullptr);
    auto lane = open_lane(*engine, kDebugCard, &why);
    REQUIRE(lane != nullptr);

    const auto audio = g.f32("frame1.user_audio");
    for (int stage = 0; stage < 7; ++stage) {
        const std::string name = "codec_dbg.enc_stage" + std::to_string(stage);
        const auto shp = g.shape(name);
        REQUIRE_EQ(shp.size(), (size_t)3);
        const int64_t C = shp[1], T = shp[2];
        std::vector<float> got;
        if (!run_step(*lane, audio, (size_t)(T * C), &got, &why)) REQUIRE(false);
        const auto flat = g.f32(name);
        std::vector<float> want((size_t)(T * C));
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < C; ++c)
                want[(size_t)(t * C + c)] = flat[(size_t)(c * T + t)];
        if (!close_enough(name, got, want, haru_test::Tol{ kEncAtol[stage], 1e-4 })) {
            std::fprintf(stderr, "    first mismatching stage: %s\n", name.c_str());
            REQUIRE(false);
        }
    }

    const auto latent = g.f32("frame0.latent_out");
    for (int stage = 0; stage < 7; ++stage) {
        const std::string name = "codec_dbg.dec_stage" + std::to_string(stage);
        const auto shp = g.shape(name);
        REQUIRE_EQ(shp.size(), (size_t)3);
        const int64_t C = shp[1], T = shp[2];
        std::vector<float> got;
        if (!run_step(*lane, latent, (size_t)(T * C), &got, &why)) REQUIRE(false);
        const auto flat = g.f32(name);
        std::vector<float> want((size_t)(T * C));
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < C; ++c)
                want[(size_t)(t * C + c)] = flat[(size_t)(c * T + t)];
        if (!close_enough(name, got, want, haru_test::Tol{ kDecAtol[stage], 1e-4 })) {
            std::fprintf(stderr, "    first mismatching stage: %s\n", name.c_str());
            REQUIRE(false);
        }
    }
    lane->close();
}

TEST(codec_samples_with_recorded_noise) {
    const haru_test::Golden& g = haru_test::Golden::get();
    std::string why;
    auto engine = start_engine(&why);
    REQUIRE(engine != nullptr);
    auto lane = open_lane(*engine, kSampleCard, &why);
    REQUIRE(lane != nullptr);

    for (int t = 1; t <= 3; ++t) {
        const auto mean = g.f32(frame_name(t, "user_codec_mean"));
        const auto noise = g.f32(frame_name(t, "user_noise"));
        const auto want = g.f32(frame_name(t, "user_latent"));
        REQUIRE_EQ(mean.size(), (size_t)64);
        REQUIRE_EQ(noise.size(), (size_t)64);
        std::vector<float> in(3200, 0.0f);
        std::copy(mean.begin(), mean.end(), in.begin());
        std::copy(noise.begin(), noise.end(), in.begin() + 64);
        std::vector<float> got;
        if (!run_step(*lane, in, 64, &got, &why)) REQUIRE(false);
        REQUIRE(close_enough(frame_name(t, "user_latent"), got, want, kSampleTol));
    }
    lane->close();
}

TEST(codec_two_lanes_do_not_mix) {
    const haru_test::Golden& g = haru_test::Golden::get();
    std::string why;
    auto engine = start_engine(&why);
    REQUIRE(engine != nullptr);

    /* A is the recorded user audio; B is a different (scaled) audio. Both lanes encode mode 1, so
     * any shared state shows as a disagreement with the single-lane runs. */
    std::vector<std::vector<float>> seq_a, seq_b;
    for (int t = 1; t <= 12; ++t) {
        const auto a = g.f32(frame_name(t, "user_audio"));
        std::vector<float> b = a;
        for (size_t i = 0; i < b.size(); ++i) b[i] = 0.5f * b[i];
        seq_a.push_back(a);
        seq_b.push_back(b);
    }

    auto run_alone = [&](const std::vector<std::vector<float>>& seq,
                         std::vector<std::vector<float>>* out) -> bool {
        std::string w;
        auto lane = open_lane(*engine, kEncodeCard, &w);
        if (!lane) return false;
        for (const auto& in : seq) {
            std::vector<float> o;
            if (!run_step(*lane, in, 64, &o, &w)) return false;
            out->push_back(o);
        }
        lane->close();
        return true;
    };

    std::vector<std::vector<float>> ref_a, ref_b;
    if (!run_alone(seq_a, &ref_a)) REQUIRE(false);
    if (!run_alone(seq_b, &ref_b)) REQUIRE(false);
    REQUIRE_EQ(ref_a.size(), (size_t)12);
    REQUIRE_EQ(ref_b.size(), (size_t)12);

    std::string wa, wb;
    auto a = open_lane(*engine, kEncodeCard, &wa);
    REQUIRE(a != nullptr);
    auto b = open_lane(*engine, kEncodeCard, &wb);
    REQUIRE(b != nullptr);

    for (size_t t = 0; t < seq_a.size(); ++t) {
        std::vector<float> oa, ob;
        if (!step_two(*a, seq_a[t], &oa, &wa, *b, seq_b[t], &ob, &wb)) {
            std::fprintf(stderr, "    concurrent step %zu failed: a=%s b=%s\n", t, wa.c_str(),
                         wb.c_str());
            REQUIRE(false);
        }
        REQUIRE(close_enough("lane A frame " + std::to_string(t + 1), oa, ref_a[t],
                             haru_test::Tol{ 0.0, 0.0 }));
        REQUIRE(close_enough("lane B frame " + std::to_string(t + 1), ob, ref_b[t],
                             haru_test::Tol{ 0.0, 0.0 }));
    }
    a->close();
    b->close();
}

TEST(codec_close_reopen_resets_state) {
    const haru_test::Golden& g = haru_test::Golden::get();
    std::string why;
    auto engine = start_engine(&why);
    REQUIRE(engine != nullptr);

    std::vector<std::vector<float>> first;
    {
        auto lane = open_lane(*engine, kEncodeCard, &why);
        REQUIRE(lane != nullptr);
        for (int t = 1; t <= 6; ++t) {
            std::vector<float> o;
            if (!run_step(*lane, g.f32(frame_name(t, "user_audio")), 64, &o, &why))
                REQUIRE(false);
            first.push_back(o);
        }
        lane->close();
    }

    /* The server releases the lane just after answering close; poll as the harness self-test does. */
    std::unique_ptr<haru_test::Lane> second;
    std::string why2;
    for (int i = 0; i < 300 && !second; ++i) {
        second = haru_test::Lane::open(*engine, kTenant, &why2, kEncodeCard);
        if (second) break;
        if (!has(why2, "session_limit")) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!second) {
        std::fprintf(stderr, "    reopen failed: %s\n", why2.c_str());
        REQUIRE(false);
    }

    for (int t = 1; t <= 6; ++t) {
        std::vector<float> o;
        if (!run_step(*second, g.f32(frame_name(t, "user_audio")), 64, &o, &why2))
            REQUIRE(false);
        REQUIRE(close_enough("reopened frame " + std::to_string(t), o, first[(size_t)(t - 1)],
                             haru_test::Tol{ 0.0, 0.0 }));
    }
    second->close();
}

RAD_TEST_MAIN()
