/* duplex_host_test.cpp -- the production host source (core/server/duplex_host.h) with fakes.
 *
 * No engine, device or plugin: the tests hand it the same DuplexHostPass the driver builds, a
 * fake detokenizer callback and, for the recipe tests, an injected normal stream. The reference
 * lines are in docs/HARU-H6-DESIGN.md sections 3-5. */
#include "rad_test.h"

#include "server/duplex_host.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace rad;

namespace {

using server::HostSampleSource;
using gen::DuplexHostPass;

HostSampleSource::Options options(int64_t latent_dim = 2, int64_t n_vocab = 128,
                                  int64_t max_lanes = 2) {
    HostSampleSource::Options o;
    o.codec_noise_scale = 0.625;   /* .haru-ref/config.py:26: 0.5 / 0.8 */
    o.pad_id = 0;
    o.eop_id = 1;
    o.eos_id = 2;
    o.latent_dim = latent_dim;
    o.n_vocab = n_vocab;
    o.max_lanes = max_lanes;
    o.sampler.temperature = 0.0f;  /* greedy, so a test can force a token */
    o.sampler.top_k = 25;
    o.sampler.top_p = 1.0f;
    o.decode = [](const std::vector<int>&) { return std::string(); };
    return o;
}

DuplexHostPass prompt_pass(const int32_t* lane, int64_t n_tok, int64_t latent_dim,
                           int64_t n_vocab, const int32_t* ref_frame) {
    DuplexHostPass p;
    p.kind = RAD_GEN_PASS_DUPLEX_PROMPT;
    p.n_lane = 1;
    p.lane = lane;
    p.n_tok = n_tok;
    p.latent_dim = latent_dim;
    p.n_vocab = n_vocab;
    p.ref_frame = ref_frame;
    return p;
}

DuplexHostPass step_pass(const int32_t* lane, const int32_t* frame, int64_t n_lane,
                         int64_t latent_dim, int64_t n_vocab, const float* logits,
                         const float* pad_eop) {
    DuplexHostPass p;
    p.kind = RAD_GEN_PASS_DUPLEX_STEP;
    p.n_lane = n_lane;
    p.lane = lane;
    p.frame = frame;
    p.latent_dim = latent_dim;
    p.n_vocab = n_vocab;
    p.logits = logits;
    p.pad_eop = pad_eop;
    return p;
}

/* The one-scale-for-the-clip recipe: scale * (latent_std / 0.8) * raw, reference frames in
 * frame-major order, then frame 0's head. */
TEST(reference_noise_is_one_scale_times_the_combined_term) {
    HostSampleSource::Options o = options(/*latent_dim=*/2, /*n_vocab=*/8, /*max_lanes=*/1);
    std::vector<double> draws = { 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    size_t at = 0;
    o.normal = [&] { return at < draws.size() ? draws[at++] : 0.0; };
    HostSampleSource src(std::move(o));
    src.lane_opened(0, 1234);

    const int32_t lanes[1] = { 0 };
    const int32_t ref_frame[4] = { -1, 0, 1, -1 };
    DuplexHostPass p = prompt_pass(lanes, 4, 2, 8, ref_frame);
    std::vector<float> codec(4 * 2, 0.0f), head(2, 0.0f);
    std::string why;
    CHECK_EQ(src.noise(p, codec.data(), head.data(), &why), RAD_OK);

    CHECK_EQ(at, (size_t)7);
    /* Card rows: no codec noise. */
    CHECK_EQ(codec[0], 0.0f);
    CHECK_EQ(codec[1], 0.0f);
    CHECK_EQ(codec[6], 0.0f);
    CHECK_EQ(codec[7], 0.0f);
    /* Frame 0: scale 2, raw [1, 2]. Frame 1: raw [3, 4]. Scale factor 2 * 0.625. */
    CHECK_EQ(codec[2], 1.25f);
    CHECK_EQ(codec[3], 2.5f);
    CHECK_EQ(codec[4], 3.75f);
    CHECK_EQ(codec[5], 5.0f);
    /* Head: raw [5, 6], no scale. */
    CHECK_EQ(head[0], 6.25f);
    CHECK_EQ(head[1], 7.5f);
}

/* A user frame and its head: scale + a full normal vector, then the head's normal vector. */
TEST(step_noise_is_entry_indexed_and_advances_each_lane) {
    HostSampleSource::Options o = options(/*latent_dim=*/2, /*n_vocab=*/8, /*max_lanes=*/2);
    std::vector<double> draws = {
        2.0, 1.0, 2.0, 3.0, 4.0,      /* entry 0: scale, user, head */
        5.0, 6.0, 7.0, 8.0, 9.0,      /* entry 1 */
    };
    size_t at = 0;
    o.normal = [&] { return at < draws.size() ? draws[at++] : 0.0; };
    HostSampleSource src(std::move(o));
    src.lane_opened(0, 10);
    src.lane_opened(1, 20);

    const int32_t lanes[2] = { 1, 0 };
    const int32_t frames[2] = { 0, 0 };
    DuplexHostPass p = step_pass(lanes, frames, 2, 2, 8, nullptr, nullptr);
    std::vector<float> codec(2 * 2, 0.0f), head(2 * 2, 0.0f);
    std::string why;
    CHECK_EQ(src.noise(p, codec.data(), head.data(), &why), RAD_OK);
    CHECK_EQ(at, (size_t)10);
    CHECK_EQ(codec[0], 1.25f);   /* lane 1, entry 0: 2 * 0.625 * 1 */
    CHECK_EQ(codec[1], 2.5f);
    CHECK_EQ(codec[2], 18.75f);  /* lane 0, entry 1: 5 * 0.625 * 6 */
    CHECK_EQ(codec[3], 21.875f);
    CHECK_EQ(head[0], 3.75f);
    CHECK_EQ(head[1], 5.0f);
    CHECK_EQ(head[2], 10.0f);
    CHECK_EQ(head[3], 11.25f);

    /* The next frame advances both lanes; a frame that skips ahead is refused. */
    const int32_t next_frames[2] = { 1, 1 };
    DuplexHostPass next = step_pass(lanes, next_frames, 2, 2, 8, nullptr, nullptr);
    CHECK_EQ(src.noise(next, codec.data(), head.data(), &why), RAD_OK);
    const int32_t skip_frames[2] = { 3, 3 };
    DuplexHostPass skip = step_pass(lanes, skip_frames, 2, 2, 8, nullptr, nullptr);
    CHECK(src.noise(skip, codec.data(), head.data(), &why) < 0);
}

/* Same seed, same bytes; different seed, different bytes. The streams are per lane, so another
 * lane's lifetime or order does not move lane 0's draws. */
TEST(noise_is_deterministic_per_lane_and_seed) {
    const int32_t lanes0[1] = { 0 };
    const int32_t ref0[2] = { 0, 1 };
    auto prompt = [&](HostSampleSource& src, float* codec, float* head) {
        DuplexHostPass p = prompt_pass(lanes0, 2, 2, 8, ref0);
        std::string why;
        return src.noise(p, codec, head, &why);
    };

    std::vector<float> a(4), b(4), ha(2), hb(2), c(4), hc(2);
    HostSampleSource src_a(options(2, 8, 2));
    src_a.lane_opened(0, 777);
    CHECK_EQ(prompt(src_a, a.data(), ha.data()), RAD_OK);

    HostSampleSource src_b(options(2, 8, 2));
    src_b.lane_opened(1, 9999);   /* another lane first: lane 0's stream is unchanged */
    src_b.lane_opened(0, 777);
    CHECK_EQ(prompt(src_b, b.data(), hb.data()), RAD_OK);
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(ha.data(), hb.data(), ha.size() * sizeof(float)) == 0);

    HostSampleSource src_c(options(2, 8, 2));
    src_c.lane_opened(0, 778);
    CHECK_EQ(prompt(src_c, c.data(), hc.data()), RAD_OK);
    CHECK(std::memcmp(a.data(), c.data(), a.size() * sizeof(float)) != 0);

    /* Close and reopen with the same seed restarts the stream. */
    src_a.lane_closed(0);
    src_a.lane_opened(0, 777);
    std::vector<float> d(4), hd(2);
    CHECK_EQ(prompt(src_a, d.data(), hd.data()), RAD_OK);
    CHECK(std::memcmp(a.data(), d.data(), a.size() * sizeof(float)) == 0);
}

/* The sampled token comes from the lane's row, not the entry's, and the detokenizer's delta is
 * returned once: PAD/EOP runs collapse, 65/66 spell A/B, and a two-id UTF-8 character is held
 * until its second id. */
TEST(tokens_are_lane_indexed_and_text_is_streamed_once) {
    HostSampleSource::Options o = options(/*latent_dim=*/64, /*n_vocab=*/128, /*max_lanes=*/2);
    o.decode = [](const std::vector<int>& ids) {
        if (ids.size() == 1 && ids[0] == 65) return std::string("A");
        if (ids.size() == 1 && ids[0] == 66) return std::string("B");
        if (ids.size() == 1 && ids[0] == 70) return std::string("\xEF\xBF\xBD");
        if (ids.size() == 2 && ids[0] == 70 && ids[1] == 71)
            return std::string("\xE2\x82\xAC");
        return std::string();
    };
    HostSampleSource src(std::move(o));
    src.lane_opened(0, 1);

    std::string why;
    std::vector<float> logits(128, -1e9f), pad_eop(2, -1e9f);
    const int32_t lanes[1] = { 0 };
    const int32_t ref_frame[1] = { -1 };
    DuplexHostPass p = prompt_pass(lanes, 1, 64, 128, ref_frame);
    pad_eop[0] = 1e9f;  /* force PAD on the (dropped) frame 0 */
    p.logits = logits.data();
    p.pad_eop = pad_eop.data();
    int32_t token = -1;
    CHECK_EQ(src.tokens(p, &token, &why), RAD_OK);
    CHECK_EQ(token, 0);

    /* Helper: force `id` with greedy sampling. */
    auto force = [&](int32_t id) {
        std::fill(logits.begin(), logits.end(), -1e9f);
        pad_eop[0] = pad_eop[1] = -1e9f;
        if (id == 0) pad_eop[0] = 1e9f;
        else if (id == 1) pad_eop[1] = 1e9f;
        else logits[(size_t)id] = 1e9f;
    };
    const int32_t frames[1] = { 0 };
    auto step = [&](int32_t id, int32_t frame, int32_t expect) {
        force(id);
        DuplexHostPass s = step_pass(lanes, &frame, 1, 64, 128, logits.data(), pad_eop.data());
        int32_t got = -1;
        CHECK_EQ(src.tokens(s, &got, &why), RAD_OK);
        CHECK_EQ(got, expect);
        std::string text;
        CHECK_EQ(src.reply_text(0, &text, &why), RAD_OK);
        return text;
    };

    CHECK_EQ(step(1, frames[0], 1), std::string(""));     /* EOP after PAD: still in the gap */
    CHECK_EQ(step(0, 1, 0), std::string(""));             /* PAD in the same gap */
    CHECK_EQ(step(65, 2, 65), std::string("A"));
    CHECK_EQ(step(66, 3, 66), std::string("B"));
    CHECK_EQ(step(70, 4, 70), std::string(""));           /* held partial character */
    CHECK_EQ(step(71, 5, 71), std::string("\xE2\x82\xAC"));
    std::string again;
    CHECK_EQ(src.reply_text(0, &again, &why), RAD_OK);
    CHECK_EQ(again, std::string(""));                     /* a delta is returned only once */

    /* Entry order [1, 0] with different rows: the token follows the lane. */
    src.lane_opened(1, 2);
    src.lane_opened(0, 3);
    std::vector<float> rows(2 * 128, -1e9f), pe(4, -1e9f);
    rows[0 * 128 + 3] = 1e9f;   /* lane 0 -> token 3 */
    rows[1 * 128 + 5] = 1e9f;   /* lane 1 -> token 5 */
    const int32_t two_lanes[2] = { 1, 0 };
    const int32_t two_frames[2] = { 0, 0 };
    DuplexHostPass s2 = step_pass(two_lanes, two_frames, 2, 64, 128, rows.data(), pe.data());
    int32_t out[2] = { -1, -1 };
    CHECK_EQ(src.tokens(s2, out, &why), RAD_OK);
    CHECK_EQ(out[0], 5);
    CHECK_EQ(out[1], 3);
}

/* A missing lane state is refused rather than silently sampled from another lane. */
TEST(a_missing_lane_is_refused) {
    HostSampleSource src(options());
    const int32_t lanes[1] = { 0 };
    const int32_t ref_frame[1] = { -1 };
    DuplexHostPass p = prompt_pass(lanes, 1, 2, 8, ref_frame);
    std::vector<float> codec(2), head(2);
    std::string why;
    CHECK_EQ(src.noise(p, codec.data(), head.data(), &why), RAD_E_STATE);
    CHECK(!why.empty());
}

}  /* namespace */

RAD_TEST_MAIN()
