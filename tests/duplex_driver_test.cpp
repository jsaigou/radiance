/* duplex_driver_test.cpp -- the duplex tenant's pass driver (core/gen/duplex_driver.h) on the host
 * backend. The plugin's side of each pass is played here, as audio_gen_test plays the audio
 * generator's: the test writes the STEP pass's audio_out/text_out/status where the driver reads
 * them back. The deterministic fake is
 *
 *     audio_out[lane] = audio_in[lane] * 2
 *     text_out[lane]  = lane * 1000 + frame
 *     status[lane]    = 0
 *
 * so the driver's staging and lane bookkeeping are visible in the replies.
 */
#include "sched_fakes.h"

using namespace sched_fake;

#include "gen/duplex_driver.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace rad;

namespace {

constexpr int64_t kSamples = 4;

struct Fixture {
    Program prog;
    FakeKV  kv;
    gen::DuplexDriver drv;
    RadDuplexDecl d{};
    std::vector<float>   audio_in, audio_out;
    std::vector<int32_t> text_out, status;
    int64_t max_ctx = 0;
    int64_t max_lanes = 0;
    int step = 0;

    explicit Fixture(int64_t ctx = 64, int64_t lanes = 2, int64_t max_prompt = 16)
        : max_ctx(ctx), max_lanes(lanes) {
        d.name = "test";
        d.tmpl = "<card>{card}</card>{references}";
        d.max_prompt = max_prompt;
        d.max_ref_frames = 2;
        d.ref_rate = 50;
        d.ref_hop = kSamples;
        d.max_lanes = lanes;
        d.sample_rate = 50;
        d.channels = 1;
        d.samples_per_frame = kSamples;
        d.latent_dim = 2;
        d.frame_rate_num = 25;
        d.frame_rate_den = 2;
        d.n_audio_in = 2;
        d.audio_ins[0] = RadDuplexAudioIn{ RAD_DUPLEX_AUDIO_USER, 2, 8, 8, 1 };
        d.audio_ins[1] = RadDuplexAudioIn{ RAD_DUPLEX_AUDIO_MODEL, 2, 8, 8, 1 };
        d.text = 1;

        audio_in.assign((size_t)(lanes * kSamples), 0.0f);
        audio_out.assign((size_t)(lanes * kSamples), -1.0f);
        text_out.assign((size_t)lanes, -2);
        status.assign((size_t)lanes, -2);
        prog.buffers.resize(5);
        buffer(1, "audio_in", RAD_F32, lanes, kSamples, audio_in.data());
        buffer(2, "audio_out", RAD_F32, lanes, kSamples, audio_out.data());
        buffer(3, "text_out", RAD_I32, lanes, 1, text_out.data());
        buffer(4, "status", RAD_I32, lanes, 1, status.data());
        d.audio_in = 1;
        d.audio_out = 2;
        d.text_out = 3;
        d.status = 4;
        prog.duplex = d;
        prog.dup_str.name = "test";
        prog.ctx.max_tok = 64;
        prog.ctx.max_seqs = 8;
        prog.ctx.max_ctx = ctx;

        prog.kv_groups.emplace_back();
        KVGroupInfo g;
        g.name = "lm";
        g.decl.kind = RAD_KV_FULL;
        g.block_size = 16;
        prog.kv_groups.push_back(g);
        kv.setup(prog.kv_groups, ctx, 128, 4);
    }

    void buffer(int h, const char* name, uint32_t dtype, int64_t rows, int64_t cols, void* ptr) {
        BufferInfo& b = prog.buffers[(size_t)h];
        b.name = name;
        b.decl.dtype = dtype;
        b.decl.rank = cols > 1 ? 2 : 1;
        b.decl.shape[0] = rows;
        b.decl.shape[1] = cols;
        b.decl.kind = RAD_BUF_PERSIST;
        b.decl.domain = RAD_DOMAIN_DEVICE;
        b.bytes = rows * (cols > 1 ? cols : 1) * 4;
        b.ptr = ptr;
    }

    int init() {
        return drv.init(prog, &kv, { gen::DuplexDriver::RankIO{ 0, nullptr, &prog } }, max_ctx);
    }

    int open(const std::vector<int32_t>& tokens, std::shared_ptr<gen::DuplexTicket>* t,
             std::string* why, std::vector<float> ref_audio = {},
             std::vector<int32_t> ref_cu = {}, std::vector<int32_t> ref_frame = {}) {
        auto r = std::make_unique<gen::DuplexRequest>();
        r->tokens = tokens;
        r->ref_audio = std::move(ref_audio);
        r->ref_cu = std::move(ref_cu);
        r->ref_frame = std::move(ref_frame);
        r->seed = 9;
        return drv.open(std::move(r), t, why);
    }

    /* The prompt pass for one lane, then done(); returns the staged batch for inspection. */
    const RadBatch* run_prompt(const std::shared_ptr<gen::DuplexTicket>& t) {
        const RadBatch* b = drv.next(step++);
        CHECK_OK(drv.done());
        bool to = false;
        CHECK(t->wait(0, &to));
        CHECK_EQ(t->status(), (int)RAD_OK);
        return b;
    }

    int queue_step(int32_t lane, uint64_t seq, float base, std::shared_ptr<gen::DuplexTicket>* t,
                   std::string* why) {
        std::vector<float> samples((size_t)d.samples_per_frame);
        for (int64_t i = 0; i < d.samples_per_frame; ++i) samples[(size_t)i] = base + (float)i;
        return drv.step(lane, samples.data(), (int64_t)samples.size(), seq, t, why);
    }

    /* The fake plugin's STEP: read what the driver uploaded into the declared audio_in buffer,
     * write the declared outputs. */
    void answer_step(const RadBatch* b) {
        const RadDuplexBatch* D = b->dup_b;
        CHECK(D != nullptr);
        if (!D) return;
        const float* in = static_cast<const float*>(prog.buffers[d.audio_in].ptr);
        float* out = nullptr;
        int32_t* text = nullptr;
        int32_t* stat = nullptr;
        for (const gen::IPassDriver::Readback& rb : drv.readbacks()) {
            if (rb.buf == d.audio_out) out = static_cast<float*>(rb.host);
            else if (rb.buf == d.text_out) text = static_cast<int32_t*>(rb.host);
            else if (rb.buf == d.status) stat = static_cast<int32_t*>(rb.host);
        }
        CHECK(out != nullptr && text != nullptr && stat != nullptr);
        if (!out || !text || !stat) return;
        for (int64_t i = 0; i < D->n_lane; ++i) {
            const int32_t lane = D->lane_h[(size_t)i];
            const int32_t frame = D->frame_h[(size_t)i];
            for (int64_t s = 0; s < d.samples_per_frame; ++s) {
                const size_t k = (size_t)lane * (size_t)d.samples_per_frame + (size_t)s;
                out[k] = in[k] * 2.0f;
            }
            text[lane] = lane * 1000 + frame;
            stat[lane] = 0;
        }
    }
};

}  /* namespace */

/* A card-only prompt: the card rows, their positions, the lane's last row, no references. */
TEST(open_stages_a_card_only_prompt) {
    Fixture f;
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> t;
    std::string why;
    REQUIRE_EQ(f.open({ 7, 8, 9 }, &t, &why), RAD_OK);
    const RadBatch* b = f.run_prompt(t);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->gen, (int32_t)RAD_GEN_PASS_DUPLEX_PROMPT);
    CHECK_EQ(b->n_tok, (int64_t)3);
    CHECK_EQ(b->n_seq, (int64_t)1);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(b->token_ids[i], 7 + i);
        CHECK_EQ(b->positions[i], i);
    }
    CHECK_EQ(b->cu_seqlens[0], 0);
    CHECK_EQ(b->cu_seqlens[1], 3);
    const RadDuplexBatch* D = b->dup_b;
    REQUIRE(D != nullptr);
    CHECK_EQ(D->n_lane, (int64_t)1);
    CHECK_EQ(D->n_tok, (int64_t)3);
    CHECK_EQ(D->lane_h[0], 0);
    CHECK_EQ(D->last_row_h[0], 2);
    CHECK_EQ(D->n_ref, (int64_t)0);
    CHECK(f.drv.next(f.step++) == nullptr);
}

/* Card rows then reference rows: the wiring the plugin embeds through. */
TEST(open_stages_reference_rows) {
    Fixture f;
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> t;
    std::string why;
    const std::vector<float> clips = { 1, 2, 3, 4, 5, 6, 7, 8 };
    REQUIRE_EQ(f.open({ 10, 11, 12, 0, 0 }, &t, &why, clips, { 0, 4, 8 },
                      { -1, -1, -1, 0, 1 }),
               RAD_OK);
    const RadBatch* b = f.run_prompt(t);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_tok, (int64_t)5);
    CHECK_EQ(b->token_ids[0], 10);
    CHECK_EQ(b->token_ids[3], 0);
    CHECK_EQ(b->token_ids[4], 0);
    const RadDuplexBatch* D = b->dup_b;
    REQUIRE(D != nullptr);
    CHECK_EQ(D->n_ref, (int64_t)2);
    CHECK_EQ(D->ref_samples, (int64_t)8);
    for (int i = 0; i < 8; ++i) CHECK_EQ(D->ref_audio[i], clips[(size_t)i]);
    CHECK_EQ(D->ref_cu[0], 0);
    CHECK_EQ(D->ref_cu[1], 4);
    CHECK_EQ(D->ref_cu[2], 8);
    REQUIRE(D->ref_frame != nullptr);
    CHECK_EQ(D->ref_frame[0], -1);
    CHECK_EQ(D->ref_frame[3], 0);
    CHECK_EQ(D->ref_frame[4], 1);
    CHECK_EQ(D->last_row_h[0], 4);
}

/* One lane's steps: one row each at its next position, one reply per input in order, with the
 * model's frame and token. */
TEST(a_step_returns_the_lanes_model_frame) {
    Fixture f;
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> open;
    std::string why;
    REQUIRE_EQ(f.open({ 1, 2 }, &open, &why), RAD_OK);
    const RadBatch* b = f.run_prompt(open);
    REQUIRE(b != nullptr);
    const int32_t lane = b->dup_b->lane_h[0];

    for (int32_t frame = 0; frame < 2; ++frame) {
        std::shared_ptr<gen::DuplexTicket> t;
        REQUIRE_EQ(f.queue_step(lane, 100 + frame, 10.0f + 10.0f * frame, &t, &why), RAD_OK);
        b = f.drv.next(f.step++);
        REQUIRE(b != nullptr);
        CHECK_EQ(b->gen, (int32_t)RAD_GEN_PASS_DUPLEX_STEP);
        CHECK_EQ(b->n_tok, (int64_t)1);
        CHECK_EQ(b->n_seq, (int64_t)1);
        CHECK_EQ(b->positions[0], 2 + frame);
        CHECK_EQ(b->cu_seqlens[0], 0);
        CHECK_EQ(b->cu_seqlens[1], 1);
        const RadDuplexBatch* D = b->dup_b;
        REQUIRE(D != nullptr);
        CHECK_EQ(D->n_lane, (int64_t)1);
        CHECK_EQ(D->lane_h[0], lane);
        CHECK_EQ(D->frame_h[0], frame);
        CHECK_EQ(D->flags_h[0], frame == 0 ? RAD_DUPLEX_F_FIRST : 0);

        f.answer_step(b);
        CHECK_OK(f.drv.done());
        gen::DuplexReply r;
        bool to = false;
        REQUIRE(t->next(&r, 0, &to));
        CHECK_EQ(r.seq, (uint64_t)(100 + frame));
        CHECK_EQ(r.status, 0);
        CHECK(r.has_audio);
        CHECK_EQ(r.token, lane * 1000 + frame);
        std::vector<float> want((size_t)f.d.samples_per_frame);
        for (int64_t i = 0; i < f.d.samples_per_frame; ++i)
            want[(size_t)i] = (10.0f + 10.0f * frame + (float)i) * 2.0f;
        REQUIRE_EQ(r.audio.size(), want.size());
        CHECK(std::memcmp(r.audio.data(), want.data(), want.size() * 4) == 0);
        CHECK(t->finished());
    }
}

/* Two lanes with inputs waiting go in one STEP pass, each with its own row, position and frame;
 * the replies stay with their lanes. */
TEST(two_lanes_step_together_keep_their_own_data) {
    Fixture f;
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> o0, o1;
    std::string why;
    REQUIRE_EQ(f.open({ 1 }, &o0, &why), RAD_OK);
    const RadBatch* b = f.run_prompt(o0);
    REQUIRE(b != nullptr);
    const int32_t lane0 = b->dup_b->lane_h[0];
    REQUIRE_EQ(f.open({ 2, 3 }, &o1, &why), RAD_OK);
    b = f.run_prompt(o1);
    REQUIRE(b != nullptr);
    const int32_t lane1 = b->dup_b->lane_h[0];
    CHECK(lane0 != lane1);

    /* Lane 0 takes its first live step alone, so the pair below have different frame numbers
     * and positions. */
    std::shared_ptr<gen::DuplexTicket> first;
    REQUIRE_EQ(f.queue_step(lane0, 10, 50.0f, &first, &why), RAD_OK);
    b = f.drv.next(f.step++);
    REQUIRE(b != nullptr);
    f.answer_step(b);
    CHECK_OK(f.drv.done());
    gen::DuplexReply rfirst;
    bool tof = false;
    REQUIRE(first->next(&rfirst, 0, &tof));
    CHECK_EQ(rfirst.token, lane0 * 1000);

    std::shared_ptr<gen::DuplexTicket> t0, t1;
    REQUIRE_EQ(f.queue_step(lane0, 11, 100.0f, &t0, &why), RAD_OK);
    REQUIRE_EQ(f.queue_step(lane1, 22, 200.0f, &t1, &why), RAD_OK);
    b = f.drv.next(f.step++);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->gen, (int32_t)RAD_GEN_PASS_DUPLEX_STEP);
    CHECK_EQ(b->n_tok, (int64_t)2);
    CHECK_EQ(b->n_seq, (int64_t)2);
    CHECK_EQ(b->cu_seqlens[0], 0);
    CHECK_EQ(b->cu_seqlens[1], 1);
    CHECK_EQ(b->cu_seqlens[2], 2);
    CHECK_EQ(b->positions[0], 2);                  /* lane 0 after its prompt and first step */
    CHECK_EQ(b->positions[1], 2);                  /* lane 1 after its two prompt rows */
    const RadDuplexBatch* D = b->dup_b;
    REQUIRE(D != nullptr);
    REQUIRE_EQ(D->n_lane, (int64_t)2);
    CHECK(D->lane_h[0] == lane0 && D->lane_h[1] == lane1);
    CHECK_EQ(D->frame_h[0], 1);
    CHECK_EQ(D->frame_h[1], 0);
    CHECK_EQ(D->flags_h[0], 0);
    CHECK_EQ(D->flags_h[1], RAD_DUPLEX_F_FIRST);

    f.answer_step(b);
    CHECK_OK(f.drv.done());
    gen::DuplexReply r0, r1;
    bool to0 = false, to1 = false;
    REQUIRE(t0->next(&r0, 0, &to0));
    REQUIRE(t1->next(&r1, 0, &to1));
    CHECK_EQ(r0.seq, (uint64_t)11);
    CHECK_EQ(r1.seq, (uint64_t)22);
    CHECK_EQ(r0.token, lane0 * 1000 + 1);
    CHECK_EQ(r1.token, lane1 * 1000);
    for (int64_t i = 0; i < f.d.samples_per_frame; ++i) {
        CHECK_EQ(r0.audio[(size_t)i], (100.0f + (float)i) * 2.0f);
        CHECK_EQ(r1.audio[(size_t)i], (200.0f + (float)i) * 2.0f);
    }
}

/* flush is step-shaped with no frame in this stage. */
TEST(flush_returns_an_empty_reply) {
    Fixture f;
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> open;
    std::string why;
    REQUIRE_EQ(f.open({ 1 }, &open, &why), RAD_OK);
    const RadBatch* b = f.run_prompt(open);
    REQUIRE(b != nullptr);
    const int32_t lane = b->dup_b->lane_h[0];
    std::shared_ptr<gen::DuplexTicket> t;
    REQUIRE_EQ(f.drv.flush(lane, 55, &t, &why), RAD_OK);
    gen::DuplexReply r;
    bool to = false;
    REQUIRE(t->next(&r, 0, &to));
    CHECK_EQ(r.seq, (uint64_t)55);
    CHECK_EQ(r.status, 0);
    CHECK(!r.has_audio);
    CHECK_EQ(r.token, -1);
}

/* The lane limit refuses an open, and close returns the lane and its KV. */
TEST(a_lane_limit_refuses_an_open_then_accepts_after_close) {
    Fixture f(64, 1);
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> t0;
    std::string why;
    REQUIRE_EQ(f.open({ 1 }, &t0, &why), RAD_OK);
    const RadBatch* b = f.run_prompt(t0);
    REQUIRE(b != nullptr);
    const int32_t lane0 = b->dup_b->lane_h[0];

    std::shared_ptr<gen::DuplexTicket> t1;
    CHECK(f.open({ 2 }, &t1, &why) < 0);
    CHECK(has(why, "lanes") || has(why, "close"));
    bool to = false;
    CHECK(t1->wait(0, &to));
    CHECK(t1->status() < 0);

    CHECK_EQ((int)f.kv.seqs.size(), 1);
    CHECK_EQ(f.drv.close(lane0), RAD_OK);
    CHECK_EQ((int)f.kv.seqs.size(), 0);
    CHECK_EQ(f.drv.close(lane0), RAD_OK);           /* twice */

    std::shared_ptr<gen::DuplexTicket> t2;
    REQUIRE_EQ(f.open({ 2 }, &t2, &why), RAD_OK);
    b = f.run_prompt(t2);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->dup_b->lane_h[0], lane0);
}

/* close while a STEP is staged but not finished fails the reply ticket, and the lane is released
 * once the pass is folded back. */
TEST(close_mid_pass_fails_the_ticket_and_does_not_hang) {
    Fixture f(64, 1);
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> open;
    std::string why;
    REQUIRE_EQ(f.open({ 1 }, &open, &why), RAD_OK);
    const RadBatch* b = f.run_prompt(open);
    REQUIRE(b != nullptr);
    const int32_t lane = b->dup_b->lane_h[0];

    std::shared_ptr<gen::DuplexTicket> t;
    REQUIRE_EQ(f.queue_step(lane, 77, 1.0f, &t, &why), RAD_OK);
    b = f.drv.next(f.step++);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->gen, (int32_t)RAD_GEN_PASS_DUPLEX_STEP);

    CHECK_EQ(f.drv.close(lane), RAD_OK);
    bool to = false;
    gen::DuplexReply no_reply;
    CHECK(!t->next(&no_reply, 0, &to));             /* no reply: the lane was closed */
    CHECK_EQ(t->status(), (int)RAD_E_CANCELLED);
    CHECK(has(t->why(), "closed"));
    CHECK_OK(f.drv.done());                         /* folding the pass must not reply or crash */
    CHECK_EQ(f.drv.close(lane), RAD_OK);

    std::shared_ptr<gen::DuplexTicket> again;
    REQUIRE_EQ(f.open({ 3 }, &again, &why), RAD_OK);
    b = f.run_prompt(again);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->dup_b->lane_h[0], lane);
}

/* shutdown fails an unready open and a queued step, and refuses later opens. */
TEST(shutdown_fails_everything) {
    Fixture f(64, 2);
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> open0, open1, step_t;
    std::string why;
    REQUIRE_EQ(f.open({ 1 }, &open0, &why), RAD_OK);
    const RadBatch* b = f.run_prompt(open0);
    REQUIRE(b != nullptr);
    const int32_t lane0 = b->dup_b->lane_h[0];
    REQUIRE_EQ(f.queue_step(lane0, 5, 3.0f, &step_t, &why), RAD_OK);
    REQUIRE_EQ(f.open({ 2 }, &open1, &why), RAD_OK);   /* not ready */

    f.drv.shutdown("the engine stopped", RAD_E_CANCELLED);
    bool to = false;
    CHECK(open1->wait(0, &to));
    CHECK_EQ(open1->status(), (int)RAD_E_CANCELLED);
    CHECK(has(open1->why(), "engine stopped"));
    gen::DuplexReply no_reply;
    CHECK(!step_t->next(&no_reply, 0, &to));
    CHECK_EQ(step_t->status(), (int)RAD_E_CANCELLED);

    std::shared_ptr<gen::DuplexTicket> late;
    CHECK(f.open({ 3 }, &late, &why) < 0);
    CHECK(late->wait(0, &to));
    CHECK(has(late->why(), "engine stopped"));
    CHECK(f.drv.next(f.step++) == nullptr);
}

/* A frame of the wrong length is refused before a pass. */
TEST(a_wrong_length_frame_is_refused) {
    Fixture f;
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> open;
    std::string why;
    REQUIRE_EQ(f.open({ 1 }, &open, &why), RAD_OK);
    const RadBatch* b = f.run_prompt(open);
    REQUIRE(b != nullptr);
    const int32_t lane = b->dup_b->lane_h[0];
    const float samples[3] = { 1, 2, 3 };
    std::shared_ptr<gen::DuplexTicket> t;
    CHECK(f.drv.step(lane, samples, 3, 9, &t, &why) < 0);
    CHECK(has(why, "samples"));
    bool to = false;
    CHECK(t->wait(0, &to));
    CHECK_EQ(t->status(), (int)RAD_E_INVAL);
    CHECK(f.drv.next(f.step++) == nullptr);
}

/* A step before the prompt pass is refused. */
TEST(step_before_ready_is_refused) {
    Fixture f;
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> open;
    std::string why;
    REQUIRE_EQ(f.open({ 1 }, &open, &why), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> t;
    CHECK(f.queue_step(0, 9, 1.0f, &t, &why) < 0);
    CHECK(has(why, "not ready"));
    bool to = false;
    CHECK(t->wait(0, &to));
    CHECK_EQ(t->status(), (int)RAD_E_STATE);
}

/* The context bound is the prompt plus the live rows the driver was given; the step past it is
 * refused. */
TEST(the_context_bound_refuses_a_step_past_it) {
    Fixture f(4, 1, 3);
    REQUIRE_EQ(f.init(), RAD_OK);
    std::shared_ptr<gen::DuplexTicket> open;
    std::string why;
    REQUIRE_EQ(f.open({ 1, 2, 3 }, &open, &why), RAD_OK);
    const RadBatch* b = f.run_prompt(open);
    REQUIRE(b != nullptr);
    const int32_t lane = b->dup_b->lane_h[0];

    std::shared_ptr<gen::DuplexTicket> t0;
    REQUIRE_EQ(f.queue_step(lane, 1, 1.0f, &t0, &why), RAD_OK);
    b = f.drv.next(f.step++);
    REQUIRE(b != nullptr);
    f.answer_step(b);
    CHECK_OK(f.drv.done());
    gen::DuplexReply r;
    bool to = false;
    REQUIRE(t0->next(&r, 0, &to));                  /* prompt rows 0..2, this row 3 */

    std::shared_ptr<gen::DuplexTicket> t1;
    CHECK(f.queue_step(lane, 2, 2.0f, &t1, &why) < 0);
    CHECK(has(why, "context"));
    CHECK(t1->wait(0, &to));
    CHECK_EQ(t1->status(), (int)RAD_E_FULL);
}

RAD_TEST_MAIN()
