/* tests/duplex_engine_bridge_test.cpp -- the blocking engine bridge's deadlines. The real
 * DuplexDriver runs against sched_fakes' KV manager and device layer; the test withholds the
 * engine pass so the bridge must time out and close the lane instead of holding a session.
 */
#include "sched_fakes.h"

using namespace sched_fake;

#include "gen/duplex_driver.h"
#include "server/duplex_engine.h"

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace rad;

namespace {

constexpr int64_t kSamples = 4;

struct Fixture {
    Program prog;
    FakeKV  kv;
    gen::DuplexDriver drv;
    RadDuplexDecl d{};
    std::vector<float> audio_in, audio_out;
    std::vector<int32_t> text_out, status;
    int step = 0;

    Fixture() {
        d.name = "test";
        d.tmpl = "<c>{card}</c>{references}";
        d.max_prompt = 16;
        d.max_ref_frames = 2;
        d.ref_rate = 50;
        d.ref_hop = kSamples;
        d.max_lanes = 1;
        d.sample_rate = 50;
        d.channels = 1;
        d.samples_per_frame = kSamples;
        d.latent_dim = 2;
        d.frame_rate_num = 25;
        d.frame_rate_den = 2;
        d.n_audio_in = 0;
        d.text = 1;
        audio_in.assign((size_t)kSamples, 0.0f);
        audio_out.assign((size_t)kSamples, 0.0f);
        text_out.assign(1, -1);
        status.assign(1, 0);
        prog.buffers.resize(5);
        buffer(1, "audio_in", RAD_F32, kSamples, audio_in.data());
        buffer(2, "audio_out", RAD_F32, kSamples, audio_out.data());
        buffer(3, "text_out", RAD_I32, 1, text_out.data());
        buffer(4, "status", RAD_I32, 1, status.data());
        d.audio_in = 1;
        d.audio_out = 2;
        d.text_out = 3;
        d.status = 4;
        prog.duplex = d;
        prog.dup_str.name = "test";
        prog.ctx.max_tok = 64;
        prog.ctx.max_seqs = 8;
        prog.ctx.max_ctx = 64;
        prog.kv_groups.emplace_back();
        KVGroupInfo g;
        g.name = "lm";
        g.decl.kind = RAD_KV_FULL;
        g.block_size = 16;
        prog.kv_groups.push_back(g);
        kv.setup(prog.kv_groups, 64, 128, 4);
    }

    void buffer(int h, const char* name, uint32_t dtype, int64_t rows, void* ptr) {
        BufferInfo& b = prog.buffers[(size_t)h];
        b.name = name;
        b.decl.dtype = dtype;
        b.decl.rank = 1;
        b.decl.shape[0] = rows;
        b.decl.kind = RAD_BUF_PERSIST;
        b.decl.domain = RAD_DOMAIN_DEVICE;
        b.bytes = rows * 4;
        b.ptr = ptr;
    }

    int init() {
        return drv.init(prog, &kv, { gen::DuplexDriver::RankIO{ 0, nullptr, &prog } }, 64);
    }
};

server::DuplexGenInfo info() {
    server::DuplexGenInfo i;
    i.name = "test";
    i.tmpl = "<c>{card}</c>{references}";
    i.text = true;
    i.max_lanes = 1;
    i.max_prompt = 16;
    i.max_ref_frames = 2;
    i.ref_rate = 50;
    i.ref_hop = kSamples;
    i.sample_rate = 50;
    i.samples_per_frame = kSamples;
    return i;
}

}  /* namespace */

/* A step whose pass never runs closes the lane and returns the timeout marker; the lane cannot be
 * stepped again. */
TEST(a_step_that_never_runs_times_out_and_closes_the_lane) {
    Fixture f;
    REQUIRE_EQ(f.init(), RAD_OK);
    server::EngineDuplexBridge bridge;
    bridge.init(&f.drv, info(), server::EngineDuplexBridge::Timeouts{ 1000, 50 });

    std::unique_ptr<server::DuplexGenRequest> req(new server::DuplexGenRequest());
    req->tokens = { 1, 2 };
    int32_t lane = -1;
    std::string why;
    int open_st = RAD_E_STATE;
    std::thread opener([&] { open_st = bridge.open(std::move(req), &lane, &why); });
    const RadBatch* b = nullptr;
    for (int i = 0; i < 2000 && !b; ++i) {
        b = f.drv.next(f.step++);
        if (!b) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(b != nullptr);
    CHECK_EQ(b->gen, (int32_t)RAD_GEN_PASS_DUPLEX_PROMPT);
    CHECK_OK(f.drv.done());
    opener.join();
    REQUIRE_EQ(open_st, RAD_OK);
    REQUIRE(lane >= 0);

    std::vector<float> samples((size_t)kSamples, 1.0f);
    server::DuplexGenReply out;
    CHECK_EQ(bridge.step(lane, samples, 7, &out, &why), server::kDuplexTimeout);
    CHECK(has(why, "did not finish"));
    CHECK(bridge.step(lane, samples, 8, &out, &why) < 0);
}

/* An open whose prompt pass never runs has its own bound, closes the lane and leaves nothing
 * queued. */
TEST(an_open_that_never_runs_times_out_and_releases_the_lane) {
    Fixture f;
    REQUIRE_EQ(f.init(), RAD_OK);
    server::EngineDuplexBridge bridge;
    bridge.init(&f.drv, info(), server::EngineDuplexBridge::Timeouts{ 50, 1000 });

    std::unique_ptr<server::DuplexGenRequest> req(new server::DuplexGenRequest());
    req->tokens = { 1 };
    int32_t lane = -1;
    std::string why;
    CHECK_EQ(bridge.open(std::move(req), &lane, &why), RAD_E_STATE);
    CHECK(has(why, "did not finish"));
    CHECK_EQ(lane, -1);
    CHECK_EQ(bridge.queued(), (int64_t)0);
}

RAD_TEST_MAIN()
