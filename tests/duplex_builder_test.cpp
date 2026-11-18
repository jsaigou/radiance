/* duplex_builder_test.cpp -- the declare phase of a duplex tenant (rad_declare_duplex_tenant).
 *
 * An empty Registry is enough: the declaration validates its own geometry, templates and buffers,
 * and it does not issue an op. The pass driver and server bridge are follow-ups, so what is pinned
 * here is the declaration a plugin states and the program the builder records from it.
 */
#include "rad_test.h"

#include "build/rad_build.h"

#include <memory>
#include <string>
#include <utility>

using namespace rad;

namespace {

constexpr int64_t kSamplesPerFrame = 4;

/* A builder and the buffers a valid duplex declaration needs. max_tok and max_seqs are the build
 * context's; a test overrides one to exercise the declaration's bounds. */
struct Fixture {
    Registry reg;
    RadModelMeta meta{};
    RadBuildCtx ctx{};
    std::unique_ptr<Builder> b;

    explicit Fixture(int64_t max_tok = 64, int64_t max_seqs = 8) {
        meta.arch_id = "duplex_test";
        meta.name = "duplex-test";
        meta.quant = "bf16";
        meta.n_layers = 1;
        meta.n_embd = 8;
        ctx.rank = 0;
        ctx.world_size = 1;
        ctx.max_tok = max_tok;
        ctx.max_seqs = max_seqs;
        ctx.max_ctx = max_tok;
        ctx.scope = "";
        b = std::make_unique<Builder>(reg, meta, ctx);
    }

    rad_buf buf(const char* name, uint32_t dtype, int64_t rows, int64_t cols = 1,
                int domain = RAD_DOMAIN_DEVICE) {
        RadBufDecl d{};
        d.dtype = dtype;
        d.rank = cols > 1 ? 2 : 1;
        d.shape[0] = rows;
        d.shape[1] = cols;
        d.kind = RAD_BUF_PERSIST;
        d.domain = domain;
        return b->decl_buffer(name, &d);
    }

    /* A valid declaration: two lanes, 4-sample frames at 50 Hz (25/2 a second), a card and up to
     * two reference frames, both audio-input projectors, text on. */
    RadDuplexDecl decl() {
        RadDuplexDecl d{};
        d.name = "tenant";
        d.tmpl = "<card>{card}</card>{references}";
        d.max_prompt = 16;
        d.max_ref_frames = 2;
        d.ref_rate = 50;
        d.ref_hop = kSamplesPerFrame;
        d.max_lanes = 2;
        d.sample_rate = 50;
        d.channels = 1;
        d.samples_per_frame = kSamplesPerFrame;
        d.latent_dim = 2;
        d.frame_rate_num = 25;
        d.frame_rate_den = 2;
        d.n_audio_in = 2;
        d.audio_ins[0] = RadDuplexAudioIn{ RAD_DUPLEX_AUDIO_USER, 2, 8, 8, 1 };
        d.audio_ins[1] = RadDuplexAudioIn{ RAD_DUPLEX_AUDIO_MODEL, 2, 8, 8, 1 };
        d.text = 1;
        d.audio_in = buf("wire_in", RAD_F32, d.max_lanes, d.samples_per_frame);
        d.audio_out = buf("wire_out", RAD_F32, d.max_lanes, d.samples_per_frame);
        d.text_out = buf("text_out", RAD_I32, d.max_lanes);
        d.status = buf("status", RAD_I32, d.max_lanes);
        return d;
    }
};

void refused_by_name(Builder& b, const RadDuplexDecl& d, const char* field) {
    const int st = b.declare_duplex_tenant(&d);
    CHECK(st < 0);
    REQUIRE(!b.errors().empty());
    const std::string& e = b.errors().back();
    CHECK(has(e, "declare_duplex_tenant"));
    CHECK(has(e, "tenant"));
    CHECK(has(e, field));
    if (!has(e, field)) fprintf(stderr, "---- %s\n", e.c_str());
}

/* The declaration a plugin states is accepted, its strings are copied into the program, and the
 * program reports the tenant as its pass-driven kind. */
TEST(a_valid_duplex_declaration_is_accepted_and_stored) {
    Fixture f;
    RadDuplexDecl d = f.decl();
    CHECK_OK(f.b->declare_duplex_tenant(&d));

    const Program& p = f.b->program();
    CHECK(p.is_duplex_tenant());
    CHECK(p.is_pass_driven());
    CHECK(!p.is_generator());
    CHECK(!p.is_audio_generator());
    CHECK(!p.is_video_generator());
    CHECK(!p.is_shape_generator());
    CHECK_EQ(p.duplex.max_lanes, (int64_t)2);
    CHECK_EQ(p.dup_str.name, std::string("tenant"));
    CHECK_EQ(p.dup_str.tmpl, std::string(d.tmpl));
    /* The program is moved after declare: the copies live in dup_str. */
    CHECK(p.duplex.name == nullptr);
    CHECK(p.duplex.tmpl == nullptr);
}

/* Every rule of the declaration names the tenant and the field it refused. */
TEST(each_validation_rule_is_refused_by_name) {
    auto check = [](auto mutate, const char* field) {
        Fixture f;
        RadDuplexDecl d = f.decl();
        mutate(f, d);
        refused_by_name(*f.b, d, field);
        CHECK(!f.b->program().is_duplex_tenant());
    };

    check([](Fixture&, RadDuplexDecl& d) { d.max_lanes = 9; }, "max_lanes");
    check([](Fixture&, RadDuplexDecl& d) { d.max_prompt = 65; }, "max_prompt");
    check([](Fixture&, RadDuplexDecl& d) { d.sample_rate = 0; }, "sample_rate");
    check([](Fixture&, RadDuplexDecl& d) { d.channels = 0; }, "channels");
    check([](Fixture&, RadDuplexDecl& d) { d.samples_per_frame = 0; }, "samples_per_frame");
    check([](Fixture&, RadDuplexDecl& d) { d.latent_dim = 0; }, "latent_dim");
    check([](Fixture&, RadDuplexDecl& d) { d.max_ref_frames = -1; }, "max_ref_frames");
    /* 4 * 24 != 50 * 2: the frame rate does not follow from the sample rate. */
    check([](Fixture&, RadDuplexDecl& d) { d.frame_rate_num = 24; }, "samples_per_frame");
    check([](Fixture&, RadDuplexDecl& d) { d.tmpl = "{references}"; }, "{card}");
    check([](Fixture&, RadDuplexDecl& d) { d.tmpl = "{card}{card}"; }, "{card}");
    check([](Fixture&, RadDuplexDecl& d) { d.tmpl = "{card}{references}{references}"; },
          "{references}");
    check([](Fixture&, RadDuplexDecl& d) { d.max_ref_frames = 2; d.ref_rate = 0; }, "ref_rate");
    check([](Fixture&, RadDuplexDecl& d) { d.max_ref_frames = 2; d.ref_hop = 5; }, "ref_hop");
    check([](Fixture&, RadDuplexDecl& d) { d.n_audio_in = 3; }, "n_audio_in");
    check([](Fixture&, RadDuplexDecl& d) { d.audio_ins[0].source = 9; }, "source");
    check([](Fixture&, RadDuplexDecl& d) { d.audio_ins[0].in_dim = 0; }, "in_dim");
    check([](Fixture&, RadDuplexDecl& d) { d.audio_ins[1].rms_norm = 2; }, "rms_norm");
    check([](Fixture& f, RadDuplexDecl& d) {
        d.audio_in = f.buf("bad_in", RAD_I32, d.max_lanes, d.samples_per_frame);
    }, "audio_in");
    check([](Fixture& f, RadDuplexDecl& d) {
        d.audio_out = f.buf("small_out", RAD_F32, 1);
    }, "audio_out");
    check([](Fixture&, RadDuplexDecl& d) { d.text_out = RAD_NULL_HANDLE; }, "text_out");
    check([](Fixture& f, RadDuplexDecl& d) {
        d.status = f.buf("host_status", RAD_I32, d.max_lanes, 1, RAD_DOMAIN_HOST);
    }, "status");
}

/* A program that already declared a piece generator refuses a duplex tenant, by the existing
 * declaration's name. */
TEST(a_program_that_declared_an_audio_generator_refuses_a_duplex_tenant) {
    Fixture f;
    RadAudioGenDecl a{};
    a.name = "piece";
    a.tmpl = "<i>{instructions}</i>{input}";
    a.max_prompt = 8;
    a.max_frames = 2;
    a.frame_rate_num = 25;
    a.frame_rate_den = 2;
    a.lat_num = 1;
    a.lat_den = 1;
    a.samples_per_latent = 4;
    a.sample_rate = 50;
    a.channels = 1;
    a.default_cfg = 1.0;
    a.max_lanes = 1;
    a.prompt_whole = 1;
    a.first_frame = 0;
    a.voc_chunk = 2;
    a.voc_context = 0;
    a.status = f.buf("piece_status", RAD_I32, 1);
    a.audio_out = f.buf("piece_audio", RAD_F32, 2 * 4);
    REQUIRE_EQ(f.b->declare_audio_generator(&a), RAD_OK);

    RadDuplexDecl d = f.decl();
    CHECK(f.b->declare_duplex_tenant(&d) < 0);
    REQUIRE(!f.b->errors().empty());
    const std::string& e = f.b->errors().back();
    CHECK(has(e, "declare_duplex_tenant"));
    CHECK(has(e, "tenant"));
    CHECK(has(e, "audio generator"));
    CHECK(!f.b->program().is_duplex_tenant());
}

/* The reverse: once a duplex tenant is declared, logits and an audio generator are refused, naming
 * the tenant. */
TEST(a_duplex_tenant_refuses_logits_and_an_audio_generator) {
    {
        Fixture f;
        RadDuplexDecl d = f.decl();
        REQUIRE_EQ(f.b->declare_duplex_tenant(&d), RAD_OK);
        const rad_buf logits = f.buf("logits", RAD_F32, 4, 8);
        CHECK(f.b->declare_logits(logits) < 0);
        REQUIRE(!f.b->errors().empty());
        const std::string& e = f.b->errors().back();
        CHECK(has(e, "declare_logits"));
        CHECK(has(e, "duplex tenant"));
        CHECK(has(e, "tenant"));
    }
    {
        Fixture f;
        RadDuplexDecl d = f.decl();
        REQUIRE_EQ(f.b->declare_duplex_tenant(&d), RAD_OK);
        RadAudioGenDecl a{};
        a.name = "piece";
        a.tmpl = "<i>{instructions}</i>{input}";
        a.max_prompt = 8;
        a.max_frames = 2;
        a.frame_rate_num = 25;
        a.frame_rate_den = 2;
        a.lat_num = 1;
        a.lat_den = 1;
        a.samples_per_latent = 4;
        a.sample_rate = 50;
        a.channels = 1;
        a.default_cfg = 1.0;
        a.max_lanes = 1;
        a.prompt_whole = 1;
        a.first_frame = 0;
        a.voc_chunk = 2;
        a.voc_context = 0;
        a.status = f.buf("piece_status", RAD_I32, 1);
        a.audio_out = f.buf("piece_audio", RAD_F32, 2 * 4);
        CHECK(f.b->declare_audio_generator(&a) < 0);
        REQUIRE(!f.b->errors().empty());
        const std::string& e = f.b->errors().back();
        CHECK(has(e, "declare_audio_generator"));
        CHECK(has(e, "duplex tenant"));
        CHECK(has(e, "tenant"));
    }
}

/* The probe names the new kind beside the ones it already had, and leaves gen_audio clear. */
TEST(the_probe_reports_a_duplex_tenant) {
    RadArchProbe pb{};
    pb.gen_rows = 16;
    pb.gen_ctx = 64;
    pb.gen_duplex = 1;
    CHECK_EQ(pb.gen_duplex, 1);
    CHECK_EQ(pb.gen_audio, 0);
    CHECK_EQ(pb.gen_rows, (int64_t)16);
    CHECK_EQ(pb.gen_ctx, (int64_t)64);
}

}  /* namespace */

RAD_TEST_MAIN()
