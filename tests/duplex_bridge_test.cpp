/* tests/duplex_bridge_test.cpp -- the engine's duplex tenant bridge (core/server/duplex.cpp's
 * EngineDuplexTenant) against a fake IDuplexGen and the in-process WebSocket fake used by
 * tests/duplex_test.cpp. The fake records the prompt the server renders and answers deterministically,
 * so the card/reference rendering, tokenisation, text mapping and lane lifecycle are visible on the
 * wire without an engine or a model.
 */
#include "server_fakes.h"
#include "mm/media.h"

#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace {

uint32_t le32(const char* p) {
    return (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) |
           ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
}

void put_le32(std::string* s, uint32_t v) {
    s->push_back((char)(v & 0xff));
    s->push_back((char)((v >> 8) & 0xff));
    s->push_back((char)((v >> 16) & 0xff));
    s->push_back((char)((v >> 24) & 0xff));
}

std::string frame(uint32_t seq, const std::vector<float>& samples) {
    std::string s;
    put_le32(&s, seq);
    for (float f : samples) {
        uint32_t u = 0;
        std::memcpy(&u, &f, sizeof u);
        put_le32(&s, u);
    }
    return s;
}

std::vector<float> samples(int seed) {
    std::vector<float> x(3200);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = (float)((int)((i * 37 + seed * 101) % 2001) - 1000) / 1000.0f;
    return x;
}

/* A 16-bit mono WAV of the given samples at `rate`. */
std::string wav_of(const std::vector<int16_t>& x, uint32_t rate) {
    std::string d;
    for (int16_t v : x) {
        d.push_back((char)(v & 0xff));
        d.push_back((char)((v >> 8) & 0xff));
    }
    auto u32 = [](uint32_t v) {
        std::string b;
        for (int i = 0; i < 4; ++i) b.push_back((char)((v >> (8 * i)) & 0xff));
        return b;
    };
    auto u16 = [](uint16_t v) { return std::string{ (char)(v & 0xff), (char)(v >> 8) }; };
    return "RIFF" + u32((uint32_t)(36 + d.size())) + "WAVEfmt " + u32(16) + u16(1) + u16(1) +
           u32(rate) + u32(rate * 2) + u16(2) + u16(16) + "data" + u32((uint32_t)d.size()) + d;
}

struct Reply {
    uint32_t seq = 0;
    int flags = 0;
    std::string text;
    std::string audio;
};

bool decode_reply(const std::string& m, Reply* r) {
    if (m.size() < 9) return false;
    r->seq = le32(m.data());
    r->flags = (uint8_t)m[4];
    const uint32_t n = le32(m.data() + 5);
    if (m.size() < 9 + (size_t)n) return false;
    r->text = m.substr(9, n);
    r->audio = m.substr(9 + n);
    if ((r->flags & 1) && r->audio.size() != 3200 * sizeof(float)) return false;
    if (!(r->flags & 1) && !r->audio.empty()) return false;
    return true;
}

struct Conn : WsConn {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::pair<std::string, bool>> in;
    bool hung_up = false;
    bool closed = false;
    int  close_code = 0;
    std::vector<std::pair<std::string, bool>> out;

    bool read(std::string* msg, bool* binary) override {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !in.empty() || hung_up || closed; });
        if (in.empty()) return false;
        *msg = std::move(in.front().first);
        *binary = in.front().second;
        in.pop_front();
        return true;
    }
    bool send(const std::string& t) override { return put(t, false); }
    bool send_binary(const std::string& b) override { return put(b, true); }
    bool put(const std::string& s, bool binary) {
        std::lock_guard<std::mutex> lk(m);
        out.emplace_back(s, binary);
        cv.notify_all();
        return !closed;
    }
    void close(int code, const std::string&) override {
        std::lock_guard<std::mutex> lk(m);
        closed = true;
        close_code = code;
        cv.notify_all();
    }
    void to_server(const std::string& s, bool binary) {
        std::lock_guard<std::mutex> lk(m);
        in.emplace_back(s, binary);
        cv.notify_all();
    }
    void client(const json& j) { to_server(j.dump(), false); }
    void binary(const std::string& b) { to_server(b, true); }
    void hangup() {
        std::lock_guard<std::mutex> lk(m);
        hung_up = true;
        cv.notify_all();
    }
    json wait_for(const std::string& type, size_t nth = 0, int timeout_ms = 10000) {
        std::unique_lock<std::mutex> lk(m);
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            size_t seen = 0;
            for (const auto& o : out) {
                if (o.second) continue;
                json j = json::parse(o.first);
                if (j.value("type", "") == type && seen++ == nth) return j;
            }
            if (cv.wait_until(lk, until) == std::cv_status::timeout) return json();
        }
    }
    std::string wait_binary(size_t nth = 0, int timeout_ms = 10000) {
        std::unique_lock<std::mutex> lk(m);
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            size_t seen = 0;
            for (const auto& o : out)
                if (o.second && seen++ == nth) return o.first;
            if (cv.wait_until(lk, until) == std::cv_status::timeout) return std::string();
        }
    }
    bool wait_closed(int timeout_ms = 10000) {
        std::unique_lock<std::mutex> lk(m);
        return cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return closed; });
    }
};

json start_json(const std::string& tenant, const json& reference = json::object()) {
    json j;
    j["type"] = "start";
    j["contract"] = "0";
    j["session_id"] = "s1";
    j["tenant"] = tenant;
    j["text"] = true;
    j["max_minutes"] = 30;
    j["card"] = json::string_t("card");
    j["reference"] = reference;
    return j;
}

struct FakeDuplex : IDuplexGen {
    DuplexGenInfo inf;
    mutable std::mutex mu;
    std::vector<DuplexGenRequest> opens;
    int closes = 0;
    int next_lane = 0;
    int open_rc = RAD_OK;
    bool no_audio = false;
    int32_t token = -1;

    FakeDuplex() {
        inf.name = "plugin";
        inf.tmpl = "<c>{card}</c>{references}";
        inf.text = true;
        inf.max_lanes = 2;
        inf.max_prompt = 100;
        inf.max_ref_frames = 4;
        inf.ref_rate = 8000;
        inf.ref_hop = 4;
        inf.sample_rate = 8000;
        inf.samples_per_frame = 4;
        inf.channels = 1;
        inf.frame_rate_num = 1;
        inf.frame_rate_den = 1;
        inf.latent_dim = 2;
    }
    const DuplexGenInfo& info() const override { return inf; }
    int open(std::unique_ptr<DuplexGenRequest> r, int32_t* lane, std::string* why) override {
        std::lock_guard<std::mutex> lk(mu);
        if (open_rc < 0) {
            if (why) *why = "no lane";
            return open_rc;
        }
        opens.push_back(*r);
        *lane = next_lane++;
        return RAD_OK;
    }
    int step(int32_t, const std::vector<float>& in, uint64_t, DuplexGenReply* out,
             std::string*) override {
        DuplexGenReply r;
        r.status = no_audio ? 1 : 0;
        if (!no_audio) r.audio = in;
        r.token = token;
        *out = std::move(r);
        return RAD_OK;
    }
    int flush(int32_t, uint64_t, DuplexGenReply* out, std::string*) override {
        *out = DuplexGenReply{};
        return RAD_OK;
    }
    void close(int32_t) override {
        std::lock_guard<std::mutex> lk(mu);
        ++closes;
    }
    int64_t queued() const override { return 0; }
    int64_t running() const override { return 0; }
    DuplexGenRequest last_open() {
        std::lock_guard<std::mutex> lk(mu);
        return opens.empty() ? DuplexGenRequest{} : opens.back();
    }
};

struct Session {
    Conn conn;
    std::thread route;
    explicit Session(Fixture& f) {
        const SocketHandler* h = f.srv->router().socket_for("/duplex/v1/session");
        REQUIRE(h != nullptr);
        route = std::thread([this, h] {
            HttpRequest q;
            q.method = "GET";
            q.path = "/duplex/v1/session";
            (*h)(q, conn);
        });
    }
    ~Session() {
        conn.hangup();
        if (route.joinable()) route.join();
    }
    void join() {
        if (route.joinable()) route.join();
    }
};

void append_chars(const std::string& s, std::vector<int32_t>* tok, std::vector<int32_t>* frame) {
    for (unsigned char c : s) {
        tok->push_back((int32_t)c);
        frame->push_back(-1);
    }
}

}  /* namespace */

TEST(info_lists_the_plugin_tenant) {
    FakeDuplex gen;
    Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.duplex = &gen; } });
    const json info = json::parse(f.get("/duplex/v1/info").body);
    REQUIRE_EQ(info["tenants"].size(), (size_t)1);
    CHECK_EQ(info["tenants"][0].value("name", std::string()), std::string("plugin"));
    CHECK_EQ(info["tenants"][0].value("text", false), true);
    CHECK_EQ(info["tenants"][0].value("max_lanes", 0), 2);
}

/* The string card and the template reach the engine's request as the expected rows; an object
 * with a string "text" renders that text. */
TEST(the_card_and_template_reach_the_engine_as_rows) {
    FakeDuplex gen;
    Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.duplex = &gen; } });
    {
        Session s(f);
        json j = start_json("plugin");
        j["card"] = "hello";
        s.conn.client(j);
        REQUIRE(!s.conn.wait_for("ready").is_null());
        const DuplexGenRequest r = gen.last_open();
        std::vector<int32_t> want, want_frame;
        append_chars("<c>hello</c>", &want, &want_frame);
        CHECK(r.tokens == want);
        CHECK(r.ref_frame == want_frame);
        CHECK(r.ref_audio.empty());
        s.conn.client({ { "type", "close" } });
        REQUIRE(s.conn.wait_closed());
        s.join();
    }
    {
        Session s(f);
        json j = start_json("plugin");
        j["card"] = { { "text", "object-card" }, { "other", 7 } };
        s.conn.client(j);
        REQUIRE(!s.conn.wait_for("ready").is_null());
        std::vector<int32_t> want, want_frame;
        append_chars("<c>object-card</c>", &want, &want_frame);
        const DuplexGenRequest r = gen.last_open();
        CHECK(r.tokens == want);
        CHECK(r.ref_frame == want_frame);
    }
}

/* No card, a null card and an empty object all render as nothing; a literal "{}" must not reach
 * the model. */
TEST(an_absent_or_empty_card_renders_nothing) {
    FakeDuplex gen;
    Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.duplex = &gen; } });
    for (int variant = 0; variant < 3; ++variant) {
        Session s(f);
        json j = start_json("plugin");
        if (variant == 0) j.erase("card");
        else if (variant == 1) j["card"] = nullptr;
        else j["card"] = json::object();
        s.conn.client(j);
        REQUIRE(!s.conn.wait_for("ready").is_null());
        std::vector<int32_t> want, want_frame;
        append_chars("<c></c>", &want, &want_frame);
        const DuplexGenRequest r = gen.last_open();
        CHECK(r.tokens == want);
        CHECK(r.ref_frame == want_frame);
        s.conn.client({ { "type", "close" } });
        REQUIRE(s.conn.wait_closed());
        s.join();
    }
}

/* Reference audio is decoded at the declaration's rate and keeps its true sample count: a final
 * partial frame is not zero-padded, because the codec's per-conv partial-hop rule is not an audio
 * pad. A 5-sample clip at hop 4 is two frames, the last 1 sample, and two placeholder rows. */
TEST(reference_bytes_become_frames_and_rows) {
    if (!mm::decoder_available()) return;
    FakeDuplex gen;
    gen.inf.max_ref_frames = 4;
    Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.duplex = &gen; } });
    Session s(f);
    const std::string wav = wav_of({ 1000, -2000, 3000, 4000, 5000 }, 8000);
    json j = start_json("plugin");
    j["card"] = "hi";
    j["reference"] = { { "bytes", (int)wav.size() } };
    s.conn.client(j);
    s.conn.binary(wav);
    REQUIRE(!s.conn.wait_for("ready").is_null());
    const DuplexGenRequest r = gen.last_open();
    /* Five samples at a 4-sample hop are two frames, the last 1 sample. */
    std::vector<int32_t> want, want_frame;
    append_chars("<c>hi</c>", &want, &want_frame);
    for (int32_t k = 0; k < 2; ++k) {
        want.push_back(0);
        want_frame.push_back(k);
    }
    CHECK(r.tokens == want);
    CHECK(r.ref_frame == want_frame);
    REQUIRE_EQ(r.ref_cu.size(), (size_t)2);
    CHECK_EQ(r.ref_cu[0], 0);
    CHECK_EQ(r.ref_cu[1], 5);
    CHECK_EQ(r.ref_audio.size(), (size_t)5);
}

/* A step's audio and text token cross the bridge unchanged. */
TEST(a_step_maps_audio_and_a_text_token_to_text) {
    FakeDuplex gen;
    gen.token = 65;                      /* 'A' in FakeTokenizer */
    Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.duplex = &gen; } });
    Session s(f);
    s.conn.client(start_json("plugin"));
    REQUIRE(!s.conn.wait_for("ready").is_null());
    const std::vector<float> in = samples(3);
    const std::string sent = frame(0, in);
    s.conn.binary(sent);
    Reply r;
    REQUIRE(decode_reply(s.conn.wait_binary(0), &r));
    CHECK_EQ(r.seq, (uint32_t)0);
    CHECK_EQ(r.flags, 3);
    CHECK_EQ(r.text, std::string("A"));
    CHECK(r.audio == sent.substr(4));
}

/* Status 1 is a silent frame: the wire reply has flags = 0. */
TEST(a_status_one_reply_is_silent_on_the_wire) {
    FakeDuplex gen;
    gen.no_audio = true;
    gen.token = -1;
    Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.duplex = &gen; } });
    Session s(f);
    s.conn.client(start_json("plugin"));
    REQUIRE(!s.conn.wait_for("ready").is_null());
    s.conn.binary(frame(0, samples(0)));
    Reply r;
    REQUIRE(decode_reply(s.conn.wait_binary(0), &r));
    CHECK_EQ(r.seq, (uint32_t)0);
    CHECK_EQ(r.flags, 0);
    CHECK(r.text.empty());
    CHECK(r.audio.empty());
}

/* The tenant's lane limit and an engine RAD_E_FULL both surface as session_limit. */
TEST(lane_limits_surface_as_session_limit) {
    {
        FakeDuplex gen;
        gen.inf.max_lanes = 1;
        Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.duplex = &gen; } });
        Session a(f);
        a.conn.client(start_json("plugin"));
        REQUIRE(!a.conn.wait_for("ready").is_null());
        Session b(f);
        b.conn.client(start_json("plugin"));
        const json e = b.conn.wait_for("error");
        REQUIRE(!e.is_null());
        CHECK_EQ(e.value("code", std::string()), std::string("session_limit"));
        CHECK(b.conn.wait_closed());
    }
    {
        FakeDuplex gen;
        gen.open_rc = RAD_E_FULL;
        Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.duplex = &gen; } });
        Session s(f);
        s.conn.client(start_json("plugin"));
        const json e = s.conn.wait_for("error");
        REQUIRE(!e.is_null());
        CHECK_EQ(e.value("code", std::string()), std::string("session_limit"));
        CHECK(s.conn.wait_closed());
    }
}

/* A card or reference the engine refuses (not audio, too long, unreadable) is the CLIENT's mistake: the contract's
 * bad_request, not internal, which means a server fault and would page someone for a user's bad clip. */
TEST(a_refused_card_or_reference_is_a_bad_request_not_an_internal_error) {
    FakeDuplex gen;
    gen.open_rc = RAD_E_INVAL;
    Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.duplex = &gen; } });
    Session s(f);
    s.conn.client(start_json("plugin"));
    const json e = s.conn.wait_for("error");
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("bad_request"));
    CHECK(e.value("fatal", false));
    /* The reason is fixed text: nothing the client sent is echoed back. */
    CHECK(e.value("message", std::string()).find("reference") != std::string::npos);
    CHECK(s.conn.wait_closed());
}

/* A close message and a dropped socket both release the lane. */
TEST(close_and_socket_drop_release_the_lane) {
    FakeDuplex gen;
    gen.inf.max_lanes = 1;
    Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.duplex = &gen; } });
    {
        Session s(f);
        s.conn.client(start_json("plugin"));
        REQUIRE(!s.conn.wait_for("ready").is_null());
        s.conn.client({ { "type", "close" } });
        REQUIRE(s.conn.wait_closed());
        s.join();
        std::lock_guard<std::mutex> lk(gen.mu);
        CHECK_EQ(gen.closes, 1);
    }
    {
        Session s(f);
        s.conn.client(start_json("plugin"));
        REQUIRE(!s.conn.wait_for("ready").is_null());
        s.conn.hangup();
        s.join();
        std::lock_guard<std::mutex> lk(gen.mu);
        CHECK_EQ(gen.closes, 2);
    }
}

RAD_TEST_MAIN()
