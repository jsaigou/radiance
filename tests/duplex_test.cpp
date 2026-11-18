/* tests/duplex_test.cpp -- the duplex engine endpoint (core/server/duplex.cpp) against the
 * model-free test tenants and a tenant that records what the endpoint delivered. */
#include "server_fakes.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
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

std::string blob(size_t n, uint32_t seed) {
    std::string b(n, '\0');
    uint32_t x = seed;
    for (char& c : b) {
        x = x * 1664525u + 1013904223u;
        c = (char)(x >> 24);
    }
    return b;
}

std::vector<float> samples(int seed) {
    std::vector<float> x(3200);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = (float)((int)((i * 37 + seed * 101) % 2001) - 1000) / 1000.0f;
    x[0] = -0.0f;
    x[1] = 0.1f;
    x[2] = 1.0f;
    x[3] = -1.0f;
    x[4] = 1e-9f;
    return x;
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
    /* What the client sends is what the server reads: the queue `read` drains, not `out`, which
     * holds what the server sent. */
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
    j["text"] = false;
    j["max_minutes"] = 30;
    j["card"] = json::object();
    j["reference"] = reference;
    return j;
}

struct RecordingTenant : DuplexTenant {
    struct Lane : DuplexLane {
        explicit Lane(RecordingTenant* o) : owner(o) {}
        DuplexReply step(uint32_t, const std::vector<float>& samples) override {
            DuplexReply r;
            r.audio = samples;
            return r;
        }
        DuplexReply flush() override { return DuplexReply{}; }
        void close() override {
            std::lock_guard<std::mutex> lk(owner->m);
            ++owner->closes;
        }
        RecordingTenant* owner;
    };
    DuplexTenantInfo info() const override {
        DuplexTenantInfo i;
        i.name = name;
        i.text = false;
        i.max_lanes = 3;
        return i;
    }
    int open(const DuplexStart& st, std::unique_ptr<DuplexLane>* out, std::string*) override {
        {
            std::lock_guard<std::mutex> lk(m);
            reference = st.reference;
            card = st.card;
        }
        out->reset(new Lane(this));
        return RAD_OK;
    }
    std::string name = "record";
    std::mutex m;
    std::vector<uint8_t> reference;
    json card;
    int closes = 0;
};

struct Session {
    Conn conn;
    std::thread route;
    Session(Fixture& f, const std::map<std::string, std::string>& headers = {}) {
        const SocketHandler* h = f.srv->router().socket_for("/duplex/v1/session");
        REQUIRE(h != nullptr);
        route = std::thread([this, h, headers] {
            HttpRequest q;
            q.method = "GET";
            q.path = "/duplex/v1/session";
            for (const auto& [k, v] : headers) q.headers[k] = v;
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

}  /* namespace */

TEST(info_lists_the_test_tenants_only_when_the_flag_is_on) {
    Fixture off;
    HttpResponse r = off.get("/duplex/v1/info");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    CHECK_EQ(j.value("contract", std::string()), std::string("0"));
    CHECK_EQ(j.value("engine", std::string()), std::string("radiance"));
    CHECK(j["tenants"].empty());
    CHECK_EQ(j["frame"].value("sample_rate", 0), 24000);
    CHECK_EQ(j["frame"].value("samples", 0), 3200);
    CHECK_EQ(j["frame"].value("format", std::string()), std::string("f32le"));
    CHECK_EQ(j.value("ready", false), true);

    Session s(off);
    s.conn.client(start_json("echo"));
    const json e = s.conn.wait_for("error");
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("tenant_unavailable"));
    CHECK_EQ(e.value("fatal", false), true);

    Fixture on(Fixture::Tweak{ [](ServerOptions& o, Deps&) {
        o.duplex_test_tenants = true;
        o.duplex_max_lanes = 3;
    } });
    r = on.get("/duplex/v1/info");
    CHECK_EQ(r.status, 200);
    j = json::parse(r.body);
    REQUIRE_EQ(j["tenants"].size(), (size_t)2);
    CHECK_EQ(j["tenants"][0].value("name", std::string()), std::string("echo"));
    CHECK_EQ(j["tenants"][0].value("text", true), false);
    CHECK_EQ(j["tenants"][0].value("max_lanes", 0), 3);
    CHECK_EQ(j["tenants"][1].value("name", std::string()), std::string("echo1"));
}

TEST(start_is_answered_with_ready) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.duplex_test_tenants = true; } });
    Session s(f);
    s.conn.client(start_json("echo"));
    const json r = s.conn.wait_for("ready");
    REQUIRE(!r.is_null());
    CHECK_EQ(r.value("text", true), false);
    REQUIRE_EQ(r["input_rates"].size(), (size_t)1);
    CHECK_EQ(r["input_rates"][0].get<int>(), 24000);
}

TEST(echo_returns_each_frame_bit_exact) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.duplex_test_tenants = true; } });
    Session s(f);
    s.conn.client(start_json("echo"));
    REQUIRE(!s.conn.wait_for("ready").is_null());
    for (uint32_t k = 0; k < 3; ++k) {
        const std::string sent = frame(k, samples((int)k));
        s.conn.binary(sent);
        const std::string got = s.conn.wait_binary(k);
        REQUIRE(!got.empty());
        Reply r;
        REQUIRE(decode_reply(got, &r));
        CHECK_EQ(r.seq, k);
        CHECK_EQ(r.flags, 1);
        CHECK(r.text.empty());
        CHECK(r.audio == sent.substr(4));
    }
}

/* The echo tenant is bit-exact by contract: the engine-path clamp must not touch it, even for a
 * sample outside the client's stated [-1, 1] range. */
TEST(echo_is_not_clamped) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.duplex_test_tenants = true; } });
    Session s(f);
    s.conn.client(start_json("echo"));
    REQUIRE(!s.conn.wait_for("ready").is_null());
    std::vector<float> loud = samples(7);
    loud[0] = 2.0f;
    loud[1] = -3.0f;
    const std::string sent = frame(0, loud);
    s.conn.binary(sent);
    const std::string got = s.conn.wait_binary(0);
    REQUIRE(!got.empty());
    Reply r;
    REQUIRE(decode_reply(got, &r));
    CHECK_EQ(r.flags, 1);
    CHECK(r.audio == sent.substr(4));
}

TEST(echo1_holds_one_frame_and_flush_returns_it) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.duplex_test_tenants = true; } });
    Session s(f);
    s.conn.client(start_json("echo1"));
    REQUIRE(!s.conn.wait_for("ready").is_null());

    const std::string f0 = frame(0, samples(0));
    const std::string f1 = frame(1, samples(1));
    const std::string f2 = frame(2, samples(2));
    s.conn.binary(f0);
    s.conn.binary(f1);
    s.conn.binary(f2);

    Reply r0, r1, r2;
    REQUIRE(decode_reply(s.conn.wait_binary(0), &r0));
    CHECK_EQ(r0.seq, (uint32_t)0);
    CHECK_EQ(r0.flags, 0);
    REQUIRE(decode_reply(s.conn.wait_binary(1), &r1));
    CHECK_EQ(r1.seq, (uint32_t)1);
    CHECK_EQ(r1.flags, 1);
    CHECK(r1.audio == f0.substr(4));
    REQUIRE(decode_reply(s.conn.wait_binary(2), &r2));
    CHECK_EQ(r2.seq, (uint32_t)2);
    CHECK_EQ(r2.flags, 1);
    CHECK(r2.audio == f1.substr(4));

    s.conn.client({ { "type", "flush" } });
    Reply rf;
    REQUIRE(decode_reply(s.conn.wait_binary(3), &rf));
    CHECK_EQ(rf.seq, (uint32_t)2);
    CHECK_EQ(rf.flags, 1);
    CHECK(rf.audio == f2.substr(4));

    s.conn.client({ { "type", "flush" } });
    Reply empty;
    REQUIRE(decode_reply(s.conn.wait_binary(4), &empty));
    CHECK_EQ(empty.seq, (uint32_t)2);
    CHECK_EQ(empty.flags, 0);
}

TEST(a_lane_limit_refuses_a_start_and_release_lets_it_open_again) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) {
        o.duplex_test_tenants = true;
        o.duplex_max_lanes = 1;
    } });
    {
        Session a(f);
        a.conn.client(start_json("echo"));
        REQUIRE(!a.conn.wait_for("ready").is_null());

        Session b(f);
        b.conn.client(start_json("echo"));
        const json e = b.conn.wait_for("error");
        REQUIRE(!e.is_null());
        CHECK_EQ(e.value("code", std::string()), std::string("session_limit"));
        CHECK(b.conn.wait_closed());

        a.conn.client({ { "type", "close" } });
        REQUIRE(a.conn.wait_closed());
        a.join();
    }
    Session c(f);
    c.conn.client(start_json("echo"));
    CHECK(!c.conn.wait_for("ready").is_null());
}

TEST(a_wrong_step_length_is_a_fatal_bad_frame) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.duplex_test_tenants = true; } });
    Session s(f);
    s.conn.client(start_json("echo"));
    REQUIRE(!s.conn.wait_for("ready").is_null());
    s.conn.binary(std::string(100, 'x'));
    const json e = s.conn.wait_for("error");
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("bad_frame"));
    CHECK_EQ(e.value("fatal", false), true);
    CHECK(s.conn.wait_closed());
}

TEST(a_wrong_seq_is_a_fatal_bad_sequence) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.duplex_test_tenants = true; } });
    Session s(f);
    s.conn.client(start_json("echo"));
    REQUIRE(!s.conn.wait_for("ready").is_null());
    s.conn.binary(frame(5, samples(0)));
    const json e = s.conn.wait_for("error");
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("bad_sequence"));
    CHECK_EQ(e.value("fatal", false), true);
    CHECK(s.conn.wait_closed());
}

TEST(reference_bytes_reach_the_tenant) {
    auto rec = std::make_shared<RecordingTenant>();
    Fixture f(Fixture::Tweak{ [rec](ServerOptions& o, Deps&) {
        o.duplex_test_tenants = true;
        o.duplex_tenants.push_back(rec);
    } });
    Session s(f);
    const std::string ref = blob(512u << 10, 7);
    json j = start_json("record");
    j["session_id"] = "session-7";
    j["card"] = { { "voice", "calm" } };
    j["reference"] = { { "id", "ref-1" }, { "bytes", (int)ref.size() },
                       { "transcript", "hello" } };
    s.conn.client(j);
    s.conn.binary(ref);
    REQUIRE(!s.conn.wait_for("ready").is_null());
    {
        std::lock_guard<std::mutex> lk(rec->m);
        CHECK(rec->reference == std::vector<uint8_t>(ref.begin(), ref.end()));
        CHECK_EQ(rec->card.value("voice", std::string()), std::string("calm"));
    }
}

TEST(a_reference_sent_in_chunks_reaches_the_tenant_byte_identical) {
    auto rec = std::make_shared<RecordingTenant>();
    Fixture f(Fixture::Tweak{ [rec](ServerOptions& o, Deps&) {
        o.duplex_test_tenants = true;
        o.duplex_tenants.push_back(rec);
    } });
    Session s(f);
    constexpr size_t kChunk = 1u << 20;
    const std::string ref = blob(20 * kChunk, 11);
    json j = start_json("record");
    j["reference"] = { { "id", "ref-1" }, { "bytes", (int)ref.size() },
                       { "transcript", "" } };
    s.conn.client(j);
    for (size_t off = 0; off < ref.size(); off += kChunk) {
        s.conn.binary(ref.substr(off, std::min(kChunk, ref.size() - off)));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(!s.conn.wait_for("ready").is_null());
    std::lock_guard<std::mutex> lk(rec->m);
    CHECK(rec->reference == std::vector<uint8_t>(ref.begin(), ref.end()));
}

TEST(a_reference_chunk_over_one_mib_is_refused) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.duplex_test_tenants = true; } });
    Session s(f);
    json j = start_json("echo");
    j["reference"] = { { "id", "" }, { "bytes", 2 << 20 }, { "transcript", "" } };
    s.conn.client(j);
    s.conn.binary(std::string((1u << 20) + 1, 'x'));
    const json e = s.conn.wait_for("error");
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("bad_request"));
    CHECK(s.conn.wait_closed());
}

TEST(more_reference_bytes_than_announced_is_refused) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.duplex_test_tenants = true; } });
    Session s(f);
    json j = start_json("echo");
    j["reference"] = { { "id", "" }, { "bytes", (1 << 20) + (512 << 10) },
                       { "transcript", "" } };
    s.conn.client(j);
    s.conn.binary(std::string(1u << 20, 'a'));
    s.conn.binary(std::string(1u << 20, 'b'));
    const json e = s.conn.wait_for("error");
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("bad_request"));
    CHECK(s.conn.wait_closed());
}

TEST(a_text_message_in_the_middle_of_a_reference_is_refused) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.duplex_test_tenants = true; } });
    Session s(f);
    json j = start_json("echo");
    j["reference"] = { { "id", "" }, { "bytes", 2 << 20 }, { "transcript", "" } };
    s.conn.client(j);
    s.conn.binary(std::string(1u << 20, 'a'));
    s.conn.client({ { "type", "flush" } });
    const json e = s.conn.wait_for("error");
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("bad_request"));
    CHECK(s.conn.wait_closed());
}

TEST(a_stalled_reference_hits_the_idle_timeout) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) {
        o.duplex_test_tenants = true;
        o.duplex_idle_s = 1;
    } });
    Session s(f);
    json j = start_json("echo");
    j["reference"] = { { "id", "" }, { "bytes", 2 << 20 }, { "transcript", "" } };
    s.conn.client(j);
    s.conn.binary(std::string(1u << 20, 'a'));
    const json e = s.conn.wait_for("error", 0, 5000);
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("timeout"));
    CHECK(s.conn.wait_closed());
}

TEST(the_reference_limit_is_advertised_and_enforced) {
    auto rec = std::make_shared<RecordingTenant>();
    Fixture f(Fixture::Tweak{ [rec](ServerOptions& o, Deps&) {
        o.duplex_test_tenants = true;
        o.duplex_max_reference_mib = 1;
        o.duplex_tenants.push_back(rec);
    } });
    const json info = json::parse(f.get("/duplex/v1/info").body);
    CHECK_EQ(info.value("max_reference_bytes", 0), 1 << 20);

    Session s(f);
    json j = start_json("record");
    j["reference"] = { { "id", "" }, { "bytes", (1 << 20) + 1 }, { "transcript", "" } };
    s.conn.client(j);
    const json e = s.conn.wait_for("error");
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("bad_request"));
    CHECK(s.conn.wait_closed());
}

TEST(an_unknown_tenant_is_unavailable) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.duplex_test_tenants = true; } });
    Session s(f);
    s.conn.client(start_json("not-a-tenant"));
    const json e = s.conn.wait_for("error");
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("tenant_unavailable"));
    CHECK(s.conn.wait_closed());
}

TEST(a_required_key_is_checked_on_info_and_the_session) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) {
        o.duplex_test_tenants = true;
        o.api_key = "k";
    } });
    CHECK_EQ(f.get("/duplex/v1/info").status, 401);
    {
        HttpRequest q;
        q.method = "GET";
        q.path = "/duplex/v1/info";
        q.headers["Authorization"] = "Bearer k";
        CHECK_EQ(f.srv->router().dispatch(q).status, 200);
    }
    {
        Session s(f);
        s.conn.client(start_json("echo"));
        const json e = s.conn.wait_for("error");
        REQUIRE(!e.is_null());
        CHECK_EQ(e.value("message", std::string()), std::string("invalid api key"));
        CHECK(s.conn.wait_closed());
    }
    {
        Session s(f, { { "Authorization", "Bearer k" } });
        s.conn.client(start_json("echo"));
        CHECK(!s.conn.wait_for("ready").is_null());
    }
}

TEST(a_silent_session_times_out_before_start_and_releases_its_lane_after) {
    auto rec = std::make_shared<RecordingTenant>();
    Fixture f(Fixture::Tweak{ [rec](ServerOptions& o, Deps&) {
        o.duplex_test_tenants = true;
        o.duplex_idle_s = 1;
        o.duplex_tenants.push_back(rec);
    } });
    {
        Session s(f);
        const json e = s.conn.wait_for("error", 0, 5000);
        REQUIRE(!e.is_null());
        CHECK_EQ(e.value("code", std::string()), std::string("timeout"));
        CHECK(s.conn.wait_closed());
    }
    {
        Session s(f);
        s.conn.client(start_json("record"));
        REQUIRE(!s.conn.wait_for("ready").is_null());
        const json e = s.conn.wait_for("error", 0, 5000);
        REQUIRE(!e.is_null());
        CHECK_EQ(e.value("code", std::string()), std::string("timeout"));
        CHECK(s.conn.wait_closed());
        s.join();
        std::lock_guard<std::mutex> lk(rec->m);
        CHECK_EQ(rec->closes, 1);
    }
}

TEST(start_max_minutes_times_out_and_releases_the_lane) {
    auto rec = std::make_shared<RecordingTenant>();
    Fixture f(Fixture::Tweak{ [rec](ServerOptions& o, Deps&) {
        o.duplex_test_tenants = true;
        o.duplex_minute_ms = 50;
        o.duplex_tenants.push_back(rec);
    } });
    Session s(f);
    json j = start_json("record");
    j["max_minutes"] = 1;
    s.conn.client(j);
    REQUIRE(!s.conn.wait_for("ready").is_null());
    const json e = s.conn.wait_for("error", 0, 5000);
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("code", std::string()), std::string("timeout"));
    CHECK(s.conn.wait_closed());
    s.join();
    std::lock_guard<std::mutex> lk(rec->m);
    CHECK_EQ(rec->closes, 1);
}

TEST(a_lane_is_released_when_the_client_drops_the_socket) {
    auto rec = std::make_shared<RecordingTenant>();
    Fixture f(Fixture::Tweak{ [rec](ServerOptions& o, Deps&) {
        o.duplex_test_tenants = true;
        o.duplex_tenants.push_back(rec);
    } });
    Session s(f);
    s.conn.client(start_json("record"));
    REQUIRE(!s.conn.wait_for("ready").is_null());
    s.conn.binary(frame(0, samples(0)));
    CHECK(!s.conn.wait_binary(0).empty());
    s.conn.hangup();
    s.join();
    std::lock_guard<std::mutex> lk(rec->m);
    CHECK_EQ(rec->closes, 1);
}

RAD_TEST_MAIN()
