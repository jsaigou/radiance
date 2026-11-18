/* duplex.cpp -- the engine side of docs/DUPLEX-ENGINE.md.
 *
 * A client owns the audio clock and sends one 3200-sample frame at a time; the engine answers each
 * frame with exactly one step-shaped binary message, in order. The endpoint owns the wire, the
 * per-tenant lane limit and the lane's lifetime; a DuplexTenant (duplex.h) owns whatever state the
 * conversation needs. This file knows no model.
 *
 * Threads follow the other socket routes: the connection's thread reads and queues, one session
 * thread owns the lane and sends every reply, so a lane's state has one owner and its messages go
 * out in the order they happen.
 *
 * The two test tenants are registered only under --duplex-test-tenants. With the flag off /info
 * lists no tenants and a start names one that is unavailable, which is what the contract's test
 * tenants are for: exercising the seam without a model.
 */
#include "server.h"
#include "server_priv.h"
#include "oai.h"
#include "mm/media.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace rad {
namespace server {

/* FNV-1a 64-bit, pinned so replay is independent of the standard library's std::hash. The
 * constants are FNV's published offset basis and prime; tests/duplex_seed_test.cpp pins the
 * "a" and "foobar" vectors. */
uint64_t duplex_seed(const std::string& session_id) {
    constexpr uint64_t kOffset = 14695981039346656037ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t h = kOffset;
    for (unsigned char c : session_id) {
        h ^= (uint64_t)c;
        h *= kPrime;
    }
    return h;
}

namespace {

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

constexpr int32_t  kSampleRate = 24000;
constexpr size_t   kSamples = 3200;
constexpr size_t   kStepBytes = 4 + kSamples * sizeof(float);
/* The start is control JSON, not audio: a megabyte is already far past any card or reference
 * description a tenant should need. */
constexpr size_t   kMaxStartBytes = 1u << 20;
/* A reference chunk stays well under the shared WebSocket transport's message ceiling. */
constexpr size_t   kMaxReferenceChunkBytes = 1u << 20;
/* The client messages a session may have queued and not yet taken in. */
constexpr size_t   kMaxQueuedBytes = 16u << 20;

size_t reference_limit_bytes(int64_t mib) {
    return (size_t)std::max<int64_t>(1, mib) << 20;
}

uint32_t read_le32(const char* p) {
    return (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) |
           ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
}

void append_le32(std::string* out, uint32_t v) {
    out->push_back((char)(v & 0xff));
    out->push_back((char)((v >> 8) & 0xff));
    out->push_back((char)((v >> 16) & 0xff));
    out->push_back((char)((v >> 24) & 0xff));
}

/* The wire's step shape: the echoed seq, flags, text length and delta, then any audio. */
std::string step_message(uint32_t seq, const DuplexReply& r) {
    std::string out;
    out.reserve(9 + r.text.size() + r.audio.size() * sizeof(float));
    append_le32(&out, seq);
    const uint8_t flags = (uint8_t)((r.audio.empty() ? 0 : 1) | (r.text.empty() ? 0 : 2));
    out.push_back((char)flags);
    append_le32(&out, (uint32_t)r.text.size());
    out += r.text;
    for (float f : r.audio) {
        uint32_t u = 0;
        std::memcpy(&u, &f, sizeof u);
        append_le32(&out, u);
    }
    return out;
}

json error_event(const char* code, const std::string& message) {
    json o;
    o["type"] = "error";
    o["code"] = code;
    o["message"] = message;
    o["fatal"] = true;
    return o;
}

/* A bridge deadline: the engine did not answer a step or flush in time and the lane is closed. */
struct DuplexTimeout : std::exception {
    const char* what() const noexcept override { return "the duplex engine timed out"; }
};

/* ------------------------------------------------------------------ echo / echo1 */

struct EchoLane : DuplexLane {
    explicit EchoLane(int delay) : delay_(delay) {}
    DuplexReply step(uint32_t, const std::vector<float>& samples) override {
        DuplexReply r;
        if (delay_ == 0) {
            r.audio = samples;
        } else if (held_) {
            r.audio = std::move(*held_);
            held_ = samples;
        } else {
            held_ = samples;
        }
        return r;
    }
    DuplexReply flush() override {
        DuplexReply r;
        if (held_) r.audio = std::move(*held_);
        held_.reset();
        return r;
    }
    void close() override { held_.reset(); }
    int delay_ = 0;
    std::optional<std::vector<float>> held_;
};

struct EchoTenant : DuplexTenant {
    EchoTenant(std::string n, int delay, int lanes)
        : info_{ std::move(n), false, std::max(1, lanes) }, delay_(delay) {}
    DuplexTenantInfo info() const override { return info_; }
    int open(const DuplexStart&, std::unique_ptr<DuplexLane>* out, std::string*) override {
        out->reset(new EchoLane(delay_));
        return RAD_OK;
    }
    DuplexTenantInfo info_;
    int delay_ = 0;
};

/* ------------------------------------------------------------------ the engine's tenant */

/* The start's card as text: a string as it is, an object's string "text", anything else as its
 * compact JSON. */
std::string card_text(const DuplexJson& card) {
    if (card.is_null()) return std::string();
    if (card.is_string()) return card.get<std::string>();
    if (card.is_object()) {
        if (card.empty()) return std::string();
        const auto it = card.find("text");
        if (it != card.end() && it->is_string()) return it->get<std::string>();
    }
    return card.dump();
}

/* One lane on the engine's IDuplexGen. Engine replies are synchronous here: the driver's ticket
 * is waited on by the bridge, and the server's session thread owns this lane. */
struct EngineDuplexLane final : DuplexLane {
    IDuplexGen* gen = nullptr;
    ITokenizer* tok = nullptr;
    int32_t     lane = -1;
    uint64_t    last_seq = 0;
    bool        want_text = false;   /* start.text and the tenant's text capability */

    DuplexReply map(const DuplexGenReply& r) const {
        if (r.status < 0) throw std::runtime_error("the duplex engine failed the lane");
        DuplexReply out;
        if (r.status == 0) {
            out.audio = r.audio;
            for (float& x : out.audio) {
                /* The plugin cannot clamp (no constant-bound op); the reference clamps every
                 * decoded frame to [-1, 1] (.haru-ref/session.py:163). One documented rule for
                 * non-finite values as well, so the wire never carries NaN or infinity. */
                if (std::isnan(x)) x = 0.0f;
                else if (std::isinf(x)) x = std::copysign(1.0f, x);
                else x = std::clamp(x, -1.0f, 1.0f);
            }
        } else if (r.status == 1) {
            /* A silent frame: the wire carries flags = 0. */
        } else {
            throw std::runtime_error("the duplex engine returned an invalid status");
        }
        if (r.host_text) {
            /* Already detokenized by the host source; an empty delta is a valid reply with the
             * text bit clear. */
            if (want_text) out.text = r.text;
        } else if (r.token >= 0) {
            /* Plugin-sampling tenants keep the server's per-token decode. */
            std::string text;
            if (!gen->decode_token(r.token, &text) && tok)
                text = tok->decode(std::vector<int32_t>{ r.token });
            out.text = std::move(text);
        }
        return out;
    }

    DuplexReply step(uint32_t seq, const std::vector<float>& samples) override {
        DuplexGenReply r;
        std::string why;
        const int st = gen->step(lane, samples, seq, &r, &why);
        if (st == kDuplexTimeout) throw DuplexTimeout();
        if (st < 0) throw std::runtime_error("the duplex engine failed the step");
        last_seq = seq;
        return map(r);
    }
    DuplexReply flush() override {
        DuplexGenReply r;
        std::string why;
        const int st = gen->flush(lane, last_seq, &r, &why);
        if (st == kDuplexTimeout) throw DuplexTimeout();
        if (st < 0) throw std::runtime_error("the duplex engine failed the flush");
        return map(r);
    }
    void close() override { gen->close(lane); }
};

struct EngineDuplexTenant final : DuplexTenant {
    IDuplexGen* gen = nullptr;
    ITokenizer* tok = nullptr;
    DuplexGenInfo inf;

    DuplexTenantInfo info() const override {
        DuplexTenantInfo i;
        i.name = inf.name.empty() ? "duplex" : inf.name;
        i.text = inf.text;
        i.max_lanes = (int)std::max<int64_t>(1, std::min<int64_t>(inf.max_lanes, INT32_MAX));
        return i;
    }

    int open(const DuplexStart& start, std::unique_ptr<DuplexLane>* out,
             std::string* why) override {
        auto r = std::make_unique<DuplexGenRequest>();
        const std::string card = card_text(start.card);
        std::vector<float> ref_audio;
        std::vector<int32_t> ref_cu;
        int64_t ref_frames = 0;
        if (!start.reference.empty()) {
            if (inf.max_ref_frames <= 0 || inf.ref_rate <= 0 || inf.ref_hop <= 0) {
                if (why) *why = "this tenant takes no reference audio";
                return RAD_E_INVAL;
            }
            if (inf.tmpl.find("{references}") == std::string::npos) {
                if (why) *why = "this tenant's prompt has no reference slot";
                return RAD_E_INVAL;
            }
            const int64_t hop = inf.ref_hop;
            const int64_t max_samples = inf.max_ref_frames * hop;
            std::vector<float> decoded;
            std::string dwhy;
            if (mm::decode_audio(start.reference.data(), start.reference.size(),
                                 (int32_t)inf.ref_rate, max_samples + 1, &decoded, &dwhy) < 0) {
                if (why) *why = "the reference audio could not be decoded";
                return RAD_E_INVAL;
            }
            if (decoded.empty()) {
                if (why) *why = "the reference audio holds no samples";
                return RAD_E_INVAL;
            }
            if ((int64_t)decoded.size() > max_samples) {
                if (why) *why = "the reference audio is longer than this tenant takes";
                return RAD_E_INVAL;
            }
            /* Keep the true sample count: the last frame may be partial (a 44,000-sample
             * reference is 14 frames, the last 2,400 samples). Zero-padding the audio to a whole
             * hop is not the reference's per-conv partial-hop rule and changes the last latent. */
            ref_audio = std::move(decoded);
            ref_cu = { 0, (int32_t)ref_audio.size() };
            ref_frames = ((int64_t)ref_audio.size() + hop - 1) / hop;
        }
        if (!tok) {
            if (why) *why = "the server has no tokenizer for this tenant";
            return RAD_E_STATE;
        }
        auto encode = [&](const std::string& t, bool parse_special) {
            const auto ids = tok->encode(t, false, parse_special);
            r->tokens.insert(r->tokens.end(), ids.begin(), ids.end());
            r->ref_frame.insert(r->ref_frame.end(), ids.size(), -1);
        };
        static const std::string kCard = "{card}", kRefs = "{references}";
        for (size_t at = 0; at <= inf.tmpl.size();) {
            const size_t a = inf.tmpl.find(kCard, at), b = inf.tmpl.find(kRefs, at);
            const size_t p = std::min(a, b);
            encode(inf.tmpl.substr(at, p == std::string::npos ? std::string::npos : p - at), true);
            if (p == std::string::npos) break;
            if (p == b) {
                for (int64_t f = 0; f < ref_frames; ++f) {
                    r->tokens.push_back(0);
                    r->ref_frame.push_back((int32_t)f);
                }
                at = p + kRefs.size();
            } else {
                encode(card, false);
                at = p + kCard.size();
            }
        }
        if ((int64_t)r->tokens.size() > inf.max_prompt) {
            if (why) *why = "the prompt is longer than this tenant takes";
            return RAD_E_INVAL;
        }
        r->ref_audio = std::move(ref_audio);
        r->ref_cu = std::move(ref_cu);
        r->seed = duplex_seed(start.session_id);
        int32_t lane = -1;
        const int st = gen->open(std::move(r), &lane, why);
        if (st < 0 || lane < 0) return st < 0 ? st : RAD_E_STATE;
        auto l = std::make_unique<EngineDuplexLane>();
        l->gen = gen;
        l->tok = tok;
        l->lane = lane;
        l->want_text = start.text && inf.text;
        out->reset(l.release());
        return RAD_OK;
    }
};

std::shared_ptr<DuplexTenant> make_engine_duplex_tenant(IDuplexGen* gen, ITokenizer* tok) {
    if (!gen) return nullptr;
    auto t = std::make_shared<EngineDuplexTenant>();
    t->gen = gen;
    t->tok = tok;
    t->inf = gen->info();
    return t;
}

}  /* namespace */

std::shared_ptr<DuplexTenant> make_echo_tenant(int max_lanes) {
    return std::make_shared<EchoTenant>("echo", 0, max_lanes);
}

std::shared_ptr<DuplexTenant> make_echo1_tenant(int max_lanes) {
    return std::make_shared<EchoTenant>("echo1", 1, max_lanes);
}

/* ------------------------------------------------------------------ registration */

void Server::init_duplex_tenants() {
    auto add = [this](std::shared_ptr<DuplexTenant> t) {
        if (!t) return;
        const std::string name = t->info().name;
        for (const auto& s : duplex_tenants_)
            if (s->tenant->info().name == name) return;
        duplex_tenants_.emplace_back(new DuplexTenantSlot(std::move(t)));
    };
    add(make_engine_duplex_tenant(deps_.duplex, deps_.tok));
    for (const auto& t : opt_.duplex_tenants) add(t);
    if (opt_.duplex_test_tenants) {
        const int lanes = (int)std::max<int64_t>(1, opt_.duplex_max_lanes);
        add(make_echo_tenant(lanes));
        add(make_echo1_tenant(lanes));
    }
}

/* ------------------------------------------------------------------ GET /duplex/v1/info */

HttpResponse Server::handle_duplex_info(const HttpRequest& req) {
    if (!api_key_ok(req.header("Authorization"), req.header("X-Api-Key"),
                    req.header("Sec-WebSocket-Protocol"), opt_.api_key)) {
        ApiError e;
        e.status = 401;
        e.type = "authentication_error";
        e.message = "invalid api key";
        return error_response(e, req.peer);
    }
    json o;
    o["contract"] = "0";
    o["engine"] = "radiance";
    o["engine_version"] = opt_.version;
    json tenants = json::array();
    for (const auto& s : duplex_tenants_) {
        const DuplexTenantInfo i = s->tenant->info();
        json t;
        t["name"] = i.name;
        t["text"] = i.text;
        t["max_lanes"] = std::max(1, i.max_lanes);
        tenants.push_back(std::move(t));
    }
    o["tenants"] = std::move(tenants);
    json frame;
    frame["sample_rate"] = kSampleRate;
    frame["samples"] = kSamples;
    frame["format"] = "f32le";
    o["frame"] = std::move(frame);
    o["max_reference_bytes"] = reference_limit_bytes(opt_.duplex_max_reference_mib);
    o["ready"] = true;
    return HttpResponse::json_body(200, json_dump(o));
}

/* ------------------------------------------------------------------ the session */

struct Server::DuplexSession {
    std::string id;
    std::string peer;

    /* The connection's thread to the session's: messages, and whether the peer is gone. */
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::pair<std::string, bool>> inbox;   /* (message, binary) */
    size_t inbox_bytes = 0;
    bool   peer_gone = false;
    bool   overflow = false;
    size_t reference_limit = 0;

    /* Everything below is the session thread's alone. */
    enum class Phase { Start, Reference, Running };
    Phase phase = Phase::Start;
    DuplexTenantSlot* slot = nullptr;
    std::unique_ptr<DuplexLane> lane;
    std::unique_ptr<DuplexStart> pending;
    size_t want_reference = 0;
    uint32_t next_seq = 0;
    uint32_t last_seq = 0;
    bool have_step = false;
    uint64_t steps = 0;
    Clock::time_point last_message;
    Clock::time_point opened;
    int64_t max_minutes = 0;

    void release() {
        if (lane) {
            try {
                lane->close();
            } catch (...) {
            }
            lane.reset();
        }
        if (slot) {
            slot->lanes.fetch_sub(1);
            slot = nullptr;
        }
        pending.reset();
    }
};

/* ------------------------------------------------------------------ the connection's thread */

void Server::handle_duplex(const HttpRequest& req, WsConn& ws) {
    auto refuse = [&](const char* code, const std::string& message, int close_code) {
        ws.send(json_dump(error_event(code, message)));
        ws.close(close_code, code);
    };
    if (!api_key_ok(req.header("Authorization"), req.header("X-Api-Key"),
                    req.header("Sec-WebSocket-Protocol"), opt_.api_key))
        return refuse("bad_request", "invalid api key", 1008);
    if (stopping_.load()) return refuse("internal", "the server is shutting down", 1001);

    DuplexSession s;
    s.id = random_id("duplex_");
    s.peer = req.peer;
    s.reference_limit = reference_limit_bytes(opt_.duplex_max_reference_mib);
    std::thread worker([this, &s, &ws] { duplex_run(s, ws); });
    struct Join {
        DuplexSession& s;
        std::thread& t;
        ~Join() {
            {
                std::lock_guard<std::mutex> lk(s.m);
                s.peer_gone = true;
            }
            s.cv.notify_one();
            if (t.joinable()) t.join();
        }
    } join{ s, worker };

    std::string msg;
    bool binary = false;
    while (ws.read(&msg, &binary)) {
        std::lock_guard<std::mutex> lk(s.m);
        if (s.inbox_bytes + msg.size() > kMaxQueuedBytes) {
            s.overflow = true;
            s.cv.notify_one();
            break;
        }
        s.inbox_bytes += msg.size();
        s.inbox.emplace_back(std::move(msg), binary);
        s.cv.notify_one();
    }
}

/* ------------------------------------------------------------------ the session's thread */

void Server::duplex_run(DuplexSession& s, WsConn& ws) {
    try {
        duplex_loop(s, ws);
    } catch (const std::exception& e) {
        /* e.what() is internal text that may quote client frames: sanitized to the log, and the
         * client gets the generic sentence, never the reason. */
        RAD_ERR("duplex %s failed: %s", s.id.c_str(), log_sanitize(e.what()).c_str());
        ws.send(json_dump(error_event("internal", "the session failed; the reason is in the server "
                                                 "log")));
        ws.close(1011, "internal error");
    }
    s.release();
    RAD_INFO("duplex %s closed from %s: %llu steps", s.id.c_str(), s.peer.c_str(),
             (unsigned long long)s.steps);
}

bool Server::duplex_loop(DuplexSession& s, WsConn& ws) {
    auto send = [&](const json& o) { ws.send(json_dump(o)); };
    auto fail = [&](const char* code, const std::string& message, int close_code) {
        send(error_event(code, message));
        ws.close(close_code, code);
        return true;
    };

    auto open_pending = [&]() -> bool {
        if (!s.slot || !s.pending) return fail("internal", "the session has no tenant", 1011);
        const bool wants_text = s.pending->text && s.slot->tenant->info().text;
        std::unique_ptr<DuplexLane> lane;
        std::string why;
        const int st = s.slot->tenant->open(*s.pending, &lane, &why);
        if (st < 0 || !lane) {
            RAD_WARN("duplex %s: the tenant could not open the lane: %s", s.id.c_str(),
                     log_sanitize(why).c_str());
            s.release();
            if (st == RAD_E_FULL)
                return fail("session_limit",
                            "the tenant has no lane or KV left; try again later", 1013);
            /* The card or the reference was refused (not audio, too long, unreadable): the client's
             * mistake, so the contract's bad_request, with fixed text that echoes nothing it sent. */
            if (st == RAD_E_INVAL)
                return fail("bad_request",
                            "the card or the reference audio was refused (unreadable, not audio, or "
                            "too long for this tenant)", 1008);
            return fail("internal", "the tenant could not open the session", 1011);
        }
        s.lane = std::move(lane);
        s.opened = Clock::now();
        s.max_minutes = s.pending->max_minutes;
        s.pending.reset();
        s.phase = DuplexSession::Phase::Running;
        s.next_seq = 0;
        s.last_seq = 0;
        s.have_step = false;
        json ready;
        ready["type"] = "ready";
        ready["text"] = wants_text;
        ready["input_rates"] = json::array({ kSampleRate });
        send(ready);
        return false;
    };

    auto take_start = [&](const std::string& msg) -> bool {
        if (msg.size() > kMaxStartBytes)
            return fail("bad_request", fmt("the start message is larger than %zu bytes",
                                           kMaxStartBytes), 1008);
        bool over = false, deep = false;
        const json j = parse_json_bounded(msg, 0, &over, &deep);
        if (j.is_discarded() || over || deep || !j.is_object() || !j.contains("type") ||
            !j["type"].is_string() || j["type"].get<std::string>() != "start")
            return fail("bad_request", "the first message is a JSON start", 1008);
        if (j.contains("contract") && (!j["contract"].is_string() ||
                                       j["contract"].get<std::string>() != "0"))
            return fail("bad_request", "this engine speaks contract \"0\"", 1008);
        if (!j.contains("tenant") || !j["tenant"].is_string() ||
            j["tenant"].get<std::string>().empty())
            return fail("bad_request", "the start names no tenant", 1008);
        const std::string tenant = j["tenant"].get<std::string>();
        bool text = false;
        if (j.contains("text")) {
            if (!j["text"].is_boolean())
                return fail("bad_request", "start.text is a boolean", 1008);
            text = j["text"].get<bool>();
        }
        int64_t max_minutes = 0;
        if (j.contains("max_minutes") && j["max_minutes"].is_number_integer())
            max_minutes = j["max_minutes"].get<int64_t>();

        uint64_t reference_bytes = 0;
        std::string reference_id, transcript;
        if (j.contains("reference")) {
            if (!j["reference"].is_object())
                return fail("bad_request", "start.reference is an object", 1008);
            const json& r = j["reference"];
            if (r.contains("bytes")) {
                if (r["bytes"].is_number_unsigned())
                    reference_bytes = r["bytes"].get<uint64_t>();
                else if (r["bytes"].is_number_integer()) {
                    const int64_t n = r["bytes"].get<int64_t>();
                    if (n < 0) return fail("bad_request", "reference.bytes is not negative", 1008);
                    reference_bytes = (uint64_t)n;
                } else {
                    return fail("bad_request", "reference.bytes is an integer", 1008);
                }
            }
            if (reference_bytes > s.reference_limit)
                return fail("bad_request", fmt("the reference audio is larger than %zu bytes",
                                               s.reference_limit), 1008);
            if (r.contains("id") && r["id"].is_string())
                reference_id = r["id"].get<std::string>();
            if (r.contains("transcript") && r["transcript"].is_string())
                transcript = r["transcript"].get<std::string>();
        }

        DuplexTenantSlot* slot = nullptr;
        for (const auto& t : duplex_tenants_)
            if (t->tenant->info().name == tenant) { slot = t.get(); break; }
        if (!slot)
            return fail("tenant_unavailable",
                        "no tenant by that name is registered; /duplex/v1/info lists them", 1008);
        const int64_t cap = std::max<int64_t>(1, slot->tenant->info().max_lanes);
        if (slot->lanes.fetch_add(1) >= cap) {
            slot->lanes.fetch_sub(1);
            return fail("session_limit",
                        fmt("this tenant holds %lld lanes at once; try again later",
                            (long long)cap), 1013);
        }
        s.slot = slot;
        s.pending.reset(new DuplexStart());
        s.pending->contract = "0";
        if (j.contains("session_id") && j["session_id"].is_string())
            s.pending->session_id = j["session_id"].get<std::string>();
        s.pending->text = text;
        s.pending->max_minutes = max_minutes;
        if (j.contains("card")) s.pending->card = j["card"];
        s.pending->reference_id = std::move(reference_id);
        s.pending->transcript = std::move(transcript);
        if (reference_bytes > 0) {
            s.want_reference = (size_t)reference_bytes;
            s.phase = DuplexSession::Phase::Reference;
            return false;
        }
        return open_pending();
    };

    auto take_reference = [&](const std::string& msg, bool binary) -> bool {
        if (!binary)
            return fail("bad_request", "the reference audio is binary messages before any other "
                                       "message", 1008);
        if (msg.size() > kMaxReferenceChunkBytes)
            return fail("bad_request",
                        fmt("a reference message is at most %zu bytes; this one is %zu",
                            kMaxReferenceChunkBytes, msg.size()), 1008);
        const size_t have = s.pending->reference.size();
        if (msg.size() > s.want_reference - have)
            return fail("bad_request", "the reference audio is longer than start stated", 1008);
        s.pending->reference.insert(s.pending->reference.end(), msg.begin(), msg.end());
        if (s.pending->reference.size() == s.want_reference) {
            s.want_reference = 0;
            return open_pending();
        }
        return false;
    };

    auto take_step = [&](const std::string& msg) -> bool {
        if (msg.size() != kStepBytes)
            return fail("bad_frame",
                        fmt("a step is %zu bytes: 4 for seq and %zu for 3200 float32 samples; "
                            "this one is %zu", kStepBytes, kSamples * sizeof(float), msg.size()),
                        1008);
        const uint32_t seq = read_le32(msg.data());
        if (seq != s.next_seq)
            return fail("bad_sequence",
                        fmt("the step's seq is %u; the next expected is %u",
                            (unsigned)seq, (unsigned)s.next_seq), 1008);
        std::vector<float> samples(kSamples);
        for (size_t i = 0; i < kSamples; ++i) {
            const uint32_t u = read_le32(msg.data() + 4 + i * sizeof(float));
            std::memcpy(&samples[i], &u, sizeof u);
        }
        DuplexReply r;
        try {
            r = s.lane->step(seq, samples);
        } catch (const DuplexTimeout&) {
            return fail("timeout", "the tenant did not answer the step in time", 1008);
        } catch (const std::exception& e) {
            RAD_ERR("duplex %s step failed: %s", s.id.c_str(), log_sanitize(e.what()).c_str());
            return fail("internal", "the tenant failed the step", 1011);
        }
        if ((!r.audio.empty() && r.audio.size() != kSamples) || r.text.size() > 0xffffffffu)
            return fail("internal", "the tenant returned an invalid step", 1011);
        ws.send_binary(step_message(seq, r));
        s.last_seq = seq;
        s.next_seq = seq + 1;
        s.have_step = true;
        ++s.steps;
        return false;
    };

    auto take_text = [&](const std::string& msg) -> bool {
        bool over = false, deep = false;
        const json j = parse_json_bounded(msg, 0, &over, &deep);
        if (j.is_discarded() || over || deep || !j.is_object() || !j.contains("type") ||
            !j["type"].is_string())
            return fail("bad_request", "a control message is a JSON object with a string type",
                        1008);
        const std::string type = j["type"].get<std::string>();
        if (type == "flush") {
            DuplexReply r;
            try {
                r = s.lane->flush();
            } catch (const DuplexTimeout&) {
                return fail("timeout", "the tenant did not answer the flush in time", 1008);
            } catch (const std::exception& e) {
                RAD_ERR("duplex %s flush failed: %s", s.id.c_str(), log_sanitize(e.what()).c_str());
                return fail("internal", "the tenant failed the flush", 1011);
            }
            if ((!r.audio.empty() && r.audio.size() != kSamples) || r.text.size() > 0xffffffffu)
                return fail("internal", "the tenant returned an invalid flush", 1011);
            ws.send_binary(step_message(s.have_step ? s.last_seq : 0, r));
            return false;
        }
        if (type == "close") {
            ws.close(1000, "session closed");
            return true;
        }
        return fail("bad_request", "this session takes start, flush, close and step frames", 1008);
    };

    auto handle = [&](const std::string& msg, bool binary) -> bool {
        if (s.phase == DuplexSession::Phase::Start) {
            if (binary) return fail("bad_request", "the first message is a JSON start", 1008);
            return take_start(msg);
        }
        if (s.phase == DuplexSession::Phase::Reference) return take_reference(msg, binary);
        if (binary) return take_step(msg);
        return take_text(msg);
    };

    const auto idle_for =
        std::chrono::seconds(std::max<int64_t>(1, opt_.duplex_idle_s));
    s.last_message = Clock::now();
    RAD_INFO("duplex %s open from %s", s.id.c_str(), s.peer.c_str());
    for (;;) {
        const auto now = Clock::now();
        auto wait = std::chrono::milliseconds(100);
        const auto idle_left = std::chrono::duration_cast<std::chrono::milliseconds>(
            idle_for - (now - s.last_message));
        if (idle_left < wait)
            wait = std::chrono::milliseconds(std::max<int64_t>(0, idle_left.count()));

        std::deque<std::pair<std::string, bool>> got;
        bool gone = false, overflow = false;
        {
            std::unique_lock<std::mutex> lk(s.m);
            if (s.inbox.empty() && !s.peer_gone && !s.overflow) s.cv.wait_for(lk, wait);
            got.swap(s.inbox);
            s.inbox_bytes = 0;
            gone = s.peer_gone;
            overflow = s.overflow;
        }
        if (!got.empty()) s.last_message = Clock::now();
        if (overflow)
            return fail("bad_request", fmt("the session fell more than %zu MiB of messages behind",
                                           kMaxQueuedBytes >> 20), 1008);
        for (const auto& m : got)
            if (handle(m.first, m.second)) return true;
        if (gone) return false;
        if (stopping_.load())
            return fail("internal", "the server is shutting down", 1001);
        if (s.max_minutes > 0) {
            const double elapsed = std::chrono::duration<double, std::milli>(
                Clock::now() - s.opened).count();
            const double limit = (double)s.max_minutes *
                                 (double)std::max<int64_t>(1, opt_.duplex_minute_ms);
            if (elapsed >= limit)
                return fail("timeout", "the session reached start.max_minutes", 1008);
        }
        if (got.empty() && Clock::now() - s.last_message >= idle_for)
            return fail("timeout",
                        fmt("the session received no message for more than %lld seconds",
                            (long long)opt_.duplex_idle_s), 1008);
    }
}

}  /* namespace server */
}  /* namespace rad */
