/* haru_harness.h -- the shared test harness for the Haru building blocks.
 *
 * The API is the contract in tests/HARU_HARNESS.md: Golden (the recorded boundary tensors),
 * Tol/Diff/compare (the tolerance arithmetic), Engine (the real radiance binary on the host
 * backend with one test plugin) and Lane (one /duplex/v1/session WebSocket). Header-only and
 * inline, so every block's test compiles it as is; the block's own plugin is the only thing that
 * differs between them.
 *
 * A step frame is 4 bytes of little-endian seq and 3200 little-endian float32 in, and a reply is
 * the echoed seq, a flags byte, a 4-byte text length, the UTF-8 text, then 3200 float32 when
 * flags bit 0 is set (docs/DUPLEX-ENGINE.md). The harness implements that client side with the
 * vendored cpp-httplib WebSocket client.
 */
#pragma once
#include "rad_test.h"
#include "tiny_model.h"

#include "cpp-httplib/httplib.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <csignal>
#include <unistd.h>

/* The block tests get these from tests/CMakeLists.txt. Empty fallbacks keep a translation unit
 * that includes the header without wiring compilable; using them fails with a clear message. */
#ifndef RAD_E2E_SERVER
#define RAD_E2E_SERVER ""
#endif
#ifndef RAD_E2E_HOME
#define RAD_E2E_HOME ""
#endif
#ifndef RAD_HARU_PLUGIN
#define RAD_HARU_PLUGIN ""
#endif
#ifndef RAD_HARU_DATA
#define RAD_HARU_DATA ""
#endif

namespace haru_test {

/* ------------------------------------------------------------------ goldens */

/* The recorded boundary tensors, loaded once from RAD_HARU_DATA "/golden.safetensors". A missing
 * file or an unknown tensor throws std::runtime_error, which rad_test reports as a failed case. */
struct Golden {
    static const Golden& get() {
        static const Golden g = load();
        return g;
    }

    bool has(const std::string& name) const {
        for (const tiny::Tensor& t : tensors)
            if (t.name == name) return true;
        return false;
    }

    std::vector<float> f32(const std::string& name) const {
        const tiny::Tensor& t = find(name);
        if (t.dtype != "F32")
            throw std::runtime_error("golden tensor '" + name + "' is " + t.dtype +
                                     ", not F32");
        const size_t n = (size_t)tiny::numel(t.shape);
        if (t.bytes.size() != n * sizeof(float))
            throw std::runtime_error("golden tensor '" + name + "' has " +
                                     std::to_string(t.bytes.size()) + " bytes, not " +
                                     std::to_string(n * sizeof(float)));
        std::vector<float> out(n);
        if (n) std::memcpy(out.data(), t.bytes.data(), n * sizeof(float));
        return out;
    }

    std::vector<int64_t> i64(const std::string& name) const {
        const tiny::Tensor& t = find(name);
        if (t.dtype != "I64")
            throw std::runtime_error("golden tensor '" + name + "' is " + t.dtype +
                                     ", not I64");
        const size_t n = (size_t)tiny::numel(t.shape);
        if (t.bytes.size() != n * sizeof(int64_t))
            throw std::runtime_error("golden tensor '" + name + "' has " +
                                     std::to_string(t.bytes.size()) + " bytes, not " +
                                     std::to_string(n * sizeof(int64_t)));
        std::vector<int64_t> out(n);
        if (n) std::memcpy(out.data(), t.bytes.data(), n * sizeof(int64_t));
        return out;
    }

    std::vector<int64_t> shape(const std::string& name) const { return find(name).shape; }

private:
    static Golden load() {
        Golden g;
        g.path_ = std::string(RAD_HARU_DATA) + "/golden.safetensors";
        g.tensors = tiny::read_safetensors(g.path_);
        if (g.tensors.empty())
            throw std::runtime_error("could not read the Haru goldens from " + g.path_ +
                                     " (is RAD_HARU_DATA right?)");
        return g;
    }

    const tiny::Tensor& find(const std::string& name) const {
        for (const tiny::Tensor& t : tensors)
            if (t.name == name) return t;
        throw std::runtime_error("the Haru goldens in " + path_ + " have no tensor '" + name +
                                 "'");
    }

    std::string path_;
    std::vector<tiny::Tensor> tensors;
};

/* ------------------------------------------------------------------ comparison */

/* An element passes when |got - want| <= atol + rtol * |want|. */
struct Tol {
    double atol = 0.0;
    double rtol = 0.0;
};

struct Diff {
    bool   ok = true;
    double max_abs = 0.0, max_rel = 0.0;   /* over all elements */
    size_t worst_index = 0;                /* the largest violation, or max_abs when ok */
    float  got = 0.0f, want = 0.0f;        /* at worst_index */

    std::string text() const {
        char buf[256];
        if (worst_index == std::numeric_limits<size_t>::max())
            return "FAIL sizes differ";
        if (ok)
            std::snprintf(buf, sizeof buf, "ok max_abs=%g max_rel=%g", max_abs, max_rel);
        else
            std::snprintf(buf, sizeof buf, "FAIL idx=%zu got=%g want=%g max_abs=%g",
                          worst_index, (double)got, (double)want, max_abs);
        return buf;
    }
};

inline Diff compare(const std::vector<float>& got, const std::vector<float>& want, Tol tol) {
    Diff d;
    if (got.size() != want.size()) {
        d.ok = false;
        d.max_abs = d.max_rel = std::numeric_limits<double>::infinity();
        d.worst_index = std::numeric_limits<size_t>::max();
        return d;
    }
    bool failed = false;
    double worst_violation = -1.0;
    size_t worst_violation_at = 0, worst_abs_at = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = (double)got[i], w = (double)want[i];
        const bool nan = std::isnan(g) || std::isnan(w);
        const bool equal = !nan && g == w;
        const double ad = nan ? std::numeric_limits<double>::infinity()
                              : (equal ? 0.0 : std::fabs(g - w));
        const double rd = nan ? std::numeric_limits<double>::infinity()
                              : (w == 0.0
                                     ? (ad == 0.0 ? 0.0
                                                  : std::numeric_limits<double>::infinity())
                                     : ad / std::fabs(w));
        if (ad > d.max_abs) {
            d.max_abs = ad;
            worst_abs_at = i;
        }
        if (rd > d.max_rel) d.max_rel = rd;
        const double scaled = tol.rtol == 0.0 ? 0.0 : tol.rtol * std::fabs(w);
        const double limit = tol.atol + scaled;
        const bool finite = std::isfinite(g) && std::isfinite(w);
        const bool pass = !nan && (equal || (finite && ad <= limit));
        if (!pass) {
            failed = true;
            const double violation = (nan || !finite)
                                         ? std::numeric_limits<double>::infinity()
                                         : ad - limit;
            if (violation > worst_violation) {
                worst_violation = violation;
                worst_violation_at = i;
            }
        }
    }
    d.ok = !failed;
    const size_t at = d.ok ? worst_abs_at : worst_violation_at;
    d.worst_index = at;
    if (!got.empty()) {
        d.got = got[at];
        d.want = want[at];
    }
    return d;
}

/* REQUIRE-style: prints the Diff and leaves the case, because the caller's next line usually
 * indexes the same tensors. The label is the stringified tensor arguments. The tolerance is
 * variadic so an inline `Tol{atol, rtol}` (its comma would otherwise split the macro arguments)
 * is accepted as well as a Tol variable. */
#define HARU_REQUIRE_CLOSE(got, want, ...)                                                   \
    do {                                                                                     \
        ++::radtest::checks();                                                               \
        const std::vector<float> _hg = (got);                                                \
        const std::vector<float> _hw = (want);                                               \
        const ::haru_test::Diff _hd = ::haru_test::compare(_hg, _hw, (__VA_ARGS__));         \
        if (!_hd.ok) {                                                                       \
            ::radtest::fail(__FILE__, __LINE__,                                              \
                std::string("HARU_REQUIRE_CLOSE(" #got ", " #want "): ") + _hd.text());      \
            return;                                                                          \
        }                                                                                    \
    } while (0)

/* ------------------------------------------------------------------ the engine */

/* The real radiance binary on the host backend (libref), with a temporary RADIANCE_HOME holding
 * only libref and the test plugin, and a model directory passed explicitly. */
class Engine {
public:
    static constexpr size_t kSamples = 3200;

    /* The block tests' engine: RAD_HARU_PLUGIN and RAD_HARU_DATA "/haru_tiny". */
    static std::unique_ptr<Engine> start(const std::vector<std::string>& extra_args,
                                         std::string* why) {
        return start_with(RAD_HARU_PLUGIN, std::string(RAD_HARU_DATA) + "/haru_tiny",
                          extra_args, why);
    }

    /* An explicit plugin and model directory, for the harness' own self-test and for a package
     * whose checkpoint lives somewhere else. */
    static std::unique_ptr<Engine> start_with(const std::string& plugin_so,
                                              const std::string& model_dir,
                                              const std::vector<std::string>& extra_args,
                                              std::string* why) {
        std::unique_ptr<Engine> e(new Engine());
        std::string w;
        if (!e->launch(plugin_so, model_dir, extra_args, &w)) {
            if (why) *why = w;
            return nullptr;
        }
        return e;
    }

    int port() const { return port_; }

    /* The child's output so far, for failure messages. */
    std::string log() const {
        std::lock_guard<std::mutex> lk(mu_);
        return log_;
    }

    /* Stop the child and return its exit status (128 + signal, as wait(2) reports it). The
     * destructor does this too; calling it earlier is how a test checks a clean exit. */
    int stop() {
        if (child_.pid <= 0) return exit_status_;
        kill(child_.pid, SIGTERM);
        if (drain_.joinable()) drain_.join();
        if (child_.fd >= 0) {
            ::close(child_.fd);
            child_.fd = -1;
        }
        std::string rest;
        exit_status_ = child_.wait(&rest);
        return exit_status_;
    }

    ~Engine() { stop(); }

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

private:
    Engine() = default;

    static std::string path_leaf(const std::string& path) {
        const size_t slash = path.find_last_of('/');
        return slash == std::string::npos ? path : path.substr(slash + 1);
    }

    static bool make_home(const std::string& home, const std::string& plugin_so,
                          std::string* why) {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(home + "/kernels", ec);
        fs::create_directories(home + "/architectures", ec);
        fs::create_directories(home + "/quantizers", ec);
        if (ec) {
            *why = "could not make the temporary RADIANCE_HOME " + home + ": " + ec.message();
            return false;
        }
        fs::copy_file(std::string(RAD_E2E_HOME) + "/kernels/libref.so",
                      home + "/kernels/libref.so", fs::copy_options::overwrite_existing, ec);
        if (ec) {
            *why = "could not copy libref.so from " + std::string(RAD_E2E_HOME) + ": " +
                   ec.message();
            return false;
        }
        fs::copy_file(plugin_so, home + "/architectures/" + path_leaf(plugin_so),
                      fs::copy_options::overwrite_existing, ec);
        if (ec) {
            *why = "could not copy the test plugin " + plugin_so + ": " + ec.message();
            return false;
        }
        return true;
    }

    bool launch(const std::string& plugin_so, const std::string& model_dir,
                const std::vector<std::string>& extra_args, std::string* why) {
        const bool keep = std::getenv("RAD_DUPLEX_KEEP") != nullptr;
        dir_.reset(new tiny::TempDir("haru_harness", keep));
        if (!dir_->ok()) {
            *why = "could not make a temporary directory for the engine";
            return false;
        }
        const std::string home = dir_->path + "/home";
        if (!make_home(home, plugin_so, why)) return false;

        std::vector<std::string> argv = {
            RAD_E2E_SERVER, "--model", model_dir, "--kernels", "libref",
            "--debug-accept-reference-kernels", "--host", "127.0.0.1", "--port", "0",
            "--http-threads", "4",
        };
        argv.insert(argv.end(), extra_args.begin(), extra_args.end());
        const std::vector<std::string> env = {
            "ROCR_VISIBLE_DEVICES=99", "RADIANCE_HOST_VRAM_MIB=512",
            "OMP_WAIT_POLICY=passive", "OMP_NUM_THREADS=2", "RADIANCE_HOME=" + home,
        };
        if (!child_.start(argv, env)) {
            *why = std::string("could not start ") + RAD_E2E_SERVER;
            return false;
        }
        drain_ = std::thread([this] {
            char buf[4096];
            for (;;) {
                const ssize_t n = ::read(child_.fd, buf, sizeof buf);
                if (n < 0 && errno == EINTR) continue;
                std::lock_guard<std::mutex> lk(mu_);
                if (n <= 0) {
                    eof_ = true;
                    cv_.notify_all();
                    return;
                }
                log_.append(buf, (size_t)n);
                cv_.notify_all();
            }
        });

        static const std::string kTag = "listening on http://127.0.0.1:";
        {
            std::unique_lock<std::mutex> lk(mu_);
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(120);
            while (!eof_ && log_.find(kTag) == std::string::npos) {
                if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) break;
            }
            const size_t at = log_.find(kTag);
            if (at != std::string::npos)
                port_ = std::atoi(log_.c_str() + at + kTag.size());
        }
        if (port_ <= 0) {
            *why = "the engine did not start (" + std::string(RAD_E2E_SERVER) + " --model " +
                   model_dir + "):\n" + log();
            return false;
        }
        return true;
    }

    tiny::Child child_;
    std::thread drain_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::string log_;
    bool eof_ = false;
    int port_ = 0;
    int exit_status_ = -1;
    std::unique_ptr<tiny::TempDir> dir_;
};

/* ------------------------------------------------------------------ a lane */

/* One duplex lane over WebSocket. The engine answers each 3200-sample frame with exactly one
 * binary reply, in seq order; the text channel and status are parsed and skipped. */
class Lane {
public:
    static std::unique_ptr<Lane> open(Engine& e, const std::string& tenant, std::string* why,
                                      const std::string& card = "",
                                      const std::vector<uint8_t>& reference = {}) {
        std::unique_ptr<Lane> lane(new Lane());
        lane->engine_ = &e;
        const std::string url = "ws://127.0.0.1:" + std::to_string(e.port()) +
                                "/duplex/v1/session";
        lane->ws_.reset(new httplib::ws::WebSocketClient(url));
        lane->ws_->set_connection_timeout(30, 0);
        lane->ws_->set_read_timeout(120, 0);
        const httplib::ws::Result r = lane->ws_->connect();
        if (!r) {
            std::string w = "could not connect to " + url + ": " +
                            httplib::to_string(r.error());
            if (r.status() >= 0) w += " (HTTP " + std::to_string(r.status()) + ")";
            if (why) *why = w + "\nengine log:\n" + e.log();
            return nullptr;
        }

        static std::atomic<uint64_t> counter{ 0 };
        nlohmann::json start;
        start["type"] = "start";
        start["contract"] = "0";
        start["session_id"] = "haru-harness-" + std::to_string(++counter);
        start["tenant"] = tenant;
        start["text"] = true;
        start["max_minutes"] = 30;
        if (!card.empty()) start["card"] = card;
        nlohmann::json ref;
        ref["id"] = "";
        ref["bytes"] = reference.size();
        ref["transcript"] = "";
        start["reference"] = ref;

        const std::string text = start.dump();
        if (!lane->ws_->send(text)) {
            if (why) *why = "the connection closed before the start was sent\nengine log:\n" +
                            e.log();
            return nullptr;
        }
        const size_t kChunk = 1u << 20;
        for (size_t off = 0; off < reference.size(); off += kChunk) {
            const size_t n = std::min(kChunk, reference.size() - off);
            if (!lane->ws_->send((const char*)reference.data() + off, n)) {
                if (why)
                    *why = "the connection closed while the reference audio was sent\nengine "
                           "log:\n" + e.log();
                return nullptr;
            }
        }

        for (;;) {
            std::string msg;
            const httplib::ws::ReadResult rr = lane->ws_->read(msg);
            if (rr == httplib::ws::Timeout) {
                if (why)
                    *why = "the engine did not answer the start within 120 s\nengine log:\n" +
                           e.log();
                return nullptr;
            }
            if (rr == httplib::ws::Fail) {
                if (why)
                    *why = "the connection closed before the engine sent ready\nengine log:\n" +
                           e.log();
                return nullptr;
            }
            if (rr == httplib::ws::Binary) {
                if (why)
                    *why = "the engine sent a binary frame before ready\nengine log:\n" +
                           e.log();
                return nullptr;
            }
            const nlohmann::json j = nlohmann::json::parse(msg, nullptr, false);
            if (j.is_discarded() || !j.is_object() || !j.contains("type") ||
                !j["type"].is_string()) {
                if (why)
                    *why = "the engine sent a text frame that is not a JSON control message: " +
                           msg + "\nengine log:\n" + e.log();
                return nullptr;
            }
            const std::string type = j["type"].get<std::string>();
            if (type == "ready") {
                lane->open_ = true;
                lane->next_seq_ = 0;
                return lane;
            }
            if (type == "error") {
                if (why)
                    *why = "the engine refused the lane: " + error_text(j) +
                           "\nengine log:\n" + e.log();
                return nullptr;
            }
            /* Anything else before ready is a diagnostic this harness does not need. */
        }
    }

    /* One frame in (shorter input is zero padded), one reply out (3200 floats; a silent reply is
     * all zeros). false + why on any failure. */
    bool step(const std::vector<float>& in, std::vector<float>* out, std::string* why) {
        if (!ws_ || !open_)
            return fail("the lane is closed; open it before stepping", why);
        if (in.size() > Engine::kSamples)
            return fail("a step carries at most " + std::to_string(Engine::kSamples) +
                            " samples; this one is " + std::to_string(in.size()), why);

        std::string frame;
        frame.reserve(4 + Engine::kSamples * sizeof(float));
        put_le32(&frame, next_seq_);
        for (size_t i = 0; i < Engine::kSamples; ++i) {
            uint32_t u = 0;
            if (i < in.size()) std::memcpy(&u, &in[i], sizeof u);
            put_le32(&frame, u);
        }
        if (!ws_->send(frame.data(), frame.size()))
            return fail("the connection closed while sending step seq " +
                            std::to_string(next_seq_), why);

        const uint32_t seq = next_seq_;
        for (;;) {
            std::string msg;
            const httplib::ws::ReadResult rr = ws_->read(msg);
            if (rr == httplib::ws::Timeout)
                return fail("the engine did not reply to step seq " + std::to_string(seq) +
                                " within 120 s", why);
            if (rr == httplib::ws::Fail)
                return fail("the connection closed before the reply to step seq " +
                                std::to_string(seq), why);
            if (rr == httplib::ws::Text) {
                const nlohmann::json j = nlohmann::json::parse(msg, nullptr, false);
                if (j.is_discarded() || !j.is_object() || !j.contains("type") ||
                    !j["type"].is_string())
                    return fail("the engine sent a text frame that is not a JSON control "
                                "message: " + msg, why);
                if (j["type"].get<std::string>() == "error")
                    return fail("the engine failed the lane: " + error_text(j), why);
                /* A meta frame is per-frame diagnostics, not the step's reply. */
                continue;
            }
            if (msg.size() < 9)
                return fail("the engine sent a step reply shorter than 9 bytes", why);
            const uint32_t got_seq = read_le32(msg.data());
            if (got_seq != seq)
                return fail("the engine replied to step seq " + std::to_string(got_seq) +
                                "; the next expected is " + std::to_string(seq), why);
            const uint8_t flags = (uint8_t)msg[4];
            const uint32_t text_len = read_le32(msg.data() + 5);
            if (msg.size() < 9 + (size_t)text_len)
                return fail("the step reply announces " + std::to_string(text_len) +
                                " text bytes but is only " + std::to_string(msg.size()) +
                                " bytes", why);
            const size_t audio_at = 9 + (size_t)text_len;
            const size_t audio_bytes = Engine::kSamples * sizeof(float);
            if (flags & 1) {
                if (msg.size() != audio_at + audio_bytes)
                    return fail("the step reply is " + std::to_string(msg.size()) +
                                    " bytes; with 3200 samples it should be " +
                                    std::to_string(audio_at + audio_bytes), why);
            } else if (msg.size() != audio_at) {
                return fail("the silent step reply carries " +
                                std::to_string(msg.size() - audio_at) + " trailing bytes", why);
            }
            if (out) {
                out->assign(Engine::kSamples, 0.0f);
                if (flags & 1)
                    for (size_t i = 0; i < Engine::kSamples; ++i) {
                        const uint32_t u = read_le32(msg.data() + audio_at + i * sizeof(float));
                        std::memcpy(&(*out)[i], &u, sizeof u);
                    }
            }
            ++next_seq_;
            return true;
        }
    }

    /* A tensor of any length <= 3200 in, the first n_out floats of the reply out. */
    bool run(const std::vector<float>& tensor, size_t n_out, std::vector<float>* out,
             std::string* why) {
        if (tensor.size() > Engine::kSamples)
            return fail("a step carries at most " + std::to_string(Engine::kSamples) +
                            " samples; this tensor is " +
                            std::to_string(tensor.size()), why);
        if (n_out > Engine::kSamples)
            return fail("a reply has " + std::to_string(Engine::kSamples) + " samples; " +
                            std::to_string(n_out) + " were asked for", why);
        std::vector<float> frame;
        if (!step(tensor, &frame, why)) return false;
        if (out) out->assign(frame.begin(), frame.begin() + (std::ptrdiff_t)n_out);
        return true;
    }

    void close() {
        if (!ws_) return;
        if (open_) {
            const std::string bye = "{\"type\":\"close\"}";
            ws_->send(bye);
            open_ = false;
            ws_->close();
        }
        ws_.reset();
    }

    ~Lane() { close(); }

    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;

private:
    Lane() = default;

    static void put_le32(std::string* out, uint32_t v) {
        out->push_back((char)(v & 0xff));
        out->push_back((char)((v >> 8) & 0xff));
        out->push_back((char)((v >> 16) & 0xff));
        out->push_back((char)((v >> 24) & 0xff));
    }

    static uint32_t read_le32(const char* p) {
        return (uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) |
               ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24);
    }

    static std::string error_text(const nlohmann::json& j) {
        const std::string code = j.contains("code") && j["code"].is_string()
                                     ? j["code"].get<std::string>()
                                     : std::string("?");
        const std::string message = j.contains("message") && j["message"].is_string()
                                        ? j["message"].get<std::string>()
                                        : std::string();
        return code + ": " + message;
    }

    bool fail(const std::string& w, std::string* why) const {
        if (why) *why = w + "\nengine log:\n" + engine_->log();
        return false;
    }

    Engine* engine_ = nullptr;
    std::unique_ptr<httplib::ws::WebSocketClient> ws_;
    uint32_t next_seq_ = 0;
    bool open_ = false;
};

}  /* namespace haru_test */
