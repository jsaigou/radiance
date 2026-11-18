/* tests/duplex_plugin_test.cpp -- the first real run of the duplex engine wiring: the built
 * radiance binary serves tests/duplex_plugin (architecture radtestduplex) over a tiny checkpoint,
 * and /duplex/v1/info lists that plugin's tenant. No WebSocket client here; the reviewer drives
 * the session from Forge.
 */
#include "rad_test.h"
#include "tiny_model.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <signal.h>
#include <unistd.h>

using json = nlohmann::ordered_json;

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

bool write_checkpoint(const std::string& dir) {
    tiny::Tensor one{ "duplex.one", "F32", { 1 }, {} };
    one.bytes.resize(4);
    const float f = 1.0f;
    std::memcpy(one.bytes.data(), &f, 4);

    json c;
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

/* A RADIANCE_HOME with only what this run needs: libref and the test plugin. */
bool make_home(const std::string& home, std::string* why) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(home + "/kernels", ec);
    fs::create_directories(home + "/architectures", ec);
    fs::create_directories(home + "/quantizers", ec);
    if (ec) {
        *why = "could not make the temporary RADIANCE_HOME: " + ec.message();
        return false;
    }
    fs::copy_file(std::string(RAD_E2E_HOME) + "/kernels/libref.so", home + "/kernels/libref.so",
                  fs::copy_options::overwrite_existing, ec);
    if (ec) {
        *why = "could not copy libref.so: " + ec.message();
        return false;
    }
    fs::copy_file(RAD_DUPLEX_PLUGIN, home + "/architectures/radtest_duplex.so",
                  fs::copy_options::overwrite_existing, ec);
    if (ec) {
        *why = "could not copy radtest_duplex.so: " + ec.message();
        return false;
    }
    return true;
}

/* A server on a free loopback port, its output drained on a thread. */
struct Server {
    tiny::Child child;
    std::thread drain;
    std::mutex mu;
    std::condition_variable cv;
    std::string log;
    bool eof = false;
    int port = 0;
    int exit_status = -1;

    bool start(const std::string& model, const std::string& home) {
        std::vector<std::string> argv = {
            RAD_E2E_SERVER, "--model", model, "--kernels", "libref",
            "--debug-accept-reference-kernels", "--host", "127.0.0.1", "--port", "0",
            "--http-threads", "2",
        };
        std::vector<std::string> env = {
            "ROCR_VISIBLE_DEVICES=99", "RADIANCE_HOST_VRAM_MIB=512", "OMP_WAIT_POLICY=passive",
            "OMP_NUM_THREADS=2", "RADIANCE_HOME=" + home,
        };
        if (!child.start(argv, env)) return false;
        drain = std::thread([this] {
            char buf[4096];
            for (;;) {
                const ssize_t n = ::read(child.fd, buf, sizeof buf);
                if (n < 0 && errno == EINTR) continue;
                std::lock_guard<std::mutex> lk(mu);
                if (n <= 0) { eof = true; cv.notify_all(); return; }
                log.append(buf, (size_t)n);
                cv.notify_all();
            }
        });
        static const std::string kTag = "listening on http://127.0.0.1:";
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return eof || log.find(kTag) != std::string::npos; });
        const size_t at = log.find(kTag);
        if (at != std::string::npos) port = std::atoi(log.c_str() + at + kTag.size());
        if (port <= 0)
            fprintf(stderr, "    the server did not start:\n%s\n", log.c_str());
        return port > 0;
    }

    int stop() {
        if (child.pid <= 0) return exit_status;
        kill(child.pid, SIGTERM);
        if (drain.joinable()) drain.join();
        close(child.fd);
        child.fd = -1;
        std::string rest;
        exit_status = child.wait(&rest);
        return exit_status;
    }

    ~Server() {
        if (child.pid > 0) kill(child.pid, SIGKILL);
        if (drain.joinable()) drain.join();
    }
};

struct Get {
    int fd = -1;
    bool send(int port, const std::string& path) {
        fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) return false;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t)port);
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (connect(fd, (sockaddr*)&a, sizeof a) != 0) return false;
        const std::string req = "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                "Connection: close\r\n\r\n";
        for (size_t off = 0; off < req.size();) {
            const ssize_t n = ::send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
            if (n <= 0) return false;
            off += (size_t)n;
        }
        return true;
    }
    int read(std::string* body) {
        std::string raw;
        char buf[16384];
        for (;;) {
            const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            raw.append(buf, (size_t)n);
        }
        close(fd);
        fd = -1;
        const size_t h = raw.find("\r\n\r\n");
        if (raw.compare(0, 9, "HTTP/1.1 ") != 0 || h == std::string::npos) return -1;
        *body = raw.substr(h + 4);
        return std::atoi(raw.c_str() + 9);
    }
    ~Get() { if (fd >= 0) close(fd); }
};

}  /* namespace */

TEST(the_real_plugin_tenant_starts_lists_itself_and_stops_cleanly) {
    const bool keep = std::getenv("RAD_DUPLEX_KEEP") != nullptr;
    tiny::TempDir dir("duplex_plugin", keep);
    REQUIRE(dir.ok());
    REQUIRE(write_checkpoint(dir.path));
    const std::string home = dir / "home";
    std::string why;
    REQUIRE(make_home(home, &why));

    Server s;
    if (!s.start(dir.path, home)) {
        fprintf(stderr, "    %s\n", why.c_str());
        REQUIRE(false);
    }
    Get g;
    std::string body;
    REQUIRE(g.send(s.port, "/duplex/v1/info"));
    REQUIRE_EQ(g.read(&body), 200);
    const json j = json::parse(body, nullptr, false);
    REQUIRE(j.is_object());
    REQUIRE(j.contains("tenants"));
    bool found = false, echo = false;
    for (const auto& t : j["tenants"]) {
        const std::string name = t.value("name", std::string());
        found = found || name == "radtest-duplex";
        echo = echo || name == "echo" || name == "echo1";
    }
    CHECK(found);
    CHECK(!echo);
    CHECK_EQ(s.stop(), 0);
    if (keep)
        fprintf(stderr, "kept the checkpoint and RADIANCE_HOME under %s\n", dir.path.c_str());
}

RAD_TEST_MAIN()
