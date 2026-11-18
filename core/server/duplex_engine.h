/* core/server/duplex_engine.h -- the engine-backed IDuplexGen: a blocking server-facing view of
 * gen::DuplexDriver (core/gen/duplex_driver.h).
 *
 * The server's session thread calls one lane operation at a time; this class queues it on the
 * driver and waits for the engine pass to complete. Open has its own longer bound (a reference
 * prompt may be slow); step and flush have a short one because the wire client's deadline is about
 * a frame. A timeout closes the lane before returning, so the endpoint can answer fatally and
 * release it.
 */
#pragma once
#include "gen/duplex_driver.h"
#include "server/iface.h"

#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace server {

class EngineDuplexBridge final : public IDuplexGen {
public:
    struct Timeouts {
        int open_ms = 120000;
        int step_ms = 5000;
    };

    void init(gen::DuplexDriver* drv, const DuplexGenInfo& info, Timeouts timeouts) {
        drv_ = drv;
        inf_ = info;
        timeouts_ = timeouts;
    }

    const DuplexGenInfo& info() const override { return inf_; }
    int open(std::unique_ptr<DuplexGenRequest> r, int32_t* lane, std::string* why) override;
    int step(int32_t lane, const std::vector<float>& samples, uint64_t seq,
             DuplexGenReply* out, std::string* why) override;
    int flush(int32_t lane, uint64_t seq, DuplexGenReply* out, std::string* why) override;
    void close(int32_t lane) override {
        if (drv_) drv_->close(lane);
    }
    int64_t queued() const override { return drv_ ? drv_->queued() : 0; }
    int64_t running() const override { return drv_ ? drv_->running() : 0; }

private:
    gen::DuplexDriver* drv_ = nullptr;
    DuplexGenInfo      inf_{};
    Timeouts           timeouts_{};
};

}  /* namespace server */
}  /* namespace rad */
