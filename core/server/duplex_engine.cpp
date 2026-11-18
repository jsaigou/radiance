/* core/server/duplex_engine.cpp -- wait on a DuplexDriver ticket without holding the engine. */
#include "server/duplex_engine.h"

namespace rad {
namespace server {

int EngineDuplexBridge::open(std::unique_ptr<DuplexGenRequest> r, int32_t* lane,
                             std::string* why) {
    if (!drv_) return RAD_E_STATE;
    auto q = std::make_unique<gen::DuplexRequest>();
    q->tokens = std::move(r->tokens);
    q->ref_audio = std::move(r->ref_audio);
    q->ref_cu = std::move(r->ref_cu);
    q->ref_frame = std::move(r->ref_frame);
    q->seed = r->seed;
    std::shared_ptr<gen::DuplexTicket> t;
    const int st = drv_->open(std::move(q), &t, why);
    if (st < 0) {
        if (lane) *lane = -1;
        return st;
    }
    bool timed_out = false;
    if (!t->wait(timeouts_.open_ms, &timed_out)) {
        if (t->lane() >= 0) drv_->close(t->lane());
        if (why) *why = "the duplex prompt did not finish";
        if (lane) *lane = -1;
        return RAD_E_STATE;
    }
    if (t->status() < 0) {
        if (why) *why = t->why();
        if (lane) *lane = -1;
        return t->status();
    }
    if (lane) *lane = t->lane();
    return RAD_OK;
}

int EngineDuplexBridge::step(int32_t lane, const std::vector<float>& samples, uint64_t seq,
                             DuplexGenReply* out, std::string* why) {
    if (!drv_) return RAD_E_STATE;
    std::shared_ptr<gen::DuplexTicket> t;
    const int st = drv_->step(lane, samples.data(), (int64_t)samples.size(), seq, &t, why);
    if (st < 0) return st;
    gen::DuplexReply r;
    bool timed_out = false;
    if (!t->next(&r, timeouts_.step_ms, &timed_out)) {
        if (timed_out) {
            drv_->close(lane);
            if (why) *why = "the duplex step did not finish";
            return kDuplexTimeout;
        }
        if (why) *why = t->why();
        return t->status() < 0 ? t->status() : RAD_E_STATE;
    }
    out->audio = std::move(r.audio);
    out->token = r.token;
    out->status = r.status;
    return RAD_OK;
}

int EngineDuplexBridge::flush(int32_t lane, uint64_t seq, DuplexGenReply* out,
                              std::string* why) {
    if (!drv_) return RAD_E_STATE;
    std::shared_ptr<gen::DuplexTicket> t;
    const int st = drv_->flush(lane, seq, &t, why);
    if (st < 0) return st;
    gen::DuplexReply r;
    bool timed_out = false;
    if (!t->next(&r, timeouts_.step_ms, &timed_out)) {
        if (timed_out) {
            drv_->close(lane);
            if (why) *why = "the duplex flush did not finish";
            return kDuplexTimeout;
        }
        if (why) *why = t->why();
        return t->status() < 0 ? t->status() : RAD_E_STATE;
    }
    out->audio = std::move(r.audio);
    out->token = r.token;
    out->status = r.status;
    return RAD_OK;
}

}  /* namespace server */
}  /* namespace rad */
