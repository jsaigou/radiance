/* duplex_driver.cpp -- the duplex tenant's pass driver (duplex_driver.h). */
#include "duplex_driver.h"
#include "sched/mem_iface.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace rad {
namespace gen {

/* ================================================================== the ticket */
void DuplexTicket::ready() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (done_) return;
        done_ = true;
        status_ = RAD_OK;
    }
    cv_.notify_all();
}

void DuplexTicket::reply(DuplexReply&& r) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (done_ && q_.empty()) return;
        q_.push_back(std::move(r));
        done_ = true;
        status_ = RAD_OK;
    }
    cv_.notify_all();
}

void DuplexTicket::finish(int status, const std::string& why) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (done_) return;
        done_ = true;
        status_ = status;
        why_ = why;
    }
    cv_.notify_all();
}

bool DuplexTicket::wait(int ms, bool* timed_out) {
    std::unique_lock<std::mutex> lk(mu_);
    if (timed_out) *timed_out = false;
    if (!done_)
        cv_.wait_for(lk, std::chrono::milliseconds(ms), [this] { return done_; });
    if (!done_ && timed_out) *timed_out = true;
    return done_;
}

bool DuplexTicket::next(DuplexReply* out, int ms, bool* timed_out) {
    std::unique_lock<std::mutex> lk(mu_);
    if (timed_out) *timed_out = false;
    if (q_.empty() && !done_)
        cv_.wait_for(lk, std::chrono::milliseconds(ms), [this] { return !q_.empty() || done_; });
    if (!q_.empty()) {
        *out = std::move(q_.front());
        q_.pop_front();
        return true;
    }
    if (!done_ && timed_out) *timed_out = true;
    return false;
}

bool DuplexTicket::finished() const {
    std::lock_guard<std::mutex> lk(mu_);
    return done_ && q_.empty();
}

int DuplexTicket::status() const {
    std::lock_guard<std::mutex> lk(mu_);
    return status_;
}

std::string DuplexTicket::why() const {
    std::lock_guard<std::mutex> lk(mu_);
    return why_;
}

/* ================================================================== sizing */
namespace {

constexpr int64_t kAlign = 256;

struct Sizes {
    int64_t T = 0, S = 0, L = 0, NB = 0, SPF = 1;
    int64_t NG = 0;
    int64_t RS = 0, RN = 0;
};

/* The one paged full-attention group, and how many stateful groups sit beside it. */
int32_t full_group(const Program& prog, int64_t* n_state) {
    int32_t full = 0;
    int64_t ns = 0;
    for (size_t g = 1; g < prog.kv_groups.size(); ++g) {
        const int kind = prog.kv_groups[g].decl.kind;
        if (kind == RAD_KV_FULL && !full) full = (int32_t)g;
        else if (kind == RAD_KV_LINEAR || kind == RAD_KV_CONV) ++ns;
    }
    if (n_state) *n_state = ns;
    return full;
}

Sizes sizes_of(const Program& prog, int64_t block_size, int64_t max_ctx) {
    const RadDuplexDecl& d = prog.duplex;
    Sizes z;
    z.T  = std::max<int64_t>(prog.ctx.max_tok, 1);
    z.S  = std::max<int64_t>(prog.ctx.max_seqs, 1);
    z.L  = std::max<int64_t>(d.max_lanes, 1);
    z.SPF = std::max<int64_t>(d.samples_per_frame, 1);
    z.NB = block_size > 0 ? (std::max<int64_t>(max_ctx, 1) + block_size - 1) / block_size : 0;
    full_group(prog, &z.NG);
    z.RS = d.max_ref_frames > 0 ? d.max_ref_frames * d.ref_hop : 0;
    z.RN = d.max_ref_frames;
    return z;
}

int64_t carve(const Sizes& z, int64_t (*bytes)[2], int n) {
    const int64_t I = 4;
    const int64_t S = z.S, T = z.T, L = z.L;
    const int64_t want[] = {
        T * I,                                      /* R_TOK */
        T * I,                                      /* R_POS */
        (S + 1) * I,                                /* R_CU */
        T * I,                                      /* R_SLOT */
        S * z.NB * I,                               /* R_BT */
        S * I,                                      /* R_USED */
        S * I,                                      /* R_LANE */
        S * I,                                      /* R_LROW */
        S * I,                                      /* R_FRAME */
        2 * S * I,                                  /* R_SEED */
        S * I,                                      /* R_FLAGS */
        S * I,                                      /* R_CTX */
        S * I,                                      /* R_ACC */
        z.NG * S * I,                               /* R_SBT */
        z.NG * S * DuplexDriver::kMaxStateWidth * I,/* R_SIDX */
        z.RS * 4,                                   /* R_RAUD */
        z.RN > 0 ? (z.RN + 1) * I : 0,              /* R_RCU */
        z.RN > 0 ? T * I : 0,                       /* R_RFRM */
        L * z.SPF * 4,                              /* R_AIN */
        L * z.SPF * 4,                              /* R_AOUT */
        L * I,                                      /* R_TOUT */
        L * I,                                      /* R_STAT */
    };
    int64_t off = 0;
    for (int r = 0; r < n; ++r) {
        bytes[r][0] = off;
        bytes[r][1] = want[r];
        off = align_up(off + want[r], kAlign);
    }
    return off;
}

int64_t decl_numel(uint32_t rank, const int64_t* shape) {
    if (rank == 0 || rank > RAD_MAX_RANK) return 0;
    int64_t n = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (shape[i] <= 0) return 0;
        n *= shape[i];
    }
    return n;
}

/* The KV slot of position `p` in a sequence with this block table. */
int32_t slot_of(const std::vector<int32_t>& bt, int64_t p, int64_t bs) {
    const int64_t bi = p / bs;
    return bi < (int64_t)bt.size() ? (int32_t)(bt[(size_t)bi] * bs + p % bs) : -1;
}

}  /* namespace */

/* ================================================================== init and teardown */
int64_t DuplexDriver::device_bytes(const Program& prog, int64_t max_ctx) {
    if (!prog.is_duplex_tenant()) return 0;
    const int32_t fg = full_group(prog, nullptr);
    if (!fg) return 0;
    const Sizes z = sizes_of(prog, prog.kv_groups[(size_t)fg].block_size, max_ctx);
    int64_t b[R_N][2];
    return carve(z, b, R_N);
}

int DuplexDriver::init(const Program& prog, IKVManager* kv, const std::vector<RankIO>& ranks,
                       int64_t max_ctx) {
    fini();
    if (!prog.is_duplex_tenant()) return RAD_E_STATE;
    if (!kv || ranks.empty()) {
        RAD_ERR("duplex tenant: no KV manager or no rank");
        return RAD_E_INVAL;
    }
    decl_  = prog.duplex;
    name_  = prog.dup_str.name;
    kv_    = kv;
    ranks_ = ranks;

    state_groups_.clear();
    int n_full = 0;
    for (size_t g = 1; g < prog.kv_groups.size(); ++g) {
        const int kind = prog.kv_groups[g].decl.kind;
        if (kind == RAD_KV_FULL) {
            ++n_full;
        } else if (kind == RAD_KV_LINEAR || kind == RAD_KV_CONV) {
            StateGroup sg;
            sg.g = (int32_t)g;
            sg.width = std::max<int64_t>(1, kv->state_index_width((int32_t)g));
            if (sg.width > kMaxStateWidth) {
                RAD_ERR("duplex tenant: KV group %zu asks for %lld state slots a sequence; the "
                        "driver stages at most %lld", g, (long long)sg.width,
                        (long long)kMaxStateWidth);
                return RAD_E_UNSUPPORTED;
            }
            state_groups_.push_back(sg);
        } else {
            n_full = -1;
            break;
        }
    }
    kv_group_ = full_group(prog, nullptr);
    if (n_full != 1 || !kv_group_ || prog.kv_groups[(size_t)kv_group_].block_size <= 0) {
        RAD_ERR("duplex tenant: the program declares %zu KV group(s); the driver stages one "
                "paged full-attention group, the language model's, and the state and convolution "
                "groups of a hybrid one beside it", prog.kv_groups.size() - 1);
        return RAD_E_UNSUPPORTED;
    }
    block_size_ = prog.kv_groups[(size_t)kv_group_].block_size;
    max_tok_    = prog.ctx.max_tok;
    max_seqs_   = prog.ctx.max_seqs;
    max_lanes_  = decl_.max_lanes;
    max_ctx_    = max_ctx > 0 ? max_ctx : (prog.ctx.max_ctx > 0 ? prog.ctx.max_ctx
                                                               : decl_.max_prompt);
    max_ctx_    = std::max<int64_t>(max_ctx_, 1);
    max_blocks_ = (max_ctx_ + block_size_ - 1) / block_size_;
    if (max_lanes_ <= 0 || max_seqs_ < max_lanes_ || max_tok_ < decl_.max_prompt ||
        max_ctx_ < decl_.max_prompt) {
        RAD_ERR("duplex tenant '%s': %lld lanes and a prompt of up to %lld rows need max_seqs %lld "
                "and max_tok %lld, and a context bound of at least the prompt (got %lld)",
                name_.c_str(), (long long)max_lanes_, (long long)decl_.max_prompt,
                (long long)max_seqs_, (long long)max_tok_, (long long)max_ctx_);
        return RAD_E_UNSUPPORTED;
    }

    /* The declaration's handles must be the buffers it said they were, and every rank's copy of
     * audio_in must already be bound (Ctx::bind_buffers sets BufferInfo::ptr). */
    auto buf_ok = [&](rad_buf h, uint32_t dtype, int64_t numel, const char* what) {
        if (h == RAD_NULL_HANDLE || (size_t)h >= prog.buffers.size()) {
            RAD_ERR("duplex tenant '%s': %s %u is not a declared buffer", name_.c_str(), what,
                    (unsigned)h);
            return false;
        }
        const RadBufDecl& ob = prog.buffers[h].decl;
        if (ob.dtype != dtype || ob.domain != RAD_DOMAIN_DEVICE ||
            decl_numel(ob.rank, ob.shape) < numel) {
            RAD_ERR("duplex tenant '%s': %s '%s' is not a device %s buffer of at least %lld "
                    "elements", name_.c_str(), what, prog.buffers[h].name.c_str(),
                    dtype == RAD_I32 ? "i32" : "f32", (long long)numel);
            return false;
        }
        return true;
    };
    if (!buf_ok(decl_.audio_in, RAD_F32, max_lanes_ * decl_.samples_per_frame, "audio_in") ||
        !buf_ok(decl_.audio_out, RAD_F32, max_lanes_ * decl_.samples_per_frame, "audio_out") ||
        !buf_ok(decl_.status, RAD_I32, max_lanes_, "status") ||
        (decl_.text && !buf_ok(decl_.text_out, RAD_I32, max_lanes_, "text_out")))
        return RAD_E_INVAL;

    audio_in_ptr_.clear();
    for (size_t r = 0; r < ranks_.size(); ++r) {
        const Program* P = ranks_[r].program ? ranks_[r].program : &prog;
        if (!P || (size_t)decl_.audio_in >= P->buffers.size() ||
            !P->buffers[decl_.audio_in].ptr) {
            RAD_ERR("duplex tenant '%s': rank %zu's audio_in buffer is not bound; pass its Program "
                    "in RankIO", name_.c_str(), r);
            return RAD_E_STATE;
        }
        audio_in_ptr_.push_back(P->buffers[decl_.audio_in].ptr);
    }

    const Sizes z = sizes_of(prog, block_size_, max_ctx_);
    int64_t b[R_N][2];
    bytes_ = carve(z, b, R_N);
    for (int r = 0; r < R_N; ++r) {
        reg_[r].off = b[r][0];
        reg_[r].bytes = b[r][1];
        reg_[r].used = 0;
    }
    host_ = static_cast<char*>(rad_dev_alloc(bytes_, RAD_MEM_HOST_PINNED));
    if (!host_) {
        RAD_ERR("duplex tenant '%s': no pinned host memory for the pass staging (%lld bytes)",
                name_.c_str(), (long long)bytes_);
        return RAD_E_NOMEM;
    }
    std::memset(host_, 0, (size_t)bytes_);
    dev_.assign(ranks_.size(), nullptr);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r].device));
        dev_[r] = rad_dev_alloc(bytes_, RAD_MEM_DEVICE);
        if (!dev_[r]) {
            RAD_ERR("duplex tenant '%s': rank %zu has no room for the pass staging (%lld bytes)",
                    name_.c_str(), r, (long long)bytes_);
            return RAD_E_NOMEM;
        }
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0].device));
    out_.assign(ranks_.size(), RadBatch{});
    aout_.assign(ranks_.size(), RadDuplexBatch{});
    kout_.assign(ranks_.size() * (1 + state_groups_.size()), RadKVGroupBatch{});
    free_lanes_.clear();
    for (int64_t i = max_lanes_ - 1; i >= 0; --i) free_lanes_.push_back((int32_t)i);
    return RAD_OK;
}

void DuplexDriver::fini() {
    for (size_t r = 0; r < dev_.size(); ++r)
        if (dev_[r]) {
            if (r < ranks_.size()) rad_dev_set(ranks_[r].device);
            rad_dev_free(dev_[r], RAD_MEM_DEVICE);
        }
    dev_.clear();
    if (!ranks_.empty()) rad_dev_set(ranks_[0].device);
    if (host_) rad_dev_free(host_, RAD_MEM_HOST_PINNED);
    host_ = nullptr;
    audio_in_ptr_.clear();
    rb_.clear();
}

/* ================================================================== requests */
DuplexDriver::Job* DuplexDriver::job_by_lane_locked(int32_t lane) {
    for (auto& j : jobs_) if (j->lane == lane) return j.get();
    return nullptr;
}

DuplexDriver::Job* DuplexDriver::job_locked(uint64_t id) {
    for (auto& j : jobs_) if (j->id == id) return j.get();
    return nullptr;
}

void DuplexDriver::release_locked(Job& j) {
    if (j.released) return;
    if (j.have_kv && j.seq) {
        kv_->free_sequence(j.seq);
        j.seq = 0;
        j.have_kv = false;
    }
    if (j.lane >= 0) {
        free_lanes_.push_back(j.lane);
        j.lane = -1;
    }
    j.released = true;
}

void DuplexDriver::fail_locked(Job& j, int status, const std::string& why,
                               std::vector<Finish>* out) {
    j.failed = true;
    if (j.open_ticket && !j.ready) {
        out->push_back({ j.open_ticket, status, why });
        j.open_ticket.reset();
    }
    if (j.step_ticket) {
        out->push_back({ j.step_ticket, status, why });
        j.step_ticket.reset();
    }
    j.has_pending = false;
    j.pending_samples.clear();
}

void DuplexDriver::fail_pass_locked(int status, const std::string& why,
                                    std::vector<Finish>* out) {
    for (uint64_t id : pass_.jobs) {
        Job* j = job_locked(id);
        if (!j) continue;
        j->in_pass = false;
        fail_locked(*j, status, why, out);
    }
    pass_ = Pass{};
    pass_active_ = false;
    rb_.clear();
}

/* Drop closed and failed lanes. Called with no pass in flight. */
void DuplexDriver::retire_locked() {
    for (auto it = jobs_.begin(); it != jobs_.end();) {
        Job& j = **it;
        if (!(j.closed || j.failed)) { ++it; continue; }
        release_locked(j);
        it = jobs_.erase(it);
    }
}

void DuplexDriver::wake(std::vector<Finish>* fin, std::vector<Send>* send) {
    if (fin)
        for (Finish& f : *fin)
            if (f.t) f.t->finish(f.status, f.why);
    if (send)
        for (Send& s : *send)
            if (s.t) s.t->reply(std::move(s.r));
}

int DuplexDriver::open(std::unique_ptr<DuplexRequest> r, std::shared_ptr<DuplexTicket>* out,
                       std::string* why) {
    auto t = std::make_shared<DuplexTicket>();
    if (out) *out = t;
    auto no = [&](int st, const std::string& w) {
        if (why) *why = w;
        t->finish(st, w);
        return st;
    };
    if (!r) return no(RAD_E_INVAL, "no request");
    const int64_t T = (int64_t)r->tokens.size();
    if (T < 1 || T > decl_.max_prompt || T > max_tok_ || T > max_ctx_)
        return no(RAD_E_INVAL, "the prompt is " + std::to_string(T) + " rows; this tenant takes "
                               "1.." + std::to_string(std::min({ decl_.max_prompt, max_tok_,
                                                                 max_ctx_ })) +
                               " rows");
    const bool has_refs = !r->ref_audio.empty() || !r->ref_cu.empty();
    const int64_t n_ref = r->ref_cu.empty() ? 0 : (int64_t)r->ref_cu.size() - 1;
    if (has_refs) {
        const int64_t hop = decl_.ref_hop;
        bool ok = decl_.max_ref_frames > 0 && n_ref >= 1 && n_ref <= decl_.max_ref_frames &&
                  r->ref_cu[0] == 0 && r->ref_cu.back() == (int64_t)r->ref_audio.size() &&
                  (int64_t)r->ref_audio.size() <= decl_.max_ref_frames * hop &&
                  (int64_t)r->ref_frame.size() == T;
        for (int64_t i = 0; ok && i < n_ref; ++i)
            ok = r->ref_cu[(size_t)i + 1] > r->ref_cu[(size_t)i] &&
                 r->ref_cu[(size_t)i + 1] % hop == 0;
        const int64_t frames = ok ? r->ref_cu.back() / hop : 0;
        for (int64_t i = 0; ok && i < T; ++i)
            ok = r->ref_frame[(size_t)i] >= -1 && r->ref_frame[(size_t)i] < frames;
        if (!ok)
            return no(RAD_E_INVAL, decl_.max_ref_frames > 0
                       ? "the references do not fit this tenant: at most " +
                             std::to_string(decl_.max_ref_frames) + " clips of whole frames"
                       : std::string("this tenant takes no reference audio"));
    }
    if (!kv_->can_ever_fit(max_ctx_))
        return no(RAD_E_FULL, "a lane of up to " + std::to_string(max_ctx_) + " rows does not fit "
                              "the KV cache even when it is empty");

    int st = RAD_OK;
    std::string w;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) {
            st = closed_status_;
            w = closed_why_;
        } else if (free_lanes_.empty()) {
            st = RAD_E_FULL;
            w = "the tenant is serving its " + std::to_string(max_lanes_) +
                " lanes; close a conversation before opening another";
        } else {
            auto j = std::make_unique<Job>();
            j->id = next_id_++;
            j->seq = ((uint64_t)1 << 61) | next_seq_++;
            if (kv_->add_sequence(j->seq) < 0 || kv_->ensure(j->seq, max_ctx_) < 0) {
                if (kv_->has_sequence(j->seq)) kv_->free_sequence(j->seq);
                st = RAD_E_FULL;
                w = "the KV cache has no room for a lane of " + std::to_string(max_ctx_) +
                    " rows";
            } else {
                j->have_kv = true;
                j->lane = free_lanes_.back();
                free_lanes_.pop_back();
                j->req = std::move(r);
                j->open_ticket = t;
                j->T = T;
                j->seed = j->req->seed;
                jobs_.push_back(std::move(j));
            }
        }
    }
    if (st < 0) return no(st, w);
    return RAD_OK;
}

int DuplexDriver::step(int32_t lane, const float* samples, int64_t n_samples, uint64_t seq,
                       std::shared_ptr<DuplexTicket>* out, std::string* why) {
    auto t = std::make_shared<DuplexTicket>();
    if (out) *out = t;
    auto no = [&](int st, const std::string& w) {
        if (why) *why = w;
        t->finish(st, w);
        return st;
    };
    if (!samples) return no(RAD_E_INVAL, "no frame samples");
    if (n_samples != decl_.samples_per_frame)
        return no(RAD_E_INVAL, "the frame is " + std::to_string(n_samples) + " samples; this "
                               "tenant takes " + std::to_string(decl_.samples_per_frame));
    int st = RAD_OK;
    std::string w;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) {
            st = closed_status_;
            w = closed_why_;
        } else if (Job* j = job_by_lane_locked(lane); !j) {
            st = RAD_E_NOTFOUND;
            w = "lane " + std::to_string(lane) + " is not open";
        } else if (!j->ready) {
            st = RAD_E_STATE;
            w = "lane " + std::to_string(lane) + "'s prompt is not ready";
        } else if (j->has_pending) {
            st = RAD_E_STATE;
            w = "lane " + std::to_string(lane) + " already has a step in flight";
        } else if (j->pos >= max_ctx_) {
            st = RAD_E_FULL;
            w = "lane " + std::to_string(lane) + " reached its context bound (" +
                std::to_string(max_ctx_) + " rows)";
        } else {
            j->pending_samples.assign(samples, samples + n_samples);
            j->pending_seq = seq;
            j->step_ticket = t;
            j->has_pending = true;
        }
    }
    if (st < 0) return no(st, w);
    return RAD_OK;
}

int DuplexDriver::flush(int32_t lane, uint64_t seq, std::shared_ptr<DuplexTicket>* out,
                        std::string* why) {
    auto t = std::make_shared<DuplexTicket>();
    if (out) *out = t;
    int st = RAD_OK;
    std::string w;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) {
            st = closed_status_;
            w = closed_why_;
        } else if (Job* j = job_by_lane_locked(lane); !j) {
            st = RAD_E_NOTFOUND;
            w = "lane " + std::to_string(lane) + " is not open";
        } else if (!j->ready) {
            st = RAD_E_STATE;
            w = "lane " + std::to_string(lane) + "'s prompt is not ready";
        } else if (j->has_pending) {
            st = RAD_E_STATE;
            w = "lane " + std::to_string(lane) + " already has a step in flight";
        }
    }
    if (st < 0) {
        if (why) *why = w;
        t->finish(st, w);
        return st;
    }
    DuplexReply r;
    r.seq = seq;
    r.status = 0;
    r.has_audio = false;
    r.token = -1;
    t->reply(std::move(r));
    return RAD_OK;
}

int DuplexDriver::close(int32_t lane, std::string* why) {
    std::vector<Finish> fin;
    bool found = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = jobs_.begin(); it != jobs_.end(); ++it) {
            Job& j = **it;
            if (j.lane != lane) continue;
            found = true;
            if (j.closed) break;
            j.closed = true;
            const std::string w = "lane " + std::to_string(lane) + " was closed";
            if (j.open_ticket && !j.ready) {
                fin.push_back({ j.open_ticket, RAD_E_CANCELLED, w });
                j.open_ticket.reset();
            }
            if (j.step_ticket) {
                fin.push_back({ j.step_ticket, RAD_E_CANCELLED, w });
                j.step_ticket.reset();
            }
            j.has_pending = false;
            j.pending_samples.clear();
            if (!j.in_pass) {
                release_locked(j);
                jobs_.erase(it);
            }
            break;
        }
    }
    wake(&fin, nullptr);
    if (!found && why) *why = "lane " + std::to_string(lane) + " is not open";
    return RAD_OK;
}

void DuplexDriver::shutdown(const char* why, int status) {
    std::vector<Finish> fin;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) {
            /* Still fail anything that arrived before the first shutdown. */
        }
        closed_ = true;
        closed_why_ = why ? why : "the engine stopped";
        closed_status_ = status;
        for (auto& jp : jobs_) {
            Job& j = *jp;
            j.closed = true;
            if (j.open_ticket && !j.ready) {
                fin.push_back({ j.open_ticket, status, closed_why_ });
                j.open_ticket.reset();
            }
            if (j.step_ticket) {
                fin.push_back({ j.step_ticket, status, closed_why_ });
                j.step_ticket.reset();
            }
            j.has_pending = false;
            j.pending_samples.clear();
            release_locked(j);
        }
        jobs_.clear();
        pass_ = Pass{};
        pass_active_ = false;
        rb_.clear();
    }
    wake(&fin, nullptr);
}

int64_t DuplexDriver::queued() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_)
        if (!j->closed && !j->failed && (!j->ready || j->has_pending)) ++n;
    return n;
}

int64_t DuplexDriver::running() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_)
        if (!j->closed && !j->failed && j->ready) ++n;
    return n;
}

/* ================================================================== staging */
int DuplexDriver::upload() {
    for (size_t r = 0; r < ranks_.size(); ++r) {
        for (int g = 0; g < R_N; ++g) {
            if (g == R_AIN || g == R_AOUT || g == R_TOUT || g == R_STAT) continue;
            if (reg_[g].used <= 0) continue;
            RAD_TRY(rad_memcpy_async((char*)dev_[r] + reg_[g].off, host_ + reg_[g].off,
                                     reg_[g].used, ranks_[r].stream));
        }
        if (reg_[R_AIN].used > 0)
            RAD_TRY(rad_memcpy_async(audio_in_ptr_[r], host_ + reg_[R_AIN].off,
                                     reg_[R_AIN].used, ranks_[r].stream));
    }
    return RAD_OK;
}

void DuplexDriver::base_batches(int64_t n_tok, int64_t n_seq, int32_t max_q, int32_t max_ctx,
                                int step, bool kv, int64_t pitch) {
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadBatch& B = out_[r];
        RadDuplexBatch& D = aout_[r];
        RadKVGroupBatch& K = kout_[r * (1 + state_groups_.size())];
        const int ri = (int)r;
        B = RadBatch{};
        B.step = step;
        B.phase = RAD_PHASE_PREFILL;
        B.n_tok = n_tok;
        B.n_seq = n_seq;
        B.max_q_len = max_q;
        B.max_ctx_len = max_ctx;
        B.max_q_len_prefill = max_q;
        B.token_ids = dptr<int32_t>(ri, R_TOK);
        B.positions = dptr<int32_t>(ri, R_POS);
        B.cu_seqlens = dptr<int32_t>(ri, R_CU);
        if (kv) {
            K = RadKVGroupBatch{};
            K.group = (rad_kvgroup)kv_group_;
            K.slot_mapping = dptr<int32_t>(ri, R_SLOT);
            K.block_table = dptr<int32_t>(ri, R_BT);
            K.block_table_pitch = pitch;
            K.seqused = dptr<int32_t>(ri, R_USED);
            K.block_size = block_size_;
            K.max_blocks = max_blocks_;
            for (size_t gi = 0; gi < state_groups_.size(); ++gi) {
                RadKVGroupBatch& Ks = (&K)[1 + gi];
                Ks = RadKVGroupBatch{};
                Ks.group = (rad_kvgroup)state_groups_[gi].g;
                Ks.block_table = dptr<int32_t>(ri, R_SBT) + gi * max_seqs_;
                Ks.block_table_pitch = 1;
                Ks.seqused = dptr<int32_t>(ri, R_USED);
                Ks.state_index = dptr<int32_t>(ri, R_SIDX) + gi * max_seqs_ * kMaxStateWidth;
                Ks.state_index_pitch = state_groups_[gi].width;
                Ks.max_blocks = 1;
            }
            B.n_kv_groups = (int)(1 + state_groups_.size());
            B.kv = &K;
            B.ctx_lens = dptr<int32_t>(ri, R_CTX);
            B.num_accepted = dptr<int32_t>(ri, R_ACC);
        }
        D = RadDuplexBatch{};
        D.lane = dptr<int32_t>(ri, R_LANE);
        D.frame = dptr<int32_t>(ri, R_FRAME);
        D.seed = dptr<uint32_t>(ri, R_SEED);
        D.flags = dptr<int32_t>(ri, R_FLAGS);
        B.dup_b = &D;
    }
}

void DuplexDriver::stage_state(const std::vector<uint64_t>& seqs,
                               const std::vector<int32_t>& ctx) {
    const int64_t n = (int64_t)seqs.size();
    int32_t* cl = hptr<int32_t>(R_CTX);
    int32_t* acc = hptr<int32_t>(R_ACC);
    for (int64_t e = 0; e < n; ++e) {
        cl[e] = ctx[(size_t)e];
        acc[e] = 1;
    }
    reg_[R_CTX].used = reg_[R_ACC].used = n * 4;
    for (size_t gi = 0; gi < state_groups_.size(); ++gi) {
        const StateGroup& sg = state_groups_[gi];
        int32_t* bt = hptr<int32_t>(R_SBT) + gi * max_seqs_;
        int32_t* si = hptr<int32_t>(R_SIDX) + gi * max_seqs_ * kMaxStateWidth;
        for (int64_t e = 0; e < n; ++e) {
            bt[e] = kv_->state_slot(seqs[(size_t)e], sg.g);
            kv_->state_index_row(seqs[(size_t)e], sg.g, si + e * sg.width, (int)sg.width);
        }
    }
    if (!state_groups_.empty()) {
        reg_[R_SBT].used = (int64_t)state_groups_.size() * max_seqs_ * 4;
        reg_[R_SIDX].used = (int64_t)state_groups_.size() * max_seqs_ * kMaxStateWidth * 4;
    }
}

int DuplexDriver::build_prompt(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    pass_ = Pass{};
    pass_.kind = RAD_GEN_PASS_DUPLEX_PROMPT;
    pass_.jobs.assign(1, j.id);
    pass_.rows = j.T;
    const DuplexRequest& q = *j.req;
    const int64_t T = j.T;
    int32_t* tok = hptr<int32_t>(R_TOK);
    int32_t* pos = hptr<int32_t>(R_POS);
    int32_t* slot = hptr<int32_t>(R_SLOT);
    int32_t* cu = hptr<int32_t>(R_CU);
    int32_t* btab = hptr<int32_t>(R_BT);
    int32_t* used = hptr<int32_t>(R_USED);
    const std::vector<int32_t>& bt = kv_->block_table(j.seq, kv_group_);
    const int64_t pitch = std::max<int64_t>(1, (int64_t)bt.size());
    cu[0] = 0;
    for (int64_t i = 0; i < T; ++i) {
        tok[i] = q.tokens[(size_t)i];
        pos[i] = (int32_t)i;
        slot[i] = slot_of(bt, i, block_size_);
    }
    cu[1] = (int32_t)T;
    for (int64_t k = 0; k < pitch; ++k) btab[k] = k < (int64_t)bt.size() ? bt[(size_t)k] : -1;
    used[0] = (int32_t)T;
    h_lane_.assign(1, j.lane);
    h_lrow_.assign(1, T - 1);
    hptr<int32_t>(R_LANE)[0] = j.lane;
    hptr<int32_t>(R_LROW)[0] = (int32_t)(T - 1);
    reg_[R_TOK].used = reg_[R_POS].used = reg_[R_SLOT].used = T * 4;
    reg_[R_CU].used = 2 * 4;
    reg_[R_BT].used = pitch * 4;
    reg_[R_USED].used = 4;
    reg_[R_LANE].used = reg_[R_LROW].used = 4;

    const int64_t n_ref = q.ref_cu.empty() ? 0 : (int64_t)q.ref_cu.size() - 1;
    h_ref_cu_.clear();
    h_ref_frame_.clear();
    if (n_ref > 0) {
        std::memcpy(hptr<float>(R_RAUD), q.ref_audio.data(), q.ref_audio.size() * 4);
        std::memcpy(hptr<int32_t>(R_RCU), q.ref_cu.data(), q.ref_cu.size() * 4);
        std::memcpy(hptr<int32_t>(R_RFRM), q.ref_frame.data(), (size_t)T * 4);
        reg_[R_RAUD].used = (int64_t)q.ref_audio.size() * 4;
        reg_[R_RCU].used = (int64_t)q.ref_cu.size() * 4;
        reg_[R_RFRM].used = T * 4;
        h_ref_cu_ = q.ref_cu;
        h_ref_frame_ = q.ref_frame;
    }

    std::vector<uint64_t> seqs{ j.seq };
    std::vector<int32_t> ctx{ 0 };
    stage_state(seqs, ctx);
    base_batches(T, 1, (int32_t)T, (int32_t)T, step, true, pitch);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadBatch& B = out_[r];
        B.gen = RAD_GEN_PASS_DUPLEX_PROMPT;
        RadDuplexBatch& D = aout_[r];
        D.n_lane = 1;
        D.n_tok = T;
        D.lane_h = h_lane_.data();
        D.last_row_h = h_lrow_.data();
        if (n_ref > 0) {
            const int ri = (int)r;
            D.n_ref = n_ref;
            D.ref_samples = (int64_t)q.ref_audio.size();
            D.ref_audio = dptr<float>(ri, R_RAUD);
            D.ref_cu = dptr<int32_t>(ri, R_RCU);
            D.ref_cu_h = h_ref_cu_.data();
            D.ref_frame = dptr<int32_t>(ri, R_RFRM);
            D.ref_frame_h = h_ref_frame_.data();
        }
    }
    pass_.pitch = pitch;
    return RAD_OK;
}

int DuplexDriver::build_step(const std::vector<Job*>& js, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    pass_ = Pass{};
    pass_.kind = RAD_GEN_PASS_DUPLEX_STEP;
    pass_.jobs.clear();
    const int64_t L = (int64_t)js.size();
    int32_t* tok = hptr<int32_t>(R_TOK);
    int32_t* pos = hptr<int32_t>(R_POS);
    int32_t* slot = hptr<int32_t>(R_SLOT);
    int32_t* cu = hptr<int32_t>(R_CU);
    int32_t* btab = hptr<int32_t>(R_BT);
    int32_t* used = hptr<int32_t>(R_USED);
    int32_t* lane = hptr<int32_t>(R_LANE);
    int32_t* frame = hptr<int32_t>(R_FRAME);
    int32_t* flags = hptr<int32_t>(R_FLAGS);
    uint32_t* seed = hptr<uint32_t>(R_SEED);
    int64_t pitch = 1;
    for (const Job* j : js)
        pitch = std::max<int64_t>(pitch, (int64_t)kv_->block_table(j->seq, kv_group_).size());
    std::memset(hptr<float>(R_AIN), 0, (size_t)(max_lanes_ * decl_.samples_per_frame * 4));
    std::memset(hptr<float>(R_AOUT), 0, (size_t)(max_lanes_ * decl_.samples_per_frame * 4));
    std::memset(hptr<int32_t>(R_TOUT), 0, (size_t)(max_lanes_ * 4));
    std::memset(hptr<int32_t>(R_STAT), 0, (size_t)(max_lanes_ * 4));

    h_lane_.assign((size_t)L, 0);
    h_frame_.assign((size_t)L, 0);
    h_flags_.assign((size_t)L, 0);
    std::vector<uint64_t> seqs;
    std::vector<int32_t> ctx;
    int32_t max_ctx = 0;
    cu[0] = 0;
    for (int64_t i = 0; i < L; ++i) {
        Job& j = *js[(size_t)i];
        const int64_t p = j.pos;
        const std::vector<int32_t>& bt = kv_->block_table(j.seq, kv_group_);
        tok[i] = 0;
        pos[i] = (int32_t)p;
        slot[i] = slot_of(bt, p, block_size_);
        cu[i + 1] = (int32_t)(i + 1);
        used[i] = (int32_t)(p + 1);
        for (int64_t k = 0; k < pitch; ++k)
            btab[i * pitch + k] = k < (int64_t)bt.size() ? bt[(size_t)k] : -1;
        lane[i] = h_lane_[(size_t)i] = j.lane;
        frame[i] = h_frame_[(size_t)i] = j.frame;
        const int32_t f = j.frame == 0 ? RAD_DUPLEX_F_FIRST : 0;
        flags[i] = h_flags_[(size_t)i] = f;
        seed[2 * i] = (uint32_t)(j.seed & 0xffffffffu);
        seed[2 * i + 1] = (uint32_t)(j.seed >> 32);
        max_ctx = std::max<int32_t>(max_ctx, (int32_t)(p + 1));
        seqs.push_back(j.seq);
        ctx.push_back((int32_t)p);
        std::memcpy(hptr<float>(R_AIN) + j.lane * decl_.samples_per_frame,
                    j.pending_samples.data(), (size_t)decl_.samples_per_frame * 4);
        pass_.jobs.push_back(j.id);
    }
    reg_[R_TOK].used = reg_[R_POS].used = reg_[R_SLOT].used = L * 4;
    reg_[R_CU].used = (L + 1) * 4;
    reg_[R_BT].used = L * pitch * 4;
    reg_[R_USED].used = L * 4;
    reg_[R_LANE].used = reg_[R_FRAME].used = reg_[R_FLAGS].used = L * 4;
    reg_[R_SEED].used = 2 * L * 4;
    reg_[R_AIN].used = max_lanes_ * decl_.samples_per_frame * 4;
    stage_state(seqs, ctx);
    base_batches(L, L, 1, max_ctx, step, true, pitch);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadBatch& B = out_[r];
        B.gen = RAD_GEN_PASS_DUPLEX_STEP;
        B.phase = RAD_PHASE_DECODE;
        B.n_seq_decode = L;
        B.n_tok_decode = L;
        B.max_q_len_decode = 1;
        B.max_q_len_prefill = 0;
        RadDuplexBatch& D = aout_[r];
        D.n_lane = L;
        D.lane_h = h_lane_.data();
        D.frame_h = h_frame_.data();
        D.flags_h = h_flags_.data();
    }
    rb_.clear();
    rb_.push_back(Readback{ decl_.audio_out, max_lanes_ * decl_.samples_per_frame * 4,
                            hptr<void>(R_AOUT) });
    if (decl_.text)
        rb_.push_back(Readback{ decl_.text_out, max_lanes_ * 4, hptr<void>(R_TOUT) });
    rb_.push_back(Readback{ decl_.status, max_lanes_ * 4, hptr<void>(R_STAT) });
    pass_.pitch = pitch;
    return RAD_OK;
}

/* ================================================================== the passes */
const RadBatch* DuplexDriver::next(int step) {
    std::vector<Finish> fin;
    std::unique_lock<std::mutex> lk(mu_);
    pass_ = Pass{};
    rb_.clear();
    if (closed_ || pass_active_) return nullptr;
    retire_locked();

    Job* pj = nullptr;
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.closed || j.failed || j.ready) continue;
        pj = &j;
        break;
    }
    if (pj) {
        if (build_prompt(*pj, step) < 0) {
            fail_pass_locked(RAD_E_STATE, "the duplex prompt could not be staged", &fin);
        } else {
            pj->in_pass = true;
            pass_active_ = true;
        }
    } else {
        std::vector<Job*> js;
        for (auto& jp : jobs_) {
            Job& j = *jp;
            if (j.closed || j.failed || !j.ready || !j.has_pending) continue;
            js.push_back(&j);
        }
        if (!js.empty()) {
            if (build_step(js, step) < 0) {
                fail_pass_locked(RAD_E_STATE, "the duplex step could not be staged", &fin);
            } else {
                for (Job* j : js) j->in_pass = true;
                pass_active_ = true;
            }
        }
    }
    if (!pass_active_) {
        lk.unlock();
        wake(&fin, nullptr);
        return nullptr;
    }
    lk.unlock();
    wake(&fin, nullptr);
    const int rc = upload();
    if (rc < 0) {
        std::lock_guard<std::mutex> lk2(mu_);
        fail_pass_locked(rc, "the duplex pass's staging copy failed", &fin);
    }
    if (rc < 0) {
        wake(&fin, nullptr);
        return nullptr;
    }
    return &out_[0];
}

const RadBatch* DuplexDriver::batch_for(int rank) const {
    if (rank < 0 || (size_t)rank >= out_.size() || pass_.kind == 0) return nullptr;
    return &out_[(size_t)rank];
}

int DuplexDriver::done() {
    std::vector<Finish> fin;
    std::vector<Send> send;
    std::vector<std::shared_ptr<DuplexTicket>> ready;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (pass_.kind == 0) return RAD_OK;
        if (pass_.kind == RAD_GEN_PASS_DUPLEX_PROMPT) {
            for (uint64_t id : pass_.jobs) {
                Job* j = job_locked(id);
                if (!j) continue;
                if (j->failed || j->closed) continue;
                j->pos = j->T;
                j->ready = true;
                if (j->open_ticket) {
                    ready.push_back(j->open_ticket);
                    j->open_ticket.reset();
                }
            }
        } else if (pass_.kind == RAD_GEN_PASS_DUPLEX_STEP) {
            const int32_t* stat = hptr<int32_t>(R_STAT);
            const int32_t* text = hptr<int32_t>(R_TOUT);
            const float* audio = hptr<float>(R_AOUT);
            for (uint64_t id : pass_.jobs) {
                Job* j = job_locked(id);
                if (!j || j->failed || j->closed) continue;
                const int32_t st = stat[j->lane];
                if (st < 0) {
                    fail_locked(*j, st, "the duplex lane failed", &fin);
                    continue;
                }
                if (j->step_ticket) {
                    Send s;
                    s.t = j->step_ticket;
                    s.r.seq = j->pending_seq;
                    s.r.status = st;
                    s.r.has_audio = true;
                    s.r.token = decl_.text ? text[j->lane] : -1;
                    const size_t off = (size_t)j->lane * (size_t)decl_.samples_per_frame;
                    s.r.audio.assign(audio + off, audio + off + (size_t)decl_.samples_per_frame);
                    send.push_back(std::move(s));
                }
                j->pos += 1;
                j->frame += 1;
                j->has_pending = false;
                j->pending_samples.clear();
                j->step_ticket.reset();
            }
        }
        for (uint64_t id : pass_.jobs) {
            Job* j = job_locked(id);
            if (j) j->in_pass = false;
        }
        pass_ = Pass{};
        pass_active_ = false;
        retire_locked();
    }
    for (auto& t : ready) if (t) t->ready();
    wake(&fin, &send);
    rb_.clear();
    return RAD_OK;
}

}  /* namespace gen */
}  /* namespace rad */
