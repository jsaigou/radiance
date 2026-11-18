/* duplex_driver.h -- the duplex tenant's pass driver (abi/rad_builder.h's RadDuplexDecl describes
 * the passes; docs/DUPLEX-PLUGIN-ABI.md sections 2.3, 2.5-2.7 and 4).
 *
 * Driven by the engine thread like the audio driver (audio_driver.h): next() picks and stages one
 * pass, done() folds its result back. The server's threads call open(), step(), flush() and
 * close(); each operation returns a ticket they wait on.
 *
 * A lane is one conversation. Its one KV sequence is reserved at open for the whole context the
 * driver was given (the prompt and every live frame the lane may add), so an admitted lane never
 * waits for KV. The paged full-attention group is the language model's; a hybrid model's linear
 * and convolution groups are staged beside it exactly as in the audio driver.
 *
 * Pass order, one pass at a time:
 *
 *   1. PROMPT a lane's card tokens and reference-frame rows, one lane a pass. The pass that ends
 *      the prompt makes the lane ready and completes its open ticket.
 *   2. STEP every ready lane with an input frame waiting, one row a lane. The driver uploads each
 *      lane's frame into the declaration's audio_in buffer, then reads audio_out, text_out and
 *      status back per the declaration.
 *
 * Guarantees: every accepted step gets exactly one reply, in the lane's input order; a lane's KV,
 * state and buffers are never shared with another lane; close (or shutdown) releases the lane and
 * fails anything outstanding; a lane that is closed, failed or shut down never leaves a ticket
 * waiting. What it deliberately does not do yet: hold a frame for flush (flush returns an empty
 * reply), implement the codec/MeanFlow work inside STEP (the plugin does that), or bind itself to
 * the engine and server. Engine::serve still refuses a duplex program before constructing a
 * driver.
 *
 * Locking order: there is one mutex (`mu_`). open/step/flush/close take it for lane and KV
 * bookkeeping; next() also prepares the host staging under it (KV reads included), then releases
 * it before any device copy. done() takes it only for bookkeeping. No ticket is finished and no
 * device call is made while it is held. close() during a pass marks the lane and fails its
 * tickets but defers releasing its KV and lane until done(), so a pass never reads freed state.
 */
#pragma once
#include "pass.h"
#include "device/device.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
struct IKVManager;
namespace gen {

/* One step's reply: the echoed wire sequence, the lane's model frame (when it made one), an
 * optional text token (-1 for none) and the plugin's status (0 replied, < 0 fatal). */
struct DuplexReply {
    uint64_t           seq = 0;
    bool               has_audio = false;
    std::vector<float> audio;
    int32_t            token = -1;
    int32_t            status = 0;
};

/* One lane operation's completion. The driver writes, the caller reads. */
class DuplexTicket {
public:
    /* driver side */
    void ready();                              /* open: the lane's prompt is in the KV */
    void reply(DuplexReply&& r);               /* step/flush: the one reply */
    void finish(int status, const std::string& why = std::string());   /* refused/closed */

    /* caller side */
    /* For an open: waits until the driver completed it. True when done (check status()). */
    bool wait(int ms, bool* timed_out);
    /* For a step or flush: the next (one) reply. False once finished without one. */
    bool next(DuplexReply* out, int ms, bool* timed_out);
    bool finished() const;
    int  status() const;
    std::string why() const;
    /* For an open: the lane the driver admitted this conversation to, valid as soon as open()
     * returned RAD_OK (the lane is assigned at admission, not when the prompt runs). It is the
     * handle step(), flush() and close() take. -1 on a ticket that was refused. */
    int32_t lane() const { return lane_.load(); }
    void    set_lane(int32_t l) { lane_.store(l); }

    std::atomic<bool> cancel{false};

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<DuplexReply> q_;
    bool                    done_ = false;
    int                     status_ = RAD_OK;
    std::string             why_;
    std::atomic<int32_t>    lane_{-1};
};

/* What a server hands open(): the prompt rows (card tokens and one placeholder per reference
 * frame), the reference clips end to end, their bounds and each row's frame index. */
struct DuplexRequest {
    std::vector<int32_t> tokens;
    std::vector<float>   ref_audio;
    std::vector<int32_t> ref_cu;
    std::vector<int32_t> ref_frame;   /* one a row: -1 card/text, >= 0 a reference frame */
    uint64_t             seed = 0;
};

class DuplexDriver final : public IPassDriver {
public:
    struct RankIO {
        int              device = 0;
        RadStream        stream = nullptr;
        /* This rank's Program: the driver writes the declared audio_in buffer through
         * BufferInfo::ptr. Null means the Program init() was handed. */
        const Program*   program = nullptr;
    };

    /* `kv` is rank 0's KV manager: one block table serves every rank. `max_ctx` bounds a lane's
     * prompt plus live frames; it is the context reserved one sequence a lane at admission. */
    int  init(const Program& prog, IKVManager* kv, const std::vector<RankIO>& ranks,
              int64_t max_ctx);
    void fini();
    ~DuplexDriver() override {
        /* A destroyed driver must not leave a ticket waiting: fail everything first. */
        shutdown("the duplex driver was destroyed", RAD_E_CANCELLED);
        fini();
    }

    /* The bytes init() allocates on each rank's card, for the VRAM budget. */
    static int64_t device_bytes(const Program& prog, int64_t max_ctx);
    /* State slots a state_index row holds at most, as the audio driver: a lane never speculates. */
    static constexpr int64_t kMaxStateWidth = 8;

    /* ---------------------------------------------------------- server threads */
    /* A lane's prompt, refused synchronously with `why` when it can never run: no lane free, no
     * KV for the context bound, a prompt longer than the declaration, or bad references. */
    int  open(std::unique_ptr<DuplexRequest> r, std::shared_ptr<DuplexTicket>* out,
              std::string* why);
    /* One live frame for a lane. Refused when the lane is unknown or not ready, a step is already
     * pending or in flight, the frame length is not samples_per_frame, or the lane's context is
     * full. */
    int  step(int32_t lane, const float* samples, int64_t n_samples, uint64_t seq,
              std::shared_ptr<DuplexTicket>* out, std::string* why);
    /* Ask for whatever the lane buffered. This stage keeps no buffered frame: the reply is empty
     * with the echoed `seq`. */
    int  flush(int32_t lane, uint64_t seq, std::shared_ptr<DuplexTicket>* out, std::string* why);
    /* Release the lane, its sequence and its state, and fail its outstanding tickets. Safe at any
     * time and idempotent. */
    int  close(int32_t lane, std::string* why = nullptr);

    void shutdown(const char* why, int status) override;

    /* ---------------------------------------------------------- the engine thread */
    const RadBatch* next(int step) override;
    const RadBatch* batch_for(int rank) const override;
    const std::vector<Readback>& readbacks() const override { return rb_; }
    int  done() override;

    int64_t queued() const override;
    int64_t running() const override;

private:
    struct Job {
        uint64_t id = 0;
        std::unique_ptr<DuplexRequest> req;
        std::shared_ptr<DuplexTicket>  open_ticket;
        int32_t  lane = -1;
        uint64_t seq = 0;
        bool     have_kv = false;
        bool     ready = false;
        bool     failed = false;
        bool     closed = false;
        bool     released = false;
        bool     in_pass = false;
        bool     has_pending = false;
        uint64_t pending_seq = 0;
        std::vector<float> pending_samples;
        std::shared_ptr<DuplexTicket> step_ticket;
        int64_t  T = 0;           /* prompt rows */
        int64_t  pos = 0;         /* next live row */
        int32_t  frame = 0;       /* next live frame number */
        uint64_t seed = 0;
    };
    struct Pass {
        int      kind = 0;
        std::vector<uint64_t> jobs;
        int64_t  rows = 0;        /* PROMPT: the lane's prompt rows */
        int64_t  pitch = 0;       /* block-table pitch */
    };
    struct StateGroup { int32_t g = 0; int64_t width = 1; };
    struct Region { int64_t off = 0, bytes = 0, used = 0; };
    struct Finish {
        std::shared_ptr<DuplexTicket> t;
        int status = RAD_E_STATE;
        std::string why;
    };
    struct Send {
        std::shared_ptr<DuplexTicket> t;
        DuplexReply r;
    };
    enum { R_TOK, R_POS, R_CU, R_SLOT, R_BT, R_USED, R_LANE, R_LROW, R_FRAME, R_SEED,
           R_FLAGS, R_CTX, R_ACC, R_SBT, R_SIDX, R_RAUD, R_RCU, R_RFRM,
           R_AIN, R_AOUT, R_TOUT, R_STAT, R_N };

    template <class T> T* hptr(int r) const { return reinterpret_cast<T*>(host_ + reg_[r].off); }
    template <class T> T* dptr(int rank, int r) const {
        return reinterpret_cast<T*>((char*)dev_[(size_t)rank] + reg_[r].off);
    }

    int  upload();
    void base_batches(int64_t n_tok, int64_t n_seq, int32_t max_q, int32_t max_ctx, int step,
                      bool kv, int64_t pitch);
    void stage_state(const std::vector<uint64_t>& seqs, const std::vector<int32_t>& ctx);
    int  build_prompt(Job& j, int step);
    int  build_step(const std::vector<Job*>& js, int step);

    Job* job_by_lane_locked(int32_t lane);
    Job* job_locked(uint64_t id);
    void release_locked(Job& j);
    void fail_locked(Job& j, int status, const std::string& why, std::vector<Finish>* out);
    void fail_pass_locked(int status, const std::string& why, std::vector<Finish>* out);
    void retire_locked();
    void add_unique_seq_locked(Job& j);

    void wake(std::vector<Finish>* fin, std::vector<Send>* send);

    RadDuplexDecl     decl_{};
    std::string       name_;
    int64_t           max_tok_ = 0, max_seqs_ = 0, max_lanes_ = 0, max_ctx_ = 0;
    int64_t           block_size_ = 0, max_blocks_ = 0;
    int32_t           kv_group_ = 0;
    std::vector<StateGroup> state_groups_;
    std::vector<RankIO>     ranks_;
    std::vector<void*>      audio_in_ptr_;
    IKVManager*             kv_ = nullptr;

    char*               host_ = nullptr;
    int64_t             bytes_ = 0;
    std::vector<void*>  dev_;
    Region              reg_[R_N];
    std::vector<Readback> rb_;

    std::vector<RadBatch>         out_;
    std::vector<RadDuplexBatch>   aout_;
    std::vector<RadKVGroupBatch>  kout_;
    Pass                          pass_;
    bool                          pass_active_ = false;
    std::vector<int32_t>          h_lane_, h_lrow_, h_frame_, h_flags_, h_ref_cu_, h_ref_frame_;

    mutable std::mutex              mu_;
    std::deque<std::unique_ptr<Job>> jobs_;
    std::vector<int32_t>            free_lanes_;
    uint64_t                        next_id_ = 1;
    uint64_t                        next_seq_ = 1;
    bool                            closed_ = false;
    std::string                     closed_why_;
    int                             closed_status_ = RAD_E_STATE;
};

}  /* namespace gen */
}  /* namespace rad */
