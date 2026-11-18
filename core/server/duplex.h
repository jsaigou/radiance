/* core/server/duplex.h -- the model-free side of the duplex engine contract (docs/DUPLEX-ENGINE.md).
 *
 * A tenant is one thing the engine can hold a conversation for; a lane is one conversation. The
 * endpoint in duplex.cpp owns the wire, the lane limits and the lane's lifetime, and knows nothing
 * of a model. A tenant takes a start request and returns a lane that answers one input frame with
 * one reply, in order.
 */
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace server {

using DuplexJson = nlohmann::ordered_json;

struct DuplexTenantInfo {
    std::string name;
    bool        text = false;
    int         max_lanes = 1;
};

struct DuplexStart {
    std::string contract;
    std::string session_id;
    bool        text = false;
    int64_t     max_minutes = 0;
    DuplexJson  card = DuplexJson::object();
    std::string reference_id;
    std::string transcript;
    std::vector<uint8_t> reference;
};

struct DuplexReply {
    std::vector<float> audio;   /* empty, or exactly 3200 samples at 24 kHz */
    std::string        text;    /* empty when there is no text delta */
};

class DuplexLane {
public:
    virtual ~DuplexLane() = default;
    virtual DuplexReply step(uint32_t seq, const std::vector<float>& samples) = 0;
    virtual DuplexReply flush() = 0;
    virtual void close() = 0;
};

class DuplexTenant {
public:
    virtual ~DuplexTenant() = default;
    virtual DuplexTenantInfo info() const = 0;
    /* A lane, or a negative status with `why` (server-facing, never sent to the client). */
    virtual int open(const DuplexStart& start, std::unique_ptr<DuplexLane>* out,
                     std::string* why) = 0;
};

/* The two test tenants of the contract's section 8, off unless --duplex-test-tenants enables
 * them. Both take no model. */
std::shared_ptr<DuplexTenant> make_echo_tenant(int max_lanes);
std::shared_ptr<DuplexTenant> make_echo1_tenant(int max_lanes);

}  /* namespace server */
}  /* namespace rad */
