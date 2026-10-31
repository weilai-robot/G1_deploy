#pragma once

// No DDS or inference dependency: the exact observation/odometry code is shared
// by the real controller and the offline MuJoCo verification executable.
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace beyondmimic {
constexpr int J = 29, B = 14, O = 160, P = 154, RecordSize = 2 * J + 7 * B;
using Vec = Eigen::Vector3f;
using Quat = Eigen::Quaternionf;
using Joints = std::array<float, J>;

struct Snapshot {
    Joints q{}, dq{};  // SDK order, never policy order
    Quat orientation = Quat::Identity();  // pelvis, wxyz on input
    Vec angular_velocity = Vec::Zero(); // pelvis body frame
};
struct Segment {
    int motor_index = -1;
    Vec xyz = Vec::Zero(), rpy = Vec::Zero(), axis = Vec::UnitZ();
};
using Chain = std::vector<Segment>;
inline Eigen::Isometry3f forward_kinematics(const Chain& chain, const Joints& q) {
    auto pose = Eigen::Isometry3f::Identity();
    for (const auto& joint : chain) {
        pose.translate(joint.xyz);
        pose.rotate(Eigen::AngleAxisf(joint.rpy.z(), Vec::UnitZ()) *
                    Eigen::AngleAxisf(joint.rpy.y(), Vec::UnitY()) *
                    Eigen::AngleAxisf(joint.rpy.x(), Vec::UnitX()));
        if (joint.motor_index >= 0) {
            pose.rotate(Eigen::AngleAxisf(q.at(joint.motor_index), joint.axis.normalized()));
        }
    }
    return pose;
}
inline Quat yaw(const Quat& q) {
    const float angle = std::atan2(2 * (q.w()*q.z()+q.x()*q.y()), 1-2*(q.y()*q.y()+q.z()*q.z()));
    return Quat(Eigen::AngleAxisf(angle, Vec::UnitZ()));
}
inline void validate_snapshot(const Snapshot& s) {
    if (!s.orientation.coeffs().allFinite() || std::abs(s.orientation.norm()-1) > 0.05f ||
        !s.angular_velocity.allFinite()) throw std::runtime_error("Invalid pelvis IMU data");
    for (int i=0; i<J; ++i) {
        if (!std::isfinite(s.q[i]) || !std::isfinite(s.dq[i])) throw std::runtime_error("Invalid motor data");
    }
}

struct Reference {
    Joints q{}, dq{}; // training/policy order
    std::array<Vec, B> positions;
    std::array<Quat, B> orientations;
};
class Motion {
public:
    explicit Motion(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open reference: " + path);
        const auto size = file.tellg();
        file.seekg(0);
        char magic[8];
        std::uint32_t joints=0, bodies=0;
        file.read(magic, 8);
        file.read(reinterpret_cast<char*>(&frames_), 4);
        file.read(reinterpret_cast<char*>(&joints), 4);
        file.read(reinterpret_cast<char*>(&bodies), 4);
        file.read(reinterpret_cast<char*>(&dt_), 4);
        const std::uint16_t endian = 1;
        if (*reinterpret_cast<const char*>(&endian) != 1 || std::memcmp(magic, "BMG1V1\0\0", 8) ||
            joints != J || bodies != B || frames_ < 2 || frames_ > 10000000 ||
            std::abs(dt_ - 0.02f) > 1e-6f ||
            size != static_cast<std::streamoff>(24 + std::uint64_t(frames_)*RecordSize*sizeof(float)))
            throw std::runtime_error("Invalid/incompatible reference table");
        data_.resize(std::size_t(frames_)*RecordSize);
        file.read(reinterpret_cast<char*>(data_.data()), data_.size()*sizeof(float));
        if (!file) throw std::runtime_error("Truncated reference table");
        for (float v : data_) if (!std::isfinite(v)) throw std::runtime_error("Non-finite reference");
    }
    Reference frame(std::size_t index) const {
        if (index >= frames_) throw std::out_of_range("Reference motion ended");
        const float* p = data_.data() + index*RecordSize;
        Reference result;
        std::copy(p, p+J, result.q.begin()); p += J;
        std::copy(p, p+J, result.dq.begin()); p += J;
        for (auto& pos : result.positions) { pos = Vec(p[0],p[1],p[2]); p += 3; }
        for (auto& quat : result.orientations) {
            quat = Quat(p[0],p[1],p[2],p[3]); p += 4;
            if (std::abs(quat.norm()-1) > 0.05f) throw std::runtime_error("Invalid reference quaternion");
            quat.normalize();
        }
        return result;
    }
    std::size_t frames() const { return frames_; }
    float dt() const { return dt_; }
private:
    std::uint32_t frames_ = 0;
    float dt_ = 0;
    std::vector<float> data_;
};

struct Config {
    std::array<int,J> sdk_indices{};
    Joints default_q{}, kp{}, kd{}, action_scale{}, lower{}, upper{}, effort{};
    Chain torso, left_foot, right_foot;
    float dt = 0.02f;
    void validate() const {
        std::set<int> ids(sdk_indices.begin(), sdk_indices.end());
        if (ids.size()!=J || *ids.begin()!=0 || *ids.rbegin()!=J-1 || std::abs(dt-0.02f)>1e-6f)
            throw std::runtime_error("Invalid SDK joint permutation or policy dt");
        for (int i=0; i<J; ++i) {
            if (!(kp[i]>0 && kd[i]>=0 && action_scale[i]>0 && effort[i]>0 && lower[i]<upper[i]))
                throw std::runtime_error("Invalid joint control parameters");
        }
        for (const auto* chain : {&torso, &left_foot, &right_foot}) {
            if (chain->empty()) throw std::runtime_error("Missing kinematic chain");
            for (const auto& j : *chain) if (j.motor_index < -1 || j.motor_index >= J ||
                !j.xyz.allFinite() || !j.rpy.allFinite() || !j.axis.allFinite() || j.axis.norm()<0.9f)
                throw std::runtime_error("Invalid kinematic segment");
        }
    }
};

// Contact kinematic odometry, using only pelvis IMU and motor encoders.
// It assumes at least one non-slipping foot on level ground. It is NOT an
// absolute localisation system and does not use reference position as feedback.
class LegOdometry {
public:
    explicit LegOdometry(const Config& cfg) : cfg_(cfg) {}
    void reset(const Snapshot& s, const Vec& initial_position) {
        position_ = initial_position;
        velocity_b_.setZero();
        auto feet = foot_positions(s.q);
        floor_ = std::min((s.orientation*feet[0]).z(), (s.orientation*feet[1]).z()) + position_.z();
        for (int i=0;i<2;++i) locks_[i] = position_ + s.orientation*feet[i];
        contact_ = {true,true};
        initialised_ = true;
    }
    void update(const Snapshot& s) {
        if (!initialised_) throw std::runtime_error("Odometry not initialised");
        const auto feet = foot_positions(s.q);
        Joints ahead = s.q;
        constexpr float eps = 1e-4f;
        for (int i=0;i<J;++i) ahead[i] += eps*s.dq[i];
        auto next_feet = foot_positions(ahead);
        const auto rotation = s.orientation.toRotationMatrix();
        const Vec prediction = position_ + rotation*velocity_b_*cfg_.dt;
        const std::array<Vec,2> world{rotation*feet[0], rotation*feet[1]};
        const float lowest = std::min(world[0].z(),world[1].z());
        Vec v = Vec::Zero(), p = Vec::Zero();
        float total = 0;
        for (int i=0;i<2;++i) {
            // Height hysteresis; never lock the visibly lifted foot.
            const float threshold = contact_[i] ? 0.045f : 0.025f;
            const bool stance = world[i].z()-lowest < threshold;
            if (stance) {
                if (!contact_[i]) {
                    locks_[i] = prediction + world[i];
                    locks_[i].z() = floor_;
                }
                const float weight = std::max(0.05f, 1-(world[i].z()-lowest)/0.05f);
                const Vec r_dot = (next_feet[i]-feet[i])/eps;
                v += weight*(-s.angular_velocity.cross(feet[i])-r_dot);
                p += weight*(locks_[i]-world[i]);
                total += weight;
            }
            contact_[i] = stance;
        }
        if (total <= 0) throw std::runtime_error("No support foot for leg odometry");
        velocity_b_ = 0.6f*velocity_b_ + 0.4f*v/total;
        position_ = 0.2f*prediction + 0.8f*p/total;
        if (!position_.allFinite() || !velocity_b_.allFinite() || velocity_b_.norm()>5)
            throw std::runtime_error("Invalid leg odometry estimate");
    }
    const Vec& position() const { return position_; }
    const Vec& velocity_b() const { return velocity_b_; }
private:
    std::array<Vec,2> foot_positions(const Joints& q) const {
        return {forward_kinematics(cfg_.left_foot,q).translation(), forward_kinematics(cfg_.right_foot,q).translation()};
    }
    const Config& cfg_;
    Vec position_ = Vec::Zero(), velocity_b_ = Vec::Zero();
    std::array<Vec,2> locks_;
    std::array<bool,2> contact_{};
    float floor_ = 0;
    bool initialised_ = false;
};

class Core {
public:
    explicit Core(Config cfg) : cfg_(std::move(cfg)), odometry_(cfg_) { cfg_.validate(); }
    void reset(const Snapshot& s, const Reference& ref) {
        validate_snapshot(s);
        alignment_ = yaw(s.orientation)*yaw(ref.orientations[0]).conjugate();
        translation_ = Vec(0,0,ref.positions[0].z()) - alignment_*ref.positions[0];
        odometry_.reset(s, Vec(0,0,ref.positions[0].z()));
        last_action_.fill(0);
        last_target_ = s.q;
    }
    // [OMA] No odometry initialization/update or position/linear-velocity input.
    void reset_proprioceptive(const Snapshot& s, const Reference& ref) {
        validate_snapshot(s);
        alignment_ = yaw(s.orientation)*yaw(ref.orientations[0]).conjugate();
        last_action_.fill(0); last_target_=s.q;
    }
    std::array<float,P> observe_proprioceptive(const Snapshot& s,const Reference& ref) {
        validate_snapshot(s);
        const Quat torso_q=s.orientation*Quat(forward_kinematics(cfg_.torso,s.q).rotation());
        const auto rot=(torso_q.conjugate()*alignment_*ref.orientations[7]).toRotationMatrix();
        std::array<float,P> obs{};int k=0;
        for(float v:ref.q) obs[k++]=v;
        for(float v:ref.dq) obs[k++]=v;
        for(int i=0;i<3;++i) for(int j=0;j<2;++j) obs[k++]=rot(i,j);
        for(int i=0;i<3;++i) obs[k++]=s.angular_velocity[i];
        for(int i=0;i<J;++i) obs[k++]=s.q[cfg_.sdk_indices[i]]-cfg_.default_q[i];
        for(int i=0;i<J;++i) obs[k++]=s.dq[cfg_.sdk_indices[i]];
        for(float v:last_action_) obs[k++]=v;
        for(float v:obs) if(!std::isfinite(v)) throw std::runtime_error("Invalid proprioceptive observation");
        return obs;
    }
    std::array<float,O> observe(const Snapshot& s, const Reference& ref, bool update_odometry=true,
                              const Vec* measured_position=nullptr, const Vec* measured_velocity_b=nullptr) {
        validate_snapshot(s);
        if (update_odometry) odometry_.update(s);
        const Vec root_pos = measured_position ? *measured_position : odometry_.position();
        const Vec velocity = measured_velocity_b ? *measured_velocity_b : odometry_.velocity_b();
        const auto torso = forward_kinematics(cfg_.torso,s.q);
        const Quat torso_q = s.orientation*Quat(torso.rotation());
        const Vec torso_pos = root_pos+s.orientation*torso.translation();
        const Vec ref_pos = alignment_*ref.positions[7]+translation_;
        const Quat ref_q = alignment_*ref.orientations[7];
        const Vec relative = torso_q.conjugate()*(ref_pos-torso_pos);
        const auto rot = (torso_q.conjugate()*ref_q).toRotationMatrix();
        std::array<float,O> obs{};
        int k=0;
        auto append = [&](const auto& values) { for (float v : values) obs[k++] = v; };
        append(ref.q); append(ref.dq);
        for(int i=0;i<3;++i) obs[k++] = relative[i];
        // Isaac Lab matrix[..., :2].reshape: row-major first two columns.
        for(int i=0;i<3;++i) for(int j=0;j<2;++j) obs[k++] = rot(i,j);
        for(int i=0;i<3;++i) obs[k++] = velocity[i];
        for(int i=0;i<3;++i) obs[k++] = s.angular_velocity[i];
        for(int i=0;i<J;++i) obs[k++] = s.q[cfg_.sdk_indices[i]]-cfg_.default_q[i];
        for(int i=0;i<J;++i) obs[k++] = s.dq[cfg_.sdk_indices[i]];
        append(last_action_);
        for(float v : obs) if (!std::isfinite(v)) throw std::runtime_error("Invalid observation");
        return obs;
    }
    Joints targets(const Joints& action, const Snapshot& state, float max_speed=8.0f) {
        Joints sdk{};
        clipped_ = 0;
        for(int i=0;i<J;++i) {
            const int id = cfg_.sdk_indices[i];
            if (!std::isfinite(action[i]) || std::abs(action[i])>100)
                throw std::runtime_error("Invalid policy action");
            float desired = cfg_.default_q[i]+cfg_.action_scale[i]*action[i];
            float limited = std::clamp(desired, cfg_.lower[i]+0.01f, cfg_.upper[i]-0.01f);
            limited = std::clamp(limited, last_target_[id]-max_speed*cfg_.dt, last_target_[id]+max_speed*cfg_.dt);
            // Bound the requested PD torque using the training actuator effort.
            const float center = state.q[id]+cfg_.kd[i]*state.dq[id]/cfg_.kp[i];
            const float radius = cfg_.effort[i]/cfg_.kp[i];
            limited = std::clamp(limited, center-radius, center+radius);
            limited = std::clamp(limited, cfg_.lower[i]+0.01f, cfg_.upper[i]-0.01f);
            clipped_ += std::abs(desired-limited)>1e-5f;
            sdk[id] = limited;
        }
        last_target_ = sdk;
        last_action_ = action; // training observes raw preceding action, not clipped targets
        return sdk;
    }
    const Vec& position() const { return odometry_.position(); }
    const Vec& velocity_b() const { return odometry_.velocity_b(); }
    int clipped() const { return clipped_; }
    const Config& config() const { return cfg_; }
private:
    Config cfg_;
    LegOdometry odometry_;
    Quat alignment_ = Quat::Identity();
    Vec translation_ = Vec::Zero();
    Joints last_action_{}, last_target_{};
    int clipped_ = 0;
};
} // namespace beyondmimic
