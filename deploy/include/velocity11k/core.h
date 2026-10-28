#pragma once

#include "velocity11k/contract.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace velocity11k {
using Joints = std::array<float,J>;
using Observation = std::array<float,O>;
using Triple = std::array<float,3>;

// q/dq use SDK order. The network, default pose and joint limits use policy order.
struct Snapshot {
    Joints q{}, dq{};
    std::array<float,4> quat{1,0,0,0}; // pelvis IMU, wxyz
    Triple omega{}, command{};         // body gyro; physical vx/vy/wz commands
};

template<class Values>
inline void finite(const Values& values) {
    for(float value:values) if(!std::isfinite(value)) throw std::runtime_error("Non-finite velocity policy data");
}

inline Triple gravity(const Snapshot& snapshot) {
    finite(snapshot.quat);
    float norm=0;
    for(float v:snapshot.quat) norm+=v*v;
    if(norm<0.5f || norm>1.5f) throw std::runtime_error("Invalid pelvis IMU quaternion norm");
    const float inverse=1.0f/std::sqrt(norm);
    const float w=snapshot.quat[0]*inverse, x=snapshot.quat[1]*inverse;
    const float y=snapshot.quat[2]*inverse, z=snapshot.quat[3]*inverse;
    return {-2*(x*z-w*y), -2*(y*z+w*x), -(1-2*(x*x+y*y))};
}

inline Triple command(const Snapshot& snapshot) {
    finite(snapshot.command);
    Triple result{};
    for(int i=0;i<3;++i) result[i]=std::clamp(snapshot.command[i],COMMAND_LOW[i],COMMAND_HIGH[i]);
    return result;
}

template<int N>
class History {
public:
    void reset(const std::array<float,N>& frame) { frames_.fill(frame); }
    void push(const std::array<float,N>& frame) {
        for(int h=1;h<H;++h) frames_[h-1]=frames_[h];
        frames_.back()=frame;
    }
    void append(Observation& result,int& offset) const {
        for(const auto& frame:frames_) for(float value:frame) result[offset++]=value;
    }
private:
    std::array<std::array<float,N>,H> frames_{};
};

class Core {
public:
    void reset(const Snapshot& snapshot) {
        last_action_.fill(0);
        const auto frame=features(snapshot);
        ang_.reset(frame.ang); gravity_.reset(frame.grav); cmd_.reset(frame.cmd);
        q_.reset(frame.q); dq_.reset(frame.dq); action_.reset(last_action_);
        initialized_=true;
    }
    Observation observe(const Snapshot& snapshot) {
        if(!initialized_) throw std::runtime_error("Velocity history must be reset before inference");
        const auto frame=features(snapshot);
        ang_.push(frame.ang); gravity_.push(frame.grav); cmd_.push(frame.cmd);
        q_.push(frame.q); dq_.push(frame.dq); action_.push(last_action_);
        Observation result{};
        int offset=0;
        ang_.append(result,offset); gravity_.append(result,offset); cmd_.append(result,offset);
        q_.append(result,offset); dq_.append(result,offset); action_.append(result,offset);
        return result;
    }
    Joints targets(const Joints& actions) {
        finite(actions);
        Joints result{};
        for(int i=0;i<J;++i)
            result[SDK_MAP[i]]=std::clamp(DEFAULT_Q[i]+ACTION_SCALE*actions[i],LOWER[i],UPPER[i]);
        last_action_=actions; // raw network output, before target position clipping
        return result;
    }
private:
    struct Features { Triple ang,grav,cmd; Joints q,dq; };
    Features features(const Snapshot& snapshot) const {
        finite(snapshot.q); finite(snapshot.dq); finite(snapshot.omega);
        Features result{};
        result.grav=gravity(snapshot); result.cmd=command(snapshot);
        for(int i=0;i<3;++i) result.ang[i]=ANGULAR_SCALE*snapshot.omega[i];
        for(int i=0;i<J;++i) {
            result.q[i]=snapshot.q[SDK_MAP[i]]-DEFAULT_Q[i];
            result.dq[i]=JOINT_VEL_SCALE*snapshot.dq[SDK_MAP[i]];
        }
        return result;
    }
    History<3> ang_,gravity_,cmd_;
    History<J> q_,dq_,action_;
    Joints last_action_{};
    bool initialized_=false;
};
} // namespace velocity11k
