#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace beyondmimic {

// Hardware-independent preparation planner. All arrays use SDK motor order.
template<std::size_t N> struct PreparationConfig {
    using Values = std::array<float, N>;
    Values standing_kp{}, standing_kd{}, policy_kp{}, policy_kd{};
    Values lower{}, upper{}, effort{};
    float pose_duration = 3, gain_duration = 0.5f, settle_timeout = 2;
    float max_target_speed = 0.5f, max_joint_speed = 0.5f, max_error = 0.25f;
    float dt = 0.02f;
};

template<std::size_t N> class Preparation {
public:
    using Values = std::array<float, N>;
    struct Command { Values q{}, kp{}, kd{}; bool ready = false; };

    Preparation(PreparationConfig<N> cfg, Values start, Values goal)
        : cfg_(cfg), start_(start), goal_(goal), previous_(start) {
        if (!positive(cfg_.pose_duration) || cfg_.pose_duration > 10 ||
            !positive(cfg_.gain_duration) || cfg_.gain_duration > 5 ||
            !positive(cfg_.settle_timeout) || cfg_.settle_timeout > 10 ||
            !positive(cfg_.max_target_speed) || cfg_.max_target_speed > 2 ||
            !positive(cfg_.max_joint_speed) || !positive(cfg_.max_error) ||
            !positive(cfg_.dt)) throw std::runtime_error("Invalid preparation timing/limits");
        pose_duration_ = cfg_.pose_duration;
        for (std::size_t i = 0; i < N; ++i) {
            if (!std::isfinite(start[i]) || !std::isfinite(goal[i]) ||
                !positive(cfg_.standing_kp[i]) || !positive(cfg_.policy_kp[i]) ||
                !nonnegative(cfg_.standing_kd[i]) || !nonnegative(cfg_.policy_kd[i]) ||
                !positive(cfg_.effort[i]) || !std::isfinite(cfg_.lower[i]) ||
                !std::isfinite(cfg_.upper[i]) || cfg_.lower[i] + 0.02f >= cfg_.upper[i] ||
                start[i] < cfg_.lower[i] || start[i] > cfg_.upper[i] ||
                goal[i] < cfg_.lower[i] || goal[i] > cfg_.upper[i])
                throw std::runtime_error("Invalid preparation posture/PD/limits");
            // Cubic smoothstep has peak slope 1.5; extend the requested duration
            // if needed instead of exceeding the configured joint target speed.
            pose_duration_ = std::max(pose_duration_, 1.5f * std::abs(goal[i]-start[i]) / cfg_.max_target_speed);
        }
    }

    Command update(float elapsed, const Values& q, const Values& dq) {
        if (!std::isfinite(elapsed) || elapsed < previous_time_)
            throw std::runtime_error("Invalid preparation elapsed time");
        previous_time_ = elapsed;
        Command cmd;
        const float pose_alpha = smooth(elapsed / pose_duration_);
        const float gain_alpha = smooth((elapsed-pose_duration_) / cfg_.gain_duration);
        float error = 0, speed = 0;
        for (std::size_t i = 0; i < N; ++i) {
            if (!std::isfinite(q[i]) || !std::isfinite(dq[i]))
                throw std::runtime_error("Invalid preparation motor feedback");
            cmd.kp[i] = cfg_.standing_kp[i] + gain_alpha*(cfg_.policy_kp[i]-cfg_.standing_kp[i]);
            cmd.kd[i] = cfg_.standing_kd[i] + gain_alpha*(cfg_.policy_kd[i]-cfg_.standing_kd[i]);
            const float desired = start_[i] + pose_alpha*(goal_[i]-start_[i]);
            const float center = q[i] + cmd.kd[i]*dq[i]/cmd.kp[i];
            const float radius = cfg_.effort[i]/cmd.kp[i];
            const float step = cfg_.max_target_speed*cfg_.dt;
            const float lo = std::max({cfg_.lower[i]+0.01f, previous_[i]-step, center-radius});
            const float hi = std::min({cfg_.upper[i]-0.01f, previous_[i]+step, center+radius});
            if (lo > hi) throw std::runtime_error("Preparation position/speed/effort constraints conflict");
            cmd.q[i] = std::clamp(desired, lo, hi);
            error = std::max(error, std::abs(q[i]-goal_[i]));
            speed = std::max(speed, std::abs(dq[i]));
        }
        previous_ = cmd.q;
        // Readiness is measured after the gain blend, under training PD. Never
        // bypass the first-frame check just because interpolation time elapsed.
        if (elapsed >= pose_duration_ + cfg_.gain_duration &&
            error <= cfg_.max_error && speed <= cfg_.max_joint_speed) {
            if (settled_since_ < 0) settled_since_ = elapsed;
            cmd.ready = elapsed-settled_since_ >= 0.2f;
        } else settled_since_ = -1;
        if (!cmd.ready && elapsed > pose_duration_ + cfg_.gain_duration + cfg_.settle_timeout)
            throw std::runtime_error("Preparation did not settle at motion first frame");
        return cmd;
    }

    float pose_duration() const { return pose_duration_; }

private:
    static bool positive(float v) { return std::isfinite(v) && v > 0; }
    static bool nonnegative(float v) { return std::isfinite(v) && v >= 0; }
    static float smooth(float t) { t = std::clamp(t, 0.0f, 1.0f); return t*t*(3-2*t); }
    PreparationConfig<N> cfg_;
    Values start_, goal_, previous_;
    float pose_duration_ = 0, settled_since_ = -1, previous_time_ = 0;
};
} // namespace beyondmimic
