#pragma once

// HugWBC R41/R42 numeric core: SDK state mapping, pelvis observation frame,
// gait clock, reference arms, hybrid action synthesis and 29-DOF command output.
// No DDS and no inference dependency: the exact code runs both in the real FSM
// state and in the hardware-free offline/verification executable.
//
// Frozen training contract (see tools/export_hugwbc.py header for sources):
// 23-dim policy joints in training order, [1,5,88] history input (old -> new),
// 0.02 s policy tick, per-frame scaling then clip to +/-100, hybrid action
// history, eight reference-arm overrides, six wrist motors held by the
// deployment (not part of the model contract).

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace hugwbc {
constexpr int J = 23;         // policy/model joints (training order)
constexpr int SDK = 29;       // SDK motors
constexpr int PRO = 75;       // per-frame proprioception slice
constexpr int CMD = 11;       // command vector (incl. interrupt flag at index 10)
constexpr int OBS = PRO + CMD + 2;  // 88: proprioception + cmd11 + gait clock pair
constexpr int HIST = 5;       // history frames, old -> new
constexpr int ARMS = 8;       // reference-overridden arm joints
constexpr int WRISTS = 6;     // SDK wrist motors held by the deployment
constexpr int CLK = 2;        // sin() gait clock pair in the observation

using Vec = Eigen::Vector3f;
using Quat = Eigen::Quaternionf;
using Joints = std::array<float, J>;   // training order
using Motors = std::array<float, SDK>; // SDK order

enum class ImuSource { Pelvis, Torso };

constexpr float TWO_PI = 6.283185307179586f;  // avoid M_PI portability (strict -std=c++17)

struct Snapshot {
    Motors q{}, dq{};                    // SDK order, never policy order
    Quat orientation = Quat::Identity(); // imu frame, wxyz on input
    Vec angular_velocity = Vec::Zero();  // imu frame
};

struct Config {
    std::array<int, J> model_to_sdk{};         // model joint -> SDK motor index
    std::array<int, ARMS> arm_model_indices{}; // [15..22], model order L then R
    std::array<int, WRISTS> wrist_sdk_indices{}; // 19,20,21,26,27,28
    Joints default_q{}, kp{}, kd{}, action_scale{}, torque_limit{}, lower{}, upper{};
    Motors wrist_kp{}, wrist_kd{};             // deployment hold gains, not trained
    Motors wrist_target{};                     // nominal wrist hold position
    std::array<float, CMD> command_scales{};
    float obs_ang_vel_scale = 0.25f;
    float obs_dof_vel_scale = 0.05f;
    float clip_obs = 100.f;
    float clip_action = 100.f;
    // arm reference parameters (accepted manifest / g1r42 controller)
    float arm_pitch_bias = 0.3f;
    float arm_shoulder_roll = 0.27f;
    float arm_elbow = 0.97f;
    float arm_amplitude = 0.09178921571001411f;
    float arm_phase_offset = 0.30418359554562746f;
    float arm_speed_reference = 0.5f;
    bool arm_velocity_feedforward = false;  // accepted execution used false
    float dt = 0.02f;                       // policy tick, not the 1 ms FSM loop
    ImuSource imu_source = ImuSource::Pelvis;
    // Deployment-side protections around (not inside) the trained contract.
    bool torque_guard = true;   // project targets into Kp*(q*-q)-Kd*dq in +/-tau
    float max_target_speed = 0.f;  // rad/s, 0 = disabled (training has no limit)

    void validate() const {
        if (std::abs(dt - 0.02f) > 1e-6f) throw std::runtime_error("Invalid policy dt");
        std::array<bool, SDK> used{};
        for (int i = 0; i < J; ++i) {
            const int sdk = model_to_sdk[i];
            if (sdk < 0 || sdk >= SDK || used[sdk])
                throw std::runtime_error("Invalid SDK joint mapping");
            used[sdk] = true;
        }
        for (int w : wrist_sdk_indices) {
            if (w < 0 || w >= SDK || used[w])
                throw std::runtime_error("Invalid wrist hold mapping");
            used[w] = true;
        }
        // The model is the 23 non-wrist SDK motors, arms included in that count.
        for (int s = 0; s < SDK; ++s)
            if (!used[s]) throw std::runtime_error("SDK motor missing from mapping");
        for (int k = 0; k < ARMS; ++k)
            if (arm_model_indices[k] != 15 + k)
                throw std::runtime_error("Invalid arm reference indices");
        for (int i = 0; i < J; ++i)
            if (!(kp[i] > 0 && kd[i] >= 0 && action_scale[i] > 0 && torque_limit[i] > 0 &&
                  lower[i] < upper[i]))
                throw std::runtime_error("Invalid joint control parameters");
        for (int w = 0; w < WRISTS; ++w) {
            const int s = wrist_sdk_indices[w];
            if (!(wrist_kp[s] > 0 && wrist_kd[s] >= 0))  // relative to Motors indexing
                throw std::runtime_error("Invalid wrist hold gains");
        }
        for (int i = 0; i < CMD; ++i)
            if (!std::isfinite(command_scales[i])) throw std::runtime_error("Invalid command scale");
        if (!(arm_pitch_bias > 0 && arm_shoulder_roll > 0 && arm_elbow > 0 &&
              arm_amplitude > 0 && arm_speed_reference > 0))
            throw std::runtime_error("Invalid arm reference parameters");
    }
};

inline void validate_snapshot(const Snapshot& s) {
    if (!s.orientation.coeffs().allFinite() || std::abs(s.orientation.norm() - 1) > 0.05f ||
        !s.angular_velocity.allFinite())
        throw std::runtime_error("Invalid pelvis IMU data");
    for (int i = 0; i < SDK; ++i)
        if (!std::isfinite(s.q[i]) || !std::isfinite(s.dq[i]))
            throw std::runtime_error("Invalid motor data");
}

// Exact per-side stance/swing remap from _step_contact_targets, including its
// strict-inequality boundary behaviour (u == d leaves the value unchanged).
inline float remap_phase(float u, float duration) {
    if (u < duration) return u * 0.5f / duration;
    if (u > duration) return 0.5f + (u - duration) * 0.5f / (1.0f - duration);
    return u;
}

struct Frame {
    std::array<float, OBS> obs{};   // scaled + clipped, exactly what enters history
    float g = 0;                    // unwrapped-by-support-ratio gait phase [0,1)
    float phase_left = 0;           // wrap(g + cmd[4]); 0 while standing
    float phase_right = 0;          // wrap(g); 0 while standing
    float clock_left = 0;           // sin(2*pi*remap(phase_left))
    float clock_right = 0;          // sin(2*pi*remap(phase_right))
};

struct Tick {
    Frame frame;
    std::array<float, OBS * HIST> history{};  // old -> new, exactly the actor input
    Joints raw_action{};       // actor output
    Joints hybrid_action{};    // arm slots replaced by reference; this enters history
    Joints clipped_action{};   // clip(hybrid, +/-100), what the PD consumes
    Joints q_target{};         // default_q + action_scale * clipped (training order)
    Motors sdk_targets{};      // 23 policy targets + 6 wrist hold targets
    std::array<float, ARMS> arm_reference{};  // rad, arm_model_indices order
    int limited_count = 0;     // deploy-side guard activations this tick
    int guard_violations = 0;  // zero for returned ticks; infeasible ticks throw
    std::uint64_t index = 0;   // policy tick counter since reset
};

class Core {
public:
    explicit Core(Config cfg) : cfg_(std::move(cfg)) { cfg_.validate(); }

    // Explicit deployment startup (documented in the runbook): action history
    // zero, gait phase zero. The next prepare() seeds all HIST slots with the
    // first real frame, mirroring a training episode reset of the history buffer.
    void reset() {
        g_ = 0;
        last_hybrid_.fill(0);
        seeded_ = false;
        tick_index_ = 0;
        last_target_.fill(0);
        first_target_ = true;
    }

    // Advance the gait clock once, build the 88-dim frame and push it into the
    // history exactly once per policy tick.
    Frame prepare(const Snapshot& s, const std::array<float, CMD>& command, bool standing) {
        validate_snapshot(s);
        g_ = std::fmod(g_ + cfg_.dt * command[3], 1.0f);
        Frame frame;
        frame.g = g_;
        // Standing zeroes the foot phases (the global g keeps advancing, as in
        // training); arms drop their periodic term via the walking mask.
        frame.phase_left = standing ? 0.f : std::fmod(g_ + command[4], 1.0f);
        frame.phase_right = standing ? 0.f : g_;
        frame.clock_left = std::sin(TWO_PI * remap_phase(frame.phase_left, command[5]));
        frame.clock_right = std::sin(TWO_PI * remap_phase(frame.phase_right, command[5]));

        Quat orientation;
        Vec angular_velocity, projected_gravity;
        pelvis_frame(s, orientation, angular_velocity, projected_gravity);

        int k = 0;
        for (int i = 0; i < 3; ++i) frame.obs[k++] = angular_velocity[i] * cfg_.obs_ang_vel_scale;
        for (int i = 0; i < 3; ++i) frame.obs[k++] = projected_gravity[i];
        for (int i = 0; i < J; ++i) frame.obs[k++] = s.q[cfg_.model_to_sdk[i]] - cfg_.default_q[i];
        for (int i = 0; i < J; ++i) frame.obs[k++] = s.dq[cfg_.model_to_sdk[i]] * cfg_.obs_dof_vel_scale;
        for (int i = 0; i < J; ++i) frame.obs[k++] = last_hybrid_[i];
        for (int i = 0; i < CMD; ++i) frame.obs[k++] = command[i] * cfg_.command_scales[i];
        frame.obs[k++] = frame.clock_left;
        frame.obs[k++] = frame.clock_right;
        if (k != OBS) throw std::runtime_error("Observation size mismatch");
        for (float& v : frame.obs) {
            if (!std::isfinite(v)) throw std::runtime_error("Invalid observation");
            v = std::clamp(v, -cfg_.clip_obs, cfg_.clip_obs);
        }

        if (seeded_) {
            // Training ObservationBuffer.insert: drop the OLDEST frame, shift
            // newer frames toward index 0, append the new frame at the end.
            std::copy(history_.begin() + OBS, history_.end(), history_.begin());
            std::copy(frame.obs.begin(), frame.obs.end(), history_.end() - OBS);
        } else {
            for (int h = 0; h < HIST; ++h)
                std::copy(frame.obs.begin(), frame.obs.end(), history_.data() + h * OBS);
            seeded_ = true;
        }
        frame_ = frame;
        return frame;
    }

    // Synthesise the hybrid action and SDK command package for one actor output.
    Tick complete(const Snapshot& s, const std::array<float, CMD>& command, bool standing,
                  const Joints& raw_action) {
        Tick tick;
        tick.frame = frame_;
        tick.history = history_;
        tick.index = tick_index_++;
        tick.raw_action = raw_action;
        for (float a : raw_action)
            if (!std::isfinite(a)) throw std::runtime_error("Invalid policy action");

        // Reference arms from the SAME phase as the observation frame above.
        const float speed_scale = std::clamp(command[0] / cfg_.arm_speed_reference, -1.f, 1.f);
        const float walking = standing ? 0.f : 1.f;
        std::array<float, ARMS> ref{};
        // reference order: L pitch, L roll, L yaw, L elbow, R pitch, R roll, R yaw, R elbow
        ref[0] = cfg_.arm_pitch_bias -
                 cfg_.arm_amplitude * std::cos(TWO_PI * frame_.phase_right - cfg_.arm_phase_offset) *
                     speed_scale * walking;
        ref[4] = cfg_.arm_pitch_bias -
                 cfg_.arm_amplitude * std::cos(TWO_PI * frame_.phase_left - cfg_.arm_phase_offset) *
                     speed_scale * walking;
        ref[1] = cfg_.arm_shoulder_roll;
        ref[5] = -cfg_.arm_shoulder_roll;
        ref[2] = 0.f;
        ref[6] = 0.f;
        ref[3] = cfg_.arm_elbow;
        ref[7] = cfg_.arm_elbow;
        tick.arm_reference = ref;

        tick.hybrid_action = raw_action;
        for (int k = 0; k < ARMS; ++k) {
            const int j = cfg_.arm_model_indices[k];
            tick.hybrid_action[j] =
                (ref[k] - cfg_.default_q[j]) / cfg_.action_scale[j];
        }
        for (int i = 0; i < J; ++i) {
            const float v = tick.hybrid_action[i];
            if (!std::isfinite(v)) throw std::runtime_error("Invalid hybrid action");
            tick.clipped_action[i] = std::clamp(v, -cfg_.clip_action, cfg_.clip_action);
            tick.q_target[i] = cfg_.default_q[i] + cfg_.action_scale[i] * tick.clipped_action[i];
        }

        // Map to SDK motors and fill the six wrist hold targets (wrist_target
        // is SDK-indexed, same convention as the loader).
        Motors targets{};
        for (int i = 0; i < J; ++i) targets[cfg_.model_to_sdk[i]] = tick.q_target[i];
        for (int w = 0; w < WRISTS; ++w)
            targets[cfg_.wrist_sdk_indices[w]] = cfg_.wrist_target[cfg_.wrist_sdk_indices[w]];

        tick.limited_count = apply_guards(targets, s);
        tick.guard_violations = guard_violations_;
        tick.sdk_targets = targets;
        last_hybrid_ = tick.hybrid_action;  // history observes the hybrid, pre-clip
        return tick;
    }

    const std::array<float, OBS * HIST>& history() const { return history_; }
    const Config& config() const { return cfg_; }
    std::uint64_t tick_index() const { return tick_index_; }
    bool seeded() const { return seeded_; }

    // Pelvis-frame attitude/gyro/gravity from a snapshot, honouring imu_source.
    // Public so the FSM tilt gate uses the SAME pelvis frame as the actor.
    void pelvis_frame(const Snapshot& s, Quat& orientation, Vec& angular_velocity,
                      Vec& projected_gravity) const {
        if (cfg_.imu_source == ImuSource::Pelvis) {
            orientation = s.orientation;
            angular_velocity = s.angular_velocity;
        } else {
            // Torso IMU: recover the pelvis through the actual waist chain
            // yaw -> roll -> pitch (training URDF origins carry no rotation).
            // R_pelvis_from_torso = Rz(yaw) Rx(roll) Ry(pitch).
            // R_world_from_pelvis = R_world_from_torso * R_pelvis_from_torso^T.
            const Eigen::Matrix3f rz =
                Eigen::AngleAxisf(s.q[12], Vec::UnitZ()).toRotationMatrix();
            const Eigen::Matrix3f rzrx =
                rz * Eigen::AngleAxisf(s.q[13], Vec::UnitX()).toRotationMatrix();
            const Eigen::Matrix3f r_torso_p =
                rzrx * Eigen::AngleAxisf(s.q[14], Vec::UnitY()).toRotationMatrix();
            const Eigen::Matrix3f r_torso = s.orientation.toRotationMatrix();
            const Eigen::Matrix3f r_pelvis = r_torso * r_torso_p.transpose();
            orientation = Quat(r_pelvis);
            // Torso gyro is the torso ABSOLUTE angular velocity in torso
            // coordinates. Pelvis gyro = torso rate - waist relative rate,
            // both expressed in pelvis coordinates (the plain rotation alone
            // leaks waist motion into the pelvis observation).
            const Vec w_torso_in_pelvis = r_torso_p * s.angular_velocity;
            const Vec w_waist_rel =
                Vec::UnitZ() * s.dq[12] + rz * (Vec::UnitX() * s.dq[13]) +
                rzrx * (Vec::UnitY() * s.dq[14]);
            angular_velocity = w_torso_in_pelvis - w_waist_rel;
        }
        orientation.normalize();
        projected_gravity = orientation.toRotationMatrix().transpose() * Vec(0.f, 0.f, -1.f);
    }

private:
    // Deployment-side protections documented in the plan (training computes
    // tau = clip(Kp*(q*-q) - Kd*dq, +/-tau): intersect the position, optional
    // speed and torque-feasible intervals. An EMPTY intersection rejects the
    // entire tick before any target package can be published. The FSM catches
    // the fault and transitions to its damping/Passive path. Do not assume the
    // firmware applies the training torque limit. Wrists have no trained
    // parameters and get no torque guard.
    int apply_guards(Motors& targets, const Snapshot& s) {
        int limited = 0;
        guard_violations_ = 0;
        for (int i = 0; i < J; ++i) {
            const int id = cfg_.model_to_sdk[i];
            float desired = targets[id];
            float lo = cfg_.lower[i] + 0.01f, hi = cfg_.upper[i] - 0.01f;
            if (cfg_.max_target_speed > 0.f && !first_target_) {
                lo = std::max(lo, last_target_[id] - cfg_.max_target_speed * cfg_.dt);
                hi = std::min(hi, last_target_[id] + cfg_.max_target_speed * cfg_.dt);
            }
            if (cfg_.torque_guard) {
                const float center = s.q[id] + cfg_.kd[i] * s.dq[id] / cfg_.kp[i];
                const float radius = cfg_.torque_limit[i] / cfg_.kp[i];
                lo = std::max(lo, center - radius);
                hi = std::min(hi, center + radius);
            }
            if (lo <= hi) {
                const float bounded = std::clamp(desired, lo, hi);
                if (std::abs(bounded - desired) > 1e-5f) ++limited;
                desired = bounded;
            } else {
                ++guard_violations_;
                throw std::runtime_error("Empty target constraint intersection at SDK motor " +
                                         std::to_string(id));
            }
            targets[id] = desired;
            last_target_[id] = desired;
        }
        for (int w = 0; w < WRISTS; ++w) {
            const int id = cfg_.wrist_sdk_indices[w];
            last_target_[id] = targets[id];  // hold target, no trained limits
        }
        first_target_ = false;
        return limited;
    }

    Config cfg_;
    std::array<float, OBS * HIST> history_{};
    Frame frame_;
    Joints last_hybrid_{};
    Motors last_target_{};
    float g_ = 0;
    std::uint64_t tick_index_ = 0;
    int guard_violations_ = 0;
    bool seeded_ = false, first_target_ = true;
};
}  // namespace hugwbc
