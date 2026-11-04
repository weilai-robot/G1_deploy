#include "State_HoloWbt.h"

#include <mutex>

State_HoloWbt::State_HoloWbt(int state, std::string name) : FSMState(state, name) {
    auto cfg = param::config["FSM"][name];
    auto path = param::proj_dir / cfg["policy_dir"].as<std::string>();
    core_ = std::make_unique<holowbt::Core>(holowbt::load_config(path / "deploy.yaml"));
    motion_ = std::make_unique<holowbt::Motion>((path / "reference.bin").string());
    runner_ = std::make_unique<holowbt::Runner>(path / "policy.onnx");
    lowstate->set_timeout_ms(100);
    start_frame_ = cfg["start_frame"].as<std::size_t>(0);
    const float duration = cfg["duration_s"].as<float>(10.0f);
    max_start_error_ = cfg["max_start_error"].as<float>(0.25f);
    max_tilt_ = cfg["max_tilt"].as<float>(0.85f);
    max_target_speed_ = cfg["max_target_speed"].as<float>(8.0f);
    blend_s_ = cfg["blend_s"].as<float>(2.5f);
    blend_rate_ = cfg["blend_rate"].as<float>(1.5f);
    prepose_s_ = cfg["prepose_s"].as<float>(3.0f);
    prepose_only_ = cfg["prepose_only"].as<bool>(false);
    if (start_frame_ >= motion_->frames() || !std::isfinite(duration) || duration <= 0 || duration > 300 ||
        !(max_start_error_ > 0 && max_start_error_ < 0.9f) || !(max_tilt_ > 0 && max_tilt_ < 1.5f) ||
        !(max_target_speed_ > 0 && max_target_speed_ <= 20) || !(blend_s_ >= 0 && blend_s_ <= 10) ||
        !(blend_rate_ > 0 && blend_rate_ <= 8) || !(prepose_s_ >= 0 && prepose_s_ <= 10))
        throw std::runtime_error("Invalid holo_wbt segment/limits");
    end_frame_ = std::min(motion_->frames(), start_frame_ + static_cast<std::size_t>(duration / core_->config().dt));
    if (end_frame_ <= start_frame_) throw std::runtime_error("Empty holo_wbt segment");
    blend_limit_ticks_ = std::max(1, static_cast<int>(std::lround(blend_s_ / core_->config().dt)));
    // 预热 ONNX 会话：首次 Run 含图优化与内存分配，可能远超控制周期，会触发看门狗直接阻尼。
    {
        holowbt::Snapshot warm;
        warm.q.fill(0.f);
        warm.dq.fill(0.f);
        warm.orientation.setIdentity();
        warm.angular_velocity.setZero();
        runner_->act(core_->observe(warm, motion_->frame(start_frame_)));
    }
    // 正常播完交给 Velocity（零速站立），异常中断才进 Passive 阻尼；顺序决定优先级。
    registered_checks.insert(registered_checks.begin(), {[this]{ return stopped_; }, FSMStringMap.right.at("Passive")});
    registered_checks.insert(registered_checks.begin(), {[this]{ return completed_; }, FSMStringMap.right.at("Velocity")});
    spdlog::info("HoloWbt loaded: {} obs / {} actions, {} frames, no leg odometry",
                 holowbt::O, holowbt::J, motion_->frames());
}

holowbt::Snapshot State_HoloWbt::snapshot() {
    std::lock_guard<std::mutex> lock(lowstate->mutex_);
    holowbt::Snapshot result;
    const auto& msg = lowstate->msg_;
    for (int i = 0; i < holowbt::J; ++i) {
        result.q[i] = msg.motor_state()[i].q();
        result.dq[i] = msg.motor_state()[i].dq();
    }
    const auto& quat = msg.imu_state().quaternion();
    const auto& gyro = msg.imu_state().gyroscope();
    result.orientation = holowbt::Quat(quat[0], quat[1], quat[2], quat[3]);
    result.angular_velocity = holowbt::Vec(gyro[0], gyro[1], gyro[2]);
    holowbt::validate_snapshot(result);
    result.orientation.normalize();
    return result;
}

void State_HoloWbt::enter() {
    stopped_ = true;
    completed_ = false;
    try_enter();
}

bool State_HoloWbt::try_enter() {
    try {
        if (lowstate->isTimeout()) throw std::runtime_error("LowState timeout at policy start");
        auto current = snapshot();
        const auto ref = motion_->frame(start_frame_);
        const auto& cfg = core_->config();
        float error = 0;
        int worst = 0;
        for (int i = 0; i < holowbt::J; ++i) {
            const float joint_error = std::abs(current.q[cfg.sdk_indices[i]] - ref.q[i]);
            if (joint_error > error) { error = joint_error; worst = i; }
        }
        spdlog::info("HoloWbt start posture error {:.3f} rad (limit {:.2f}), worst joint #{} ({}) ref {:.3f} actual {:.3f}, blend {:.2f} s at {:.2f} rad/s",
                     error, max_start_error_, worst, cfg.sdk_indices[worst], ref.q[worst],
                     current.q[cfg.sdk_indices[worst]], blend_s_, blend_rate_);
        if (error > max_start_error_)
            throw std::runtime_error("Initial posture differs from reference by " + std::to_string(error) + " rad; return to FixStand");
        for (int i = 0; i < holowbt::J; ++i) {
            auto& motor = lowcmd->msg_.motor_cmd()[cfg.sdk_indices[i]];
            motor.q() = current.q[cfg.sdk_indices[i]];
            motor.kp() = cfg.kp[i];
            motor.kd() = cfg.kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
        for (int i = 0; i < holowbt::J; ++i) prepose_goal_[cfg.sdk_indices[i]] = ref.q[i];
        for (int i = 0; i < holowbt::J; ++i) {
            prepose_goal_[cfg.sdk_indices[i]] = std::clamp(prepose_goal_[cfg.sdk_indices[i]],
                                                           cfg.lower[i] + 0.01f, cfg.upper[i] - 0.01f);
        }
        prepose_from_ = current.q;
        preposing_ = prepose_s_ > 0 && error > 0.05f;
        preposed_ = !preposing_ && prepose_only_;
        prepose_tick_ = 0;
        prepose_ticks_ = std::max(1, static_cast<int>(std::lround(prepose_s_ / cfg.dt)));
        if (!preposing_) core_->reset(current, ref);
        targets_ = current.q;
        last_command_ = current.q;
        frame_ = start_frame_;
        blend_tick_ = 0;
        blending_ = blend_s_ > 0;
        last_result_ = next_step_ = std::chrono::steady_clock::now();
        stopped_ = false;
        spdlog::info("HoloWbt engaged: frames {}..{} at {:.0f} Hz{}", start_frame_, end_frame_ - 1, 1.0f / cfg.dt,
                     preposing_ ? ", pre-positioning to the reference start pose" : "");
        return true;
    } catch (const std::exception& e) {
        stopped_ = true;
        spdlog::warn("HoloWbt -> Passive: {}", e.what());
        damping();
        return false;
    }
}

void State_HoloWbt::stop(const std::string& reason) {
    if (!stopped_) spdlog::warn("HoloWbt -> Passive: {}", reason);
    stopped_ = true;
}

void State_HoloWbt::step_policy() {
    try {
        if (lowstate->isTimeout()) throw std::runtime_error("LowState timeout");
        auto current = snapshot();
        const float tilt = std::acos(std::clamp((current.orientation * holowbt::Vec::UnitZ()).z(), -1.0f, 1.0f));
        if (tilt > max_tilt_) throw std::runtime_error("Pelvis tilt limit exceeded (tilt=" + std::to_string(tilt) + " rad)");
        // 预摆位：先用平滑插值把机器人摆到参考首帧的姿态，再让策略接管。
        if (preposing_) {
            const float alpha = std::min(1.0f, static_cast<float>(++prepose_tick_) / static_cast<float>(prepose_ticks_));
            const float s = alpha * alpha * (3.0f - 2.0f * alpha);
            holowbt::Joints cmd{};
            for (int i = 0; i < holowbt::J; ++i)
                cmd[i] = (1.0f - s) * prepose_from_[i] + s * prepose_goal_[i];
            last_command_ = cmd;
            targets_ = cmd;
            last_result_ = std::chrono::steady_clock::now();
            if (prepose_tick_ >= prepose_ticks_) {
                preposing_ = false;
                if (prepose_only_) {
                    preposed_ = true;
                    spdlog::info("HoloWbt pre-position held; press [R1 + X] to start the motion, [R1 + A] to abort");
                    return;
                }
                core_->reset(current, motion_->frame(start_frame_));
                core_->adopt_target(cmd);
                frame_ = start_frame_;
                blend_tick_ = 0;
                blending_ = blend_s_ > 0;
                spdlog::info("HoloWbt pre-position done; policy engaged at frame {}", start_frame_);
            } else if (prepose_tick_ % 25 == 0) {
                spdlog::info("HoloWbt pre-positioning {}/{}", prepose_tick_, prepose_ticks_);
            }
            return;
        }
        if (preposed_) {
            last_result_ = std::chrono::steady_clock::now();  // 保持阶段也要喂看门狗
            return;
        }
        auto ref = motion_->frame(frame_);
        auto obs = core_->observe(current, ref);
        auto action = runner_->act(obs);
        auto policy_target = core_->targets(action, current, max_target_speed_);
        // 过渡阶段：参考帧停在首帧，指令以 blend_rate 的关节速度上限逼近策略目标。
        holowbt::Joints command = policy_target;
        if (blending_) {
            const float max_step = blend_rate_ * core_->config().dt;
            float remaining = 0;
            for (int i = 0; i < holowbt::J; ++i) {
                const float delta = std::clamp(policy_target[i] - last_command_[i], -max_step, max_step);
                command[i] = last_command_[i] + delta;
                remaining = std::max(remaining, std::abs(policy_target[i] - command[i]));
            }
            if (remaining < 0.05f || ++blend_tick_ > blend_limit_ticks_) {
                blending_ = false;
                spdlog::info("HoloWbt blend done after {} ticks", blend_tick_);
            }
        } else {
            ++frame_;
        }
        last_command_ = command;
        targets_ = command;
        last_result_ = std::chrono::steady_clock::now();
        if (frame_ % 50 == 0) {
            spdlog::info("HoloWbt frame={} clipped={} blend={}", frame_, core_->clipped(), blending_ ? "on" : "off");
        }
        if (!blending_ && frame_ >= end_frame_) {
            completed_ = true;
            stopped_ = true;
            spdlog::info("HoloWbt segment completed; handing over to Velocity");
        }
    } catch (const std::exception& e) {
        stop(e.what());
    } catch (...) {
        stop("Unknown policy failure");
    }
}

void State_HoloWbt::damping() {
    for (int i = 0; i < holowbt::J; ++i) {
        auto& motor = lowcmd->msg_.motor_cmd()[i];
        motor.kp() = 0;
        motor.kd() = 3;
        motor.dq() = 0;
        motor.tau() = 0;
    }
}

void State_HoloWbt::run() {
    using Clock = std::chrono::steady_clock;
    if (stopped_) {
        // 正常完成时不再发阻尼，交给 Velocity 接管；异常停止才阻尼。
        if (!completed_) damping();
        return;
    }
    const auto now = Clock::now();
    if (now - last_result_ > std::chrono::milliseconds(100)) {
        stop("Policy output timeout");
        damping();
        return;
    }
    if (now >= next_step_) {
        step_policy();
        const auto dt = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(core_->config().dt));
        next_step_ += dt;
        if (next_step_ < now) next_step_ = now + dt;  // 落后时重新对齐，不补跑
    }
    if (stopped_) {
        if (!completed_) damping();
        return;
    }
    for (int i = 0; i < holowbt::J; ++i) lowcmd->msg_.motor_cmd()[i].q() = targets_[i];
}

void State_HoloWbt::exit() {
    stopped_ = true;
}
