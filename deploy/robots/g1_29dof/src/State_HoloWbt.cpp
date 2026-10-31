#include "State_HoloWbt.h"

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
    if (start_frame_ >= motion_->frames() || !std::isfinite(duration) || duration <= 0 || duration > 300 ||
        !(max_start_error_ > 0 && max_start_error_ < 0.8f) || !(max_tilt_ > 0 && max_tilt_ < 1.5f) ||
        !(max_target_speed_ > 0 && max_target_speed_ <= 20))
        throw std::runtime_error("Invalid holo_wbt segment/limits");
    end_frame_ = std::min(motion_->frames(), start_frame_ + static_cast<std::size_t>(duration / core_->config().dt));
    if (end_frame_ <= start_frame_) throw std::runtime_error("Empty holo_wbt segment");
    // 正常播完交给 Velocity（零速站立），异常中断才进 Passive 阻尼；顺序决定优先级。
    registered_checks.insert(registered_checks.begin(), {[this]{ return stopped_.load(); }, FSMStringMap.right.at("Passive")});
    registered_checks.insert(registered_checks.begin(), {[this]{ return completed_.load(); }, FSMStringMap.right.at("Velocity")});
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
    exit();
    stopped_ = false;
    completed_ = false;
    try {
        if (lowstate->isTimeout()) throw std::runtime_error("LowState timeout at policy start");
        auto current = snapshot();
        const auto ref = motion_->frame(start_frame_);
        const auto& cfg = core_->config();
        float error = 0;
        for (int i = 0; i < holowbt::J; ++i) {
            const int sdk = cfg.sdk_indices[i];
            error = std::max(error, std::abs(current.q[sdk] - ref.q[i]));
        }
        if (error > max_start_error_)
            throw std::runtime_error("Initial posture differs from reference by " + std::to_string(error) + " rad; return to FixStand");
        core_->reset(current);
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            targets_ = current.q;
            last_result_ = std::chrono::steady_clock::now();
        }
        for (int i = 0; i < holowbt::J; ++i) {
            auto& motor = lowcmd->msg_.motor_cmd()[cfg.sdk_indices[i]];
            motor.q() = current.q[cfg.sdk_indices[i]];
            motor.kp() = cfg.kp[i];
            motor.kd() = cfg.kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
        running_ = true;
        worker_ = std::thread(&State_HoloWbt::policy_loop, this);
    } catch (const std::exception& e) {
        stop(e.what());
        damping();
    }
}

void State_HoloWbt::stop(const std::string& reason) {
    running_ = false;
    if (!stopped_.exchange(true)) spdlog::warn("HoloWbt -> Passive: {}", reason);
}

void State_HoloWbt::policy_loop() {
    try {
        using Clock = std::chrono::steady_clock;
        const auto dt = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(core_->config().dt));
        auto deadline = Clock::now();
        for (std::size_t frame = start_frame_; running_ && frame < end_frame_; ++frame) {
            if (lowstate->isTimeout()) throw std::runtime_error("LowState timeout");
            auto current = snapshot();
            const float tilt = std::acos(std::clamp((current.orientation * holowbt::Vec::UnitZ()).z(), -1.0f, 1.0f));
            if (tilt > max_tilt_) throw std::runtime_error("Pelvis tilt limit exceeded");
            auto ref = motion_->frame(frame);
            auto obs = core_->observe(current, ref);
            auto action = runner_->act(obs);
            auto target = core_->targets(action, current, max_target_speed_);
            if (!running_) break;
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                targets_ = target;
                last_result_ = Clock::now();
            }
            if ((frame - start_frame_) % 50 == 0) {
                spdlog::info("HoloWbt frame={} clipped={}", frame, core_->clipped());
            }
            deadline += dt;
            if (Clock::now() > deadline + dt) throw std::runtime_error("Policy missed its 20 ms deadline");
            std::this_thread::sleep_until(deadline);
        }
        if (running_) {
            completed_ = true;
            stop("Segment completed; handing over to Velocity");
        }
    } catch (const std::exception& e) {
        stop(e.what());
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
    if (stopped_) {
        // 正常完成时不再发阻尼，交给 Velocity 接管；异常停止才阻尼。
        if (!completed_) damping();
        return;
    }
    std::lock_guard<std::mutex> lock(target_mutex_);
    if (std::chrono::steady_clock::now() - last_result_ > std::chrono::milliseconds(100)) {
        stop("Policy output timeout");
        damping();
        return;
    }
    for (int i = 0; i < holowbt::J; ++i) lowcmd->msg_.motor_cmd()[i].q() = targets_[i];
}

void State_HoloWbt::exit() {
    running_ = false;
    if (worker_.joinable()) worker_.join();
}
