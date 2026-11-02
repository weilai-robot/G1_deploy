#include "State_HugWBC.h"

State_HugWBC::State_HugWBC(int state, std::string name) : FSMState(state, name) {
    auto cfg = param::config["FSM"][name];
    auto path = param::proj_dir / cfg["policy_dir"].as<std::string>();
    auto package = hugwbc::load_package(path);
    checkpoint_sha_ = package.checkpoint_sha256;
    policy_sha_ = package.policy_sha256;
    core_ = std::make_unique<hugwbc::Core>(package.cfg);
    // Runner verifies policy.onnx against package.policy_sha256 before use.
    runner_ = std::make_unique<hugwbc::Runner>(path / "policy.onnx", package.policy_sha256);
    lowstate->set_timeout_ms(100);

    auto command = cfg["command"];
    if (!command || command.size() != hugwbc::CMD)
        throw std::runtime_error("HugWBC requires an explicit 11-value fixed command profile");
    for (int i = 0; i < hugwbc::CMD; ++i) {
        command_[i] = command[i].as<float>();
        if (!std::isfinite(command_[i]))
            throw std::runtime_error("Non-finite HugWBC command value");
    }
    // The accepted R41/R42 contract runs with the interrupt flag at 0; the
    // reference arms are disclosed by the state name/package, not the flag.
    if (command_[10] != 0.f)
        throw std::runtime_error("HugWBC command interrupt flag must stay 0");
    standing_ = cfg["standing"].as<bool>(false);
    max_tilt_ = cfg["max_tilt"].as<float>(0.85f);
    max_start_error_ = cfg["max_start_error"].as<float>(0.6f);
    if (!(max_tilt_ > 0 && max_tilt_ < 1.5f) || !(max_start_error_ > 0 && max_start_error_ < 1.5f))
        throw std::runtime_error("Invalid HugWBC limits");
    registered_checks.insert(registered_checks.begin(),
                             {[this] { return stopped_.load(); },
                              FSMStringMap.right.at("Passive")});
    spdlog::info("HugWBC loaded: package={} checkpoint_sha256={} policy_sha256={} "
                 "imu_source={} fixed_command_vx={} standing={}",
                 path.filename().string(), checkpoint_sha_, policy_sha_,
                 package.cfg.imu_source == hugwbc::ImuSource::Pelvis ? "pelvis" : "torso",
                 command_[0], standing_);
}

hugwbc::Snapshot State_HugWBC::snapshot() {
    std::lock_guard<std::mutex> lock(lowstate->mutex_);
    hugwbc::Snapshot result;
    const auto& msg = lowstate->msg_;
    for (int i = 0; i < hugwbc::SDK; ++i) {
        result.q[i] = msg.motor_state()[i].q();
        result.dq[i] = msg.motor_state()[i].dq();
    }
    const auto& quat = msg.imu_state().quaternion();
    const auto& gyro = msg.imu_state().gyroscope();
    result.orientation = hugwbc::Quat(quat[0], quat[1], quat[2], quat[3]);  // wxyz
    result.angular_velocity = hugwbc::Vec(gyro[0], gyro[1], gyro[2]);
    hugwbc::validate_snapshot(result);
    result.orientation.normalize();
    return result;
}

void State_HugWBC::enter() {
    exit();
    stopped_ = false;
    try {
        if (lowstate->isTimeout()) throw std::runtime_error("LowState timeout at HugWBC start");
        const auto& cfg = core_->config();
        auto current = snapshot();
        // Deployment startup wrapper (documented, closed-loop pending): compare
        // against the policy start pose (q0 legs/waist, arm centre, wrists 0),
        // then reset with zero action history and zero gait phase. The
        // max_start_error threshold is an un-validated deployment value.
        auto start_pose = [this, &cfg](int i) {
            if (i == 15 || i == 19) return cfg.arm_pitch_bias;
            if (i == 16) return cfg.arm_shoulder_roll;
            if (i == 20) return -cfg.arm_shoulder_roll;
            if (i == 18 || i == 22) return cfg.arm_elbow;
            return cfg.default_q[i];  // legs, waist, shoulder yaw
        };
        float error = 0;
        for (int i = 0; i < hugwbc::J; ++i) {
            const int sdk = cfg.model_to_sdk[i];
            error = std::max(error, std::abs(current.q[sdk] - start_pose(i)));
        }
        for (int w = 0; w < hugwbc::WRISTS; ++w) {
            const int sdk = cfg.wrist_sdk_indices[w];
            error = std::max(error, std::abs(current.q[sdk] - cfg.wrist_target[sdk]));
        }
        if (error > max_start_error_)
            throw std::runtime_error("Initial posture differs from the HugWBC start pose by " +
                                     std::to_string(error) + " rad; return to FixStand");
        core_->reset();
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            targets_ = current.q;
            last_result_ = std::chrono::steady_clock::now();
        }
        // PD switch in one package: trained gains on the 23 policy motors,
        // deployment hold gains on the six wrists (no trained wrist PD exists).
        for (int i = 0; i < hugwbc::J; ++i) {
            auto& motor = lowcmd->msg_.motor_cmd()[cfg.model_to_sdk[i]];
            motor.q() = current.q[cfg.model_to_sdk[i]];
            motor.kp() = cfg.kp[i];
            motor.kd() = cfg.kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
        for (int w = 0; w < hugwbc::WRISTS; ++w) {
            const int sdk = cfg.wrist_sdk_indices[w];
            auto& motor = lowcmd->msg_.motor_cmd()[sdk];
            motor.q() = cfg.wrist_target[sdk];
            motor.kp() = cfg.wrist_kp[sdk];
            motor.kd() = cfg.wrist_kd[sdk];
            motor.dq() = 0;
            motor.tau() = 0;
        }
        running_ = true;
        worker_ = std::thread(&State_HugWBC::policy_loop, this);
    } catch (const std::exception& e) {
        stop(e.what());
        damping();
    }
}

void State_HugWBC::stop(const std::string& reason) {
    running_ = false;
    if (!stopped_.exchange(true)) spdlog::warn("HugWBC -> Passive: {}", reason);
}

void State_HugWBC::policy_loop() {
    try {
        using Clock = std::chrono::steady_clock;
        const auto dt =
            std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(core_->config().dt));
        auto deadline = Clock::now();
        while (running_) {
            if (lowstate->isTimeout()) throw std::runtime_error("LowState timeout");
            const auto sensor_time = Clock::now();
            auto current = snapshot();
            const float tilt =
                std::acos(std::clamp((current.orientation * hugwbc::Vec::UnitZ()).z(), -1.0f, 1.0f));
            if (tilt > max_tilt_) throw std::runtime_error("Pelvis tilt limit exceeded");
            core_->prepare(current, command_, standing_);
            const auto raw = runner_->act(core_->history());
            const auto inference_done = Clock::now();
            auto tick = core_->complete(current, command_, standing_, raw);
            if (!running_) break;
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                targets_ = tick.sdk_targets;
                last_result_ = inference_done;
            }
            if (tick.index % 50 == 0) log_tick(tick, std::chrono::duration<double>(
                                                           inference_done - sensor_time).count() * 1e3);
            deadline += dt;
            if (Clock::now() > deadline + dt)
                throw std::runtime_error("Policy missed its 20 ms deadline; no catch-up inference");
            std::this_thread::sleep_until(deadline);
        }
        if (running_) stop("HugWBC worker exiting; damping enabled");
    } catch (const std::exception& e) {
        stop(e.what());
    }
}

void State_HugWBC::log_tick(const hugwbc::Tick& tick, double inference_ms) {
    // 1 Hz diagnostic line from the policy thread only; never blocks the FSM
    // send thread. Full-resolution logging stays in the offline/verification path.
    const auto& f = tick.frame;
    spdlog::info(
        "HugWBC tick={} dt_ms={:.2f} g={:.4f} phase=[{:.4f},{:.4f}] clock=[{:.3f},{:.3f}] "
        "cmd_vx={:.2f} standing={} raw[0,3,12]=[{:.3f},{:.3f},{:.3f}] hybrid[0,3,12]=[{:.3f},{:.3f},{:.3f}] "
        "arm_ref_pitch=[{:.3f},{:.3f}] limited={} targets[0,3,9,15,19]={:.3f},{:.3f},{:.3f},{:.3f},{:.3f}",
        tick.index, inference_ms, f.g, f.phase_left, f.phase_right, f.clock_left, f.clock_right,
        command_[0], standing_, tick.raw_action[0], tick.raw_action[3], tick.raw_action[12],
        tick.hybrid_action[0], tick.hybrid_action[3], tick.hybrid_action[12],
        tick.arm_reference[0], tick.arm_reference[4], tick.limited_count, tick.sdk_targets[0],
        tick.sdk_targets[3], tick.sdk_targets[9], tick.sdk_targets[15], tick.sdk_targets[19]);
}

void State_HugWBC::damping() {
    for (int i = 0; i < hugwbc::SDK; ++i) {
        auto& motor = lowcmd->msg_.motor_cmd()[i];
        motor.kp() = 0;
        motor.kd() = 3;
        motor.dq() = 0;
        motor.tau() = 0;
    }
}

void State_HugWBC::run() {
    if (stopped_) {
        damping();
        return;
    }
    std::lock_guard<std::mutex> lock(target_mutex_);
    if (std::chrono::steady_clock::now() - last_result_ > std::chrono::milliseconds(100)) {
        stop("Policy output timeout");
        damping();
        return;
    }
    // Publish the latest complete package only: q targets here, PD/wrist gains
    // were written once at enter(). No inference, history or clock updates here.
    for (int i = 0; i < hugwbc::SDK; ++i) lowcmd->msg_.motor_cmd()[i].q() = targets_[i];
}

void State_HugWBC::exit() {
    running_ = false;
    if (worker_.joinable()) worker_.join();
}
