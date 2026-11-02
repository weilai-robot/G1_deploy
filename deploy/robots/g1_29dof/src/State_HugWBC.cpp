#include "State_HugWBC.h"
#include <iomanip>
#include <limits>

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
    tick_log_enabled_ = cfg["tick_log"].as<bool>(true);
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
        open_tick_log();
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
            // Tilt gate on the SAME pelvis frame the actor receives (not on the
            // raw IMU, which is the torso frame in imu_source=torso mode).
            hugwbc::Quat pelvis;
            hugwbc::Vec pelvis_gyro, pelvis_gravity;
            core_->pelvis_frame(current, pelvis, pelvis_gyro, pelvis_gravity);
            const float tilt = std::acos(std::clamp((pelvis * hugwbc::Vec::UnitZ()).z(), -1.0f, 1.0f));
            if (tilt > max_tilt_) throw std::runtime_error("Pelvis tilt limit exceeded");
            last_snapshot_ = current;
            last_pelvis_gyro_ = pelvis_gyro;
            last_pelvis_gravity_ = pelvis_gravity;
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
            const auto target_ready_time = Clock::now();
            log_tick(tick, sensor_time, inference_done, target_ready_time);
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

void State_HugWBC::open_tick_log() {
    if (!tick_log_enabled_) return;
    try {
        const auto dir = param::proj_dir / "log";
        std::filesystem::create_directories(dir);
        const auto path = dir / ("hugwbc_ticks_" +
                                 std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                                                    std::chrono::system_clock::now()
                                                        .time_since_epoch())
                                                    .count()) +
                                 ".csv");
        tick_log_.open(path.string());
        if (!tick_log_) return;
        tick_log_ << std::setprecision(std::numeric_limits<double>::max_digits10);
        // Snapshot/target-ready times are host-side markers, not device sample
        // or DDS send timestamps. Preserve precision for per-tick differences.
        tick_log_ << "index,t_snapshot_begin_us,t_infer_done_us,t_target_ready_us,g,phase_left,phase_right,"
                     "clock_left,clock_right,standing";
        for (int i = 0; i < hugwbc::CMD; ++i) tick_log_ << ",cmd" << i;
        for (int i = 0; i < hugwbc::J; ++i) tick_log_ << ",raw" << i;
        for (int i = 0; i < hugwbc::J; ++i) tick_log_ << ",hybrid" << i;
        for (int i = 0; i < hugwbc::J; ++i) tick_log_ << ",clipped" << i;
        for (int i = 0; i < hugwbc::ARMS; ++i) tick_log_ << ",arm_ref" << i;
        for (int i = 0; i < hugwbc::SDK; ++i) tick_log_ << ",target" << i;
        for (int i = 0; i < hugwbc::SDK; ++i) tick_log_ << ",q" << i;
        for (int i = 0; i < hugwbc::SDK; ++i) tick_log_ << ",dq" << i;
        tick_log_ << ",imu_qw,imu_qx,imu_qy,imu_qz";
        for (int i = 0; i < 3; ++i) tick_log_ << ",imu_gyro" << i;
        for (int i = 0; i < 3; ++i) tick_log_ << ",pelvis_gyro" << i;
        for (int i = 0; i < 3; ++i) tick_log_ << ",pelvis_gravity" << i;
        for (int i = 0; i < hugwbc::WRISTS; ++i) tick_log_ << ",wrist_actual" << i;
        tick_log_ << ",limited,guard_violations\n";
        spdlog::info("HugWBC per-tick log: {}", path.string());
    } catch (const std::exception& e) {
        spdlog::warn("HugWBC tick log unavailable: {}", e.what());
    }
}

void State_HugWBC::log_tick(const hugwbc::Tick& tick, Clock::time_point sensor_time,
                            Clock::time_point inference_done, Clock::time_point target_ready_time) {
    // Full per-tick record for sim2real analysis (plan section 11), written by
    // the policy thread only; the FSM send loop never touches this stream.
    if (tick_log_.is_open()) {
        const auto us = [](Clock::time_point t) {
            return std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count();
        };
        tick_log_ << tick.index << ',' << us(sensor_time) << ',' << us(inference_done) << ','
                  << us(target_ready_time) << ',' << tick.frame.g << ',' << tick.frame.phase_left << ','
                  << tick.frame.phase_right << ',' << tick.frame.clock_left << ','
                  << tick.frame.clock_right << ',' << (standing_ ? 1 : 0);
        for (float v : command_) tick_log_ << ',' << v;
        for (float v : tick.raw_action) tick_log_ << ',' << v;
        for (float v : tick.hybrid_action) tick_log_ << ',' << v;
        for (float v : tick.clipped_action) tick_log_ << ',' << v;
        for (float v : tick.arm_reference) tick_log_ << ',' << v;
        for (float v : tick.sdk_targets) tick_log_ << ',' << v;
        for (float v : last_snapshot_.q) tick_log_ << ',' << v;
        for (float v : last_snapshot_.dq) tick_log_ << ',' << v;
        const auto& imu = last_snapshot_.orientation;
        tick_log_ << ',' << imu.w() << ',' << imu.x() << ',' << imu.y() << ',' << imu.z();
        for (int i = 0; i < 3; ++i) tick_log_ << ',' << last_snapshot_.angular_velocity[i];
        for (int i = 0; i < 3; ++i) tick_log_ << ',' << last_pelvis_gyro_[i];
        for (int i = 0; i < 3; ++i) tick_log_ << ',' << last_pelvis_gravity_[i];
        const auto& wrist_ids = core_->config().wrist_sdk_indices;
        for (int w = 0; w < hugwbc::WRISTS; ++w) tick_log_ << ',' << last_snapshot_.q[wrist_ids[w]];
        tick_log_ << ',' << tick.limited_count << ',' << tick.guard_violations << '\n';
    }
    if (tick.index % 50 != 0) return;  // 1 Hz console summary
    const double inference_ms =
        std::chrono::duration<double>(inference_done - sensor_time).count() * 1e3;
    const auto& f = tick.frame;
    spdlog::info(
        "HugWBC tick={} dt_ms={:.2f} g={:.4f} phase=[{:.4f},{:.4f}] clock=[{:.3f},{:.3f}] "
        "cmd_vx={:.2f} standing={} raw[0,3,12]=[{:.3f},{:.3f},{:.3f}] hybrid[0,3,12]=[{:.3f},{:.3f},{:.3f}] "
        "arm_ref_pitch=[{:.3f},{:.3f}] limited={} violated={} targets[0,3,9,15,19]={:.3f},{:.3f},{:.3f},{:.3f},{:.3f}",
        tick.index, inference_ms, f.g, f.phase_left, f.phase_right, f.clock_left, f.clock_right,
        command_[0], standing_, tick.raw_action[0], tick.raw_action[3], tick.raw_action[12],
        tick.hybrid_action[0], tick.hybrid_action[3], tick.hybrid_action[12],
        tick.arm_reference[0], tick.arm_reference[4], tick.limited_count, tick.guard_violations,
        tick.sdk_targets[0], tick.sdk_targets[3], tick.sdk_targets[9], tick.sdk_targets[15],
        tick.sdk_targets[19]);
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
    if (tick_log_.is_open()) {
        tick_log_.flush();
        tick_log_.close();
    }
}
