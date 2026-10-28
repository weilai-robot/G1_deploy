#include "State_Velocity11k.h"
#include <ctime>
#include <iomanip>

State_Velocity11k::State_Velocity11k(int state,std::string name) : FSMState(state,name) {
    const auto cfg=param::config["FSM"][name];
    auto package=std::filesystem::path(cfg["policy_dir"].as<std::string>());
    if(package.is_relative()) package=param::proj_dir/package;
    // Intentionally no latest-directory fallback: this state only runs the frozen 11k package.
    runner_=std::make_unique<velocity11k::Runner>(package/"exported/policy.onnx");
    max_start_error_=cfg["max_start_error"].as<float>(0.25f);
    max_tilt_=cfg["max_tilt"].as<float>(1.0f);
    joint_log_enabled_=cfg["joint_log"].as<bool>(true);
    if(!std::isfinite(max_start_error_) || max_start_error_<=0 ||
       !std::isfinite(max_tilt_) || max_tilt_<=0 || max_tilt_>1.0f)
        throw std::runtime_error("Invalid 11k policy startup/orientation protection");
    const auto stand=param::config["FSM"]["FixStand"];
    const auto kp=stand["kp"].as<std::vector<float>>();
    const auto kd=stand["kd"].as<std::vector<float>>();
    const auto poses=stand["qs"].as<std::vector<std::vector<float>>>();
    if(kp.size()!=velocity11k::J || kd.size()!=velocity11k::J || poses.empty() || poses.back().size()!=velocity11k::J)
        throw std::runtime_error("FixStand must contain the complete 11k SDK pose and gains");
    for(int i=0;i<velocity11k::J;++i) {
        const int sdk=velocity11k::SDK_MAP[i];
        if(!std::isfinite(kp[sdk]) || !std::isfinite(kd[sdk]) || !std::isfinite(poses.back()[sdk]) ||
           std::abs(kp[sdk]-velocity11k::KP[i])>1e-6f || std::abs(kd[sdk]-velocity11k::KD[i])>1e-6f ||
           std::abs(poses.back()[sdk]-velocity11k::DEFAULT_Q[i])>1e-6f)
            throw std::runtime_error("FixStand pose/gains differ from the compiled 11k contract");
    }
    lowstate->set_timeout_ms(100); // same transport/output watchdog as the existing deployment state
    registered_checks.insert(registered_checks.begin(),{[this]{return stopped_.load();},FSMStringMap.right.at("Passive")});
    spdlog::info("Velocity11k loaded: checkpoint={} obs={} actions={} dt={}",velocity11k::CHECKPOINT_SHA256,velocity11k::O,velocity11k::J,velocity11k::DT);
}

velocity11k::Snapshot State_Velocity11k::snapshot() {
    std::lock_guard<std::mutex> lock(lowstate->mutex_);
    velocity11k::Snapshot result;
    const auto& msg=lowstate->msg_;
    for(int i=0;i<velocity11k::J;++i) {
        result.q[i]=msg.motor_state()[i].q();
        result.dq[i]=msg.motor_state()[i].dq();
    }
    for(int i=0;i<4;++i) result.quat[i]=msg.imu_state().quaternion()[i];
    for(int i=0;i<3;++i) result.omega[i]=msg.imu_state().gyroscope()[i];
    // Preserve the original velocity controller's stick signs and physical units.
    result.command={lowstate->joystick.ly(),-lowstate->joystick.lx(),-lowstate->joystick.rx()};
    return result;
}

void State_Velocity11k::enter() {
    exit();
    stopped_=false;
    try {
        if(lowstate->isTimeout()) throw std::runtime_error("LowState timeout at 11k policy start");
        auto current=snapshot();
        core_.reset(current); // repeat current features, clear last-action history on EVERY entry
        float error=0;
        for(int i=0;i<velocity11k::J;++i)
            error=std::max(error,std::abs(current.q[velocity11k::SDK_MAP[i]]-velocity11k::DEFAULT_Q[i]));
        if(error>max_start_error_) throw std::runtime_error("Initial posture differs from 11k default; finish FixStand before entering Velocity");
        if(std::acos(std::clamp(-velocity11k::gravity(current)[2],-1.0f,1.0f))>max_tilt_)
            throw std::runtime_error("Pelvis tilt limit exceeded at startup");
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            targets_=current.q;
            last_result_=std::chrono::steady_clock::now();
        }
        for(int i=0;i<velocity11k::J;++i) {
            const int sdk=velocity11k::SDK_MAP[i];
            auto& motor=lowcmd->msg_.motor_cmd()[sdk];
            motor.q()=current.q[sdk];
            motor.kp()=velocity11k::KP[i]; motor.kd()=velocity11k::KD[i];
            motor.dq()=0; motor.tau()=0;
        }
        open_joint_log();
        running_=true;
        worker_=std::thread(&State_Velocity11k::policy_loop,this);
    } catch(const std::exception& error) { stop(error.what()); damping(); }
}

void State_Velocity11k::open_joint_log() {
    if(!joint_log_enabled_) return;
    std::filesystem::create_directories(param::proj_dir/"log");
    const auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto path=param::proj_dir/"log"/("velocity11k_"+std::to_string(stamp)+".csv");
    joint_log_.clear();
    joint_log_.open(path);
    if(!joint_log_) throw std::runtime_error("Cannot open 11k joint log");
    joint_log_ << std::setprecision(9) << "policy_step,elapsed_s,cmd_vx,cmd_vy,cmd_wz,qw,qx,qy,qz,gyro_x,gyro_y,gyro_z,gravity_x,gravity_y,gravity_z,inference_s";
    for(int i=0;i<velocity11k::J;++i) joint_log_ << ",q_sdk_"<<i<<",dq_sdk_"<<i<<",target_sdk_"<<i;
    for(int i=0;i<velocity11k::J;++i) joint_log_ << ",action_policy_"<<i;
    joint_log_ << '\n';
    spdlog::info("Velocity11k joint log: {}",path.string());
}

void State_Velocity11k::policy_loop() {
    try {
        using Clock=std::chrono::steady_clock;
        const auto dt=std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(velocity11k::DT));
        const auto start=Clock::now();
        auto deadline=start;
        std::size_t step=0;
        while(running_) {
            if(lowstate->isTimeout()) throw std::runtime_error("LowState timeout");
            const auto current=snapshot();
            const auto grav=velocity11k::gravity(current);
            if(std::acos(std::clamp(-grav[2],-1.0f,1.0f))>max_tilt_)
                throw std::runtime_error("Pelvis tilt limit exceeded");
            const auto inference_start=Clock::now();
            const auto obs=core_.observe(current);
            const auto action=runner_->act(obs);
            const auto target=core_.targets(action);
            const auto now=Clock::now();
            const double inference_seconds=std::chrono::duration<double>(now-inference_start).count();
            if(!running_) break;
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                targets_=target;
                last_result_=now;
            }
            if(joint_log_) {
                joint_log_<<step<<','<<std::chrono::duration<double>(now-start).count();
                for(float value:velocity11k::command(current)) joint_log_<<','<<value;
                for(float value:current.quat) joint_log_<<','<<value;
                for(float value:current.omega) joint_log_<<','<<value;
                for(float value:grav) joint_log_<<','<<value;
                joint_log_<<','<<inference_seconds;
                for(int i=0;i<velocity11k::J;++i) joint_log_<<','<<current.q[i]<<','<<current.dq[i]<<','<<target[i];
                for(float value:action) joint_log_<<','<<value;
                joint_log_<<'\n';
                if(step%50==0) joint_log_.flush();
                if(!joint_log_) throw std::runtime_error("Writing 11k joint log failed");
            }
            ++step;
            deadline+=dt;
            if(Clock::now()>deadline+dt) throw std::runtime_error("11k policy missed its control deadline");
            std::this_thread::sleep_until(deadline);
        }
    } catch(const std::exception& error) { stop(error.what()); }
}

void State_Velocity11k::stop(const std::string& reason) {
    running_=false;
    if(!stopped_.exchange(true)) spdlog::warn("Velocity11k -> Passive: {}",reason);
}
void State_Velocity11k::damping() {
    for(int i=0;i<velocity11k::J;++i) {
        auto& motor=lowcmd->msg_.motor_cmd()[i];
        motor.kp()=0; motor.kd()=3; motor.dq()=0; motor.tau()=0;
    }
}
void State_Velocity11k::run() {
    if(stopped_) { damping(); return; }
    std::lock_guard<std::mutex> lock(target_mutex_);
    if(std::chrono::steady_clock::now()-last_result_>std::chrono::milliseconds(100)) {
        stop("11k policy output timeout"); damping(); return;
    }
    for(int i=0;i<velocity11k::J;++i) lowcmd->msg_.motor_cmd()[i].q()=targets_[i];
}
void State_Velocity11k::exit() {
    running_=false;
    if(worker_.joinable()) worker_.join();
    if(joint_log_.is_open()) joint_log_.close();
}
