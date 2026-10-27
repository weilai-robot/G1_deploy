#include "State_BeyondMimic.h"

State_BeyondMimic::State_BeyondMimic(int state,std::string name) : FSMState(state,name) {
    auto cfg=param::config["FSM"][name];
    auto path=param::proj_dir/cfg["policy_dir"].as<std::string>();
    core_=std::make_unique<beyondmimic::Core>(beyondmimic::load_config(path/"deploy.yaml"));
    motion_=std::make_unique<beyondmimic::Motion>((path/"reference.bin").string());
    runner_=std::make_unique<beyondmimic::Runner>(path/"policy.onnx");
    lowstate->set_timeout_ms(100);
    start_frame_=cfg["start_frame"].as<std::size_t>(0);
    const float duration=cfg["duration_s"].as<float>(10.0f);
    max_start_error_=cfg["max_start_error"].as<float>(0.25f);
    max_tilt_=cfg["max_tilt"].as<float>(0.85f);
    max_target_speed_=cfg["max_target_speed"].as<float>(8.0f);
    max_anchor_error_=cfg["max_anchor_error"].as<float>(0.35f);
    if(start_frame_>=motion_->frames() || !std::isfinite(duration) || duration<=0 || duration>300 ||
       !(max_start_error_>0 && max_start_error_<0.8f) || !(max_tilt_>0 && max_tilt_<1.5f) ||
       !(max_target_speed_>0 && max_target_speed_<=20) || !(max_anchor_error_>0 && max_anchor_error_<=1))
        throw std::runtime_error("Invalid BeyondMimic segment/limits");
    end_frame_=std::min(motion_->frames(),start_frame_+static_cast<std::size_t>(duration/core_->config().dt));
    if(end_frame_<=start_frame_) throw std::runtime_error("Empty BeyondMimic segment");
    registered_checks.insert(registered_checks.begin(),{[this]{return stopped_.load();},FSMStringMap.right.at("Passive")});
    spdlog::info("BeyondMimic loaded: 160 obs / 29 actions, {} frames, leg odometry",motion_->frames());
}

beyondmimic::Snapshot State_BeyondMimic::snapshot() {
    std::lock_guard<std::mutex> lock(lowstate->mutex_);
    beyondmimic::Snapshot result;
    const auto& msg=lowstate->msg_;
    for(int i=0;i<beyondmimic::J;++i) {
        result.q[i]=msg.motor_state()[i].q();
        result.dq[i]=msg.motor_state()[i].dq();
    }
    const auto& quat=msg.imu_state().quaternion();
    const auto& gyro=msg.imu_state().gyroscope();
    result.orientation=beyondmimic::Quat(quat[0],quat[1],quat[2],quat[3]);
    result.angular_velocity=beyondmimic::Vec(gyro[0],gyro[1],gyro[2]);
    beyondmimic::validate_snapshot(result);
    result.orientation.normalize();
    return result;
}

void State_BeyondMimic::enter() {
    exit();
    stopped_=false;
    try {
        if(lowstate->isTimeout()) throw std::runtime_error("LowState timeout at policy start");
        auto current=snapshot();
        const auto ref=motion_->frame(start_frame_);
        const auto& cfg=core_->config();
        float error=0;
        for(int i=0;i<beyondmimic::J;++i) {
            const int sdk=cfg.sdk_indices[i];
            error=std::max(error,std::abs(current.q[sdk]-ref.q[i]));
        }
        if(error>max_start_error_) throw std::runtime_error("Initial posture differs from reference by "+std::to_string(error)+" rad; return to FixStand");
        core_->reset(current,ref);
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            targets_=current.q;
            last_result_=std::chrono::steady_clock::now();
        }
        for(int i=0;i<beyondmimic::J;++i) {
            auto& motor=lowcmd->msg_.motor_cmd()[cfg.sdk_indices[i]];
            motor.q()=current.q[cfg.sdk_indices[i]];
            motor.kp()=cfg.kp[i]; motor.kd()=cfg.kd[i];
            motor.dq()=0; motor.tau()=0;
        }
        running_=true;
        worker_=std::thread(&State_BeyondMimic::policy_loop,this);
    } catch(const std::exception& e) {
        stop(e.what()); damping();
    }
}

void State_BeyondMimic::stop(const std::string& reason) {
    running_=false;
    if(!stopped_.exchange(true)) spdlog::warn("BeyondMimic -> Passive: {}",reason);
}

void State_BeyondMimic::policy_loop() {
    try {
        using Clock=std::chrono::steady_clock;
        const auto dt=std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(core_->config().dt));
        auto deadline=Clock::now();
        const auto first=motion_->frame(start_frame_);
        const float reference_floor=std::min(first.positions[3].z(),first.positions[6].z());
        for(std::size_t frame=start_frame_;running_ && frame<end_frame_;++frame) {
            if(lowstate->isTimeout()) throw std::runtime_error("LowState timeout");
            auto current=snapshot();
            const float tilt=std::acos(std::clamp((current.orientation*beyondmimic::Vec::UnitZ()).z(),-1.0f,1.0f));
            if(tilt>max_tilt_) throw std::runtime_error("Pelvis tilt limit exceeded");
            auto ref=motion_->frame(frame);
            if(std::min(ref.positions[3].z(),ref.positions[6].z())>reference_floor+0.10f)
                throw std::runtime_error("Reference flight/jump is unsupported by contact leg odometry");
            auto obs=core_->observe(current,ref);
            if(beyondmimic::Vec(obs[58],obs[59],obs[60]).norm()>max_anchor_error_)
                throw std::runtime_error("Estimated torso/reference position error limit exceeded");
            auto action=runner_->act(obs);
            auto target=core_->targets(action,current,max_target_speed_);
            if(!running_) break;
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                targets_=target;
                last_result_=Clock::now();
            }
            if((frame-start_frame_)%50==0) {
                const auto& p=core_->position(); const auto& v=core_->velocity_b();
                spdlog::info("BeyondMimic frame={} p=[{:.3f},{:.3f},{:.3f}] v=[{:.3f},{:.3f},{:.3f}] limited={}",
                             frame,p.x(),p.y(),p.z(),v.x(),v.y(),v.z(),core_->clipped());
            }
            deadline+=dt;
            if(Clock::now()>deadline+dt) throw std::runtime_error("Policy missed its 20 ms deadline");
            std::this_thread::sleep_until(deadline);
        }
        if(running_) stop("Segment completed; damping enabled (keep fall-arrest support attached)");
    } catch(const std::exception& e) { stop(e.what()); }
}

void State_BeyondMimic::damping() {
    for(int i=0;i<beyondmimic::J;++i) {
        auto& motor=lowcmd->msg_.motor_cmd()[i];
        motor.kp()=0; motor.kd()=3; motor.dq()=0; motor.tau()=0;
    }
}

void State_BeyondMimic::run() {
    if(stopped_) { damping(); return; }
    std::lock_guard<std::mutex> lock(target_mutex_);
    if(std::chrono::steady_clock::now()-last_result_>std::chrono::milliseconds(100)) {
        stop("Policy output timeout"); damping(); return;
    }
    for(int i=0;i<beyondmimic::J;++i) lowcmd->msg_.motor_cmd()[i].q()=targets_[i];
}

void State_BeyondMimic::exit() {
    running_=false;
    if(worker_.joinable()) worker_.join();
}
