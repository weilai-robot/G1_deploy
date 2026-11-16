#include "State_BeyondMimic.h"

State_BeyondMimic::State_BeyondMimic(int state,std::string name) : FSMState(state,name) {
    auto cfg=param::config["FSM"][name];
    first_frame_test_=cfg["first_frame_test"].as<bool>(false);
    auto path=param::proj_dir/cfg["policy_dir"].as<std::string>();
    core_=std::make_unique<beyondmimic::Core>(beyondmimic::load_config(path/"deploy.yaml"));
    motion_=std::make_unique<beyondmimic::Motion>((path/"reference.bin").string());
    if(!first_frame_test_) runner_=std::make_unique<beyondmimic::Runner>(path/"policy.onnx");
    lowstate->set_timeout_ms(100);
    start_frame_=cfg["start_frame"].as<std::size_t>(0);
    const float duration=cfg["duration_s"].as<float>(10.0f);
    max_start_error_=cfg["max_start_error"].as<float>(0.25f);
    max_tilt_=cfg["max_tilt"].as<float>(0.85f);
    max_target_speed_=cfg["max_target_speed"].as<float>(8.0f);
    max_anchor_error_=cfg["max_anchor_error"].as<float>(0.35f);
    const float prepare_duration=cfg["prepare_duration_s"].as<float>(0);
    if(!std::isfinite(prepare_duration) || prepare_duration<0)
        throw std::runtime_error("Invalid preparation duration");
    preparation_enabled_=prepare_duration>0;
    if(first_frame_test_ && !preparation_enabled_)
        throw std::runtime_error("First-frame test requires preparation");
    if(start_frame_>=motion_->frames() || !std::isfinite(duration) || duration<=0 || duration>300 ||
       !(max_start_error_>0 && max_start_error_<0.8f) || !(max_tilt_>0 && max_tilt_<1.5f) ||
       !(max_target_speed_>0 && max_target_speed_<=20) || !(max_anchor_error_>0 && max_anchor_error_<=1))
        throw std::runtime_error("Invalid BeyondMimic segment/limits");
    end_frame_=std::min(motion_->frames(),start_frame_+static_cast<std::size_t>(duration/core_->config().dt));
    if(end_frame_<=start_frame_) throw std::runtime_error("Empty BeyondMimic segment");
    if(preparation_enabled_) {
        const auto stand=param::config["FSM"]["FixStand"];
        const auto poses=stand["qs"];
        if(!poses.IsSequence() || poses.size()==0) throw std::runtime_error("Missing FixStand preparation posture");
        standing_q_=beyondmimic::read_array<beyondmimic::J>(poses[poses.size()-1]);
        preparation_cfg_.standing_kp=beyondmimic::read_array<beyondmimic::J>(stand["kp"]);
        preparation_cfg_.standing_kd=beyondmimic::read_array<beyondmimic::J>(stand["kd"]);
        preparation_cfg_.pose_duration=prepare_duration;
        preparation_cfg_.gain_duration=cfg["prepare_gain_duration_s"].as<float>(0.5f);
        preparation_cfg_.settle_timeout=cfg["prepare_settle_timeout_s"].as<float>(2);
        preparation_cfg_.settle_duration=cfg["prepare_settle_duration_s"].as<float>(0.2f);
        preparation_cfg_.max_target_speed=cfg["prepare_max_target_speed"].as<float>(0.5f);
        preparation_cfg_.max_joint_speed=cfg["prepare_max_joint_speed"].as<float>(0.5f);
        preparation_cfg_.max_error=max_start_error_;
        preparation_cfg_.dt=core_->config().dt;
        const auto& control=core_->config();
        beyondmimic::Joints goal{};
        const auto first=motion_->frame(start_frame_);
        for(int i=0;i<beyondmimic::J;++i) {
            const int sdk=control.sdk_indices[i];
            goal[sdk]=first.q[i];
            preparation_cfg_.policy_kp[sdk]=control.kp[i];
            preparation_cfg_.policy_kd[sdk]=control.kd[i];
            preparation_cfg_.lower[sdk]=control.lower[i];
            preparation_cfg_.upper[sdk]=control.upper[i];
            preparation_cfg_.effort[sdk]=control.effort[i];
        }
        if(first_frame_test_) {
            // Diagnostic gains are explicitly in SDK order. Never modify the
            // model package or silently use these gains for policy inference.
            preparation_cfg_.policy_kp=beyondmimic::read_array<beyondmimic::J>(cfg["first_frame_kp"]);
            preparation_cfg_.policy_kd=beyondmimic::read_array<beyondmimic::J>(cfg["first_frame_kd"]);
        }
        // Validate parameters and reference posture before connecting a worker.
        beyondmimic::Preparation<beyondmimic::J> check(preparation_cfg_,standing_q_,goal);
    }
    registered_checks.insert(registered_checks.begin(),{[this]{return stopped_.load();},FSMStringMap.right.at("Passive")});
    if(first_frame_test_) spdlog::info("First-frame test reference loaded: {} frames; no policy runner created",motion_->frames());
    else spdlog::info("BeyondMimic loaded: 160 obs / 29 actions, {} frames, leg odometry",motion_->frames());
    if(first_frame_test_) {
        spdlog::warn("FIRST FRAME TEST ONLY: policy inference DISABLED; holding reference frame {} until L2+B or protection exit",
                     start_frame_);
        spdlog::info("First-frame PD: hip_pitch L/R={}/{} knee L/R={}/{} ankle_pitch L/R={}/{}",
                     preparation_cfg_.policy_kp[0],preparation_cfg_.policy_kp[6],
                     preparation_cfg_.policy_kp[3],preparation_cfg_.policy_kp[9],
                     preparation_cfg_.policy_kp[4],preparation_cfg_.policy_kp[10]);
    } else {
        spdlog::info("BeyondMimic segment: start_frame={} end_frame={} (exclusive), {} s; segment end enters damping",
                     start_frame_, end_frame_, (end_frame_-start_frame_)*core_->config().dt);
    }
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
            const float expected=preparation_enabled_ ? standing_q_[sdk] : ref.q[i];
            error=std::max(error,std::abs(current.q[sdk]-expected));
        }
        if(error>max_start_error_) throw std::runtime_error("Initial posture differs from required start by "+std::to_string(error)+" rad; return to FixStand");
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            targets_=current.q;
            for(int i=0;i<beyondmimic::J;++i) {
                const int sdk=cfg.sdk_indices[i];
                target_kp_[sdk]=preparation_enabled_ ? preparation_cfg_.standing_kp[sdk] : cfg.kp[i];
                target_kd_[sdk]=preparation_enabled_ ? preparation_cfg_.standing_kd[sdk] : cfg.kd[i];
            }
            last_result_=std::chrono::steady_clock::now();
        }
        for(int i=0;i<beyondmimic::J;++i) {
            auto& motor=lowcmd->msg_.motor_cmd()[cfg.sdk_indices[i]];
            motor.q()=current.q[cfg.sdk_indices[i]];
            motor.kp()=target_kp_[cfg.sdk_indices[i]]; motor.kd()=target_kd_[cfg.sdk_indices[i]];
            motor.dq()=0; motor.tau()=0;
        }
        if(preparation_enabled_) spdlog::info("BeyondMimic: starting preparation from FixStand; policy clock has not started");
        running_=true;
        worker_=std::thread(&State_BeyondMimic::policy_loop,this);
    } catch(const std::exception& e) {
        stop(e.what()); damping();
    }
}

bool State_BeyondMimic::prepare() {
    using Clock=std::chrono::steady_clock;
    const auto initial=snapshot();
    beyondmimic::Joints goal{};
    const auto first=motion_->frame(start_frame_);
    for(int i=0;i<beyondmimic::J;++i) goal[core_->config().sdk_indices[i]]=first.q[i];
    beyondmimic::Preparation<beyondmimic::J> planner(preparation_cfg_,initial.q,goal);
    spdlog::info("BeyondMimic preparation: pose {} s, gain blend {} s, settle confirmation {} s, settling timeout {} s",
                 planner.pose_duration(),preparation_cfg_.gain_duration,preparation_cfg_.settle_duration,preparation_cfg_.settle_timeout);
    const auto begin=Clock::now();
    auto deadline=begin;
    const auto dt=std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(core_->config().dt));
    float next_log=0;
    bool holding_first_frame=false;
    while(running_) {
        if(lowstate->isTimeout()) throw std::runtime_error("LowState timeout during preparation");
        const auto current=snapshot();
        const float tilt=std::acos(std::clamp((current.orientation*beyondmimic::Vec::UnitZ()).z(),-1.0f,1.0f));
        if(tilt>max_tilt_) throw std::runtime_error("Pelvis tilt limit exceeded during preparation");
        const float elapsed=std::chrono::duration<float>(Clock::now()-begin).count();
        const auto cmd=planner.update(elapsed,current.q,current.dq);
        if(!running_) return false;
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            targets_=cmd.q; target_kp_=cmd.kp; target_kd_=cmd.kd;
            last_result_=Clock::now();
        }
        if(elapsed>=next_log) {
            float error=0, speed=0;
            for(int i=0;i<beyondmimic::J;++i) {
                error=std::max(error,std::abs(current.q[i]-goal[i]));
                speed=std::max(speed,std::abs(current.dq[i]));
            }
            spdlog::info("BeyondMimic {}: elapsed={:.2f} s max_error={:.3f} rad max_speed={:.3f} rad/s tilt={:.3f} rad",
                         holding_first_frame ? "FIRST FRAME HOLD (policy disabled)" : "preparing",
                         elapsed,error,speed,tilt);
            if(first_frame_test_) {
                auto torque=[&](int i) { return cmd.kp[i]*(cmd.q[i]-current.q[i])-cmd.kd[i]*current.dq[i]; };
                spdlog::info("First-frame legs q/target/tau_PD: knee L={:.3f}/{:.3f}/{:.2f} R={:.3f}/{:.3f}/{:.2f}; ankle_pitch L={:.3f}/{:.3f}/{:.2f} R={:.3f}/{:.3f}/{:.2f}",
                             current.q[3],cmd.q[3],torque(3),current.q[9],cmd.q[9],torque(9),
                             current.q[4],cmd.q[4],torque(4),current.q[10],cmd.q[10],torque(10));
            }
            next_log=elapsed+1;
        }
        if(cmd.ready) {
            if(!first_frame_test_) return true;
            if(!holding_first_frame) {
                holding_first_frame=true;
                spdlog::warn("FIRST FRAME HOLD reached: high diagnostic PD active, policy inference DISABLED; L2+B exits to damping");
            }
        }
        if(holding_first_frame && !cmd.ready)
            throw std::runtime_error("First-frame hold lost posture/velocity readiness");
        deadline+=dt;
        if(Clock::now()>deadline+dt) throw std::runtime_error("Preparation missed its 20 ms deadline");
        std::this_thread::sleep_until(deadline);
    }
    return false;
}

void State_BeyondMimic::stop(const std::string& reason) {
    running_=false;
    if(!stopped_.exchange(true)) spdlog::warn("BeyondMimic -> Passive: {}",reason);
}

void State_BeyondMimic::policy_loop() {
    try {
        using Clock=std::chrono::steady_clock;
        if(preparation_enabled_ && !prepare()) return;
        if(!running_) return;
        // Defense in depth: a diagnostic configuration must never reach act().
        if(first_frame_test_) return;
        const auto initial=snapshot();
        const auto first=motion_->frame(start_frame_);
        float error=0;
        for(int i=0;i<beyondmimic::J;++i)
            error=std::max(error,std::abs(initial.q[core_->config().sdk_indices[i]]-first.q[i]));
        if(error>max_start_error_) throw std::runtime_error("Motion first-frame posture check failed after preparation");
        core_->reset(initial,first);
        spdlog::info("BeyondMimic: restored training PD from deploy.yaml for {} joints; starting reference frame {}",
                     beyondmimic::J, start_frame_);
        const auto dt=std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(core_->config().dt));
        auto deadline=Clock::now();
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
    for(int i=0;i<beyondmimic::J;++i) {
        auto& motor=lowcmd->msg_.motor_cmd()[i];
        motor.q()=targets_[i]; motor.kp()=target_kp_[i]; motor.kd()=target_kd_[i];
        motor.dq()=0; motor.tau()=0;
    }
}

void State_BeyondMimic::exit() {
    running_=false;
    if(worker_.joinable()) worker_.join();
}
