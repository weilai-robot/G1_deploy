#include "State_BeyondMimicWalk.h"
#include <cctype>
#include <iomanip>

State_BeyondMimicWalk::State_BeyondMimicWalk(int state,std::string name):FSMState(state,name) {
    auto cfg=param::config["FSM"][name];
    auto path=param::proj_dir/cfg["policy_dir"].as<std::string>();
    auto package=YAML::LoadFile((path/"deploy.yaml").string());
    if(package["controller_mode"].as<std::string>()!="velocity_cyclic_v1" ||
       package["reference"].as<std::string>()!="gait.bin")
        throw std::runtime_error("Walking requires a native velocity-cyclic package");
    model_id_=package["model_id"].as<std::string>();
    if(model_id_.empty() || model_id_.size()>100 ||
       !std::all_of(model_id_.begin(),model_id_.end(),[](unsigned char c){return std::isalnum(c)||c=='_';}))
        throw std::runtime_error("Invalid walking model label");
    core_=std::make_unique<beyondmimic::Core>(beyondmimic::load_config(path/"deploy.yaml"));
    gait_=std::make_unique<beyondmimic::CyclicGait>((path/"gait.bin").string());
    runner_=std::make_unique<beyondmimic::Runner>(path/"policy.onnx");
    initial_support_=cfg["initial_support_index"].as<std::size_t>(0);
    deadzone_=cfg["command_deadzone"].as<float>(0.1f);
    minimum_speed_scale_=cfg["minimum_speed_scale"].as<float>();
    const auto trained_min=package["training_command_curriculum"]["speed_scale_range"][0].as<float>();
    if(minimum_speed_scale_<trained_min || minimum_speed_scale_>1 ||
       !std::isfinite(minimum_speed_scale_)) throw std::runtime_error("Speed scale outside training curriculum");
    gait_->reset(initial_support_);
    beyondmimic::walking_sticks(0,0,0,*gait_,deadzone_,minimum_speed_scale_);
    max_start_error_=cfg["max_start_error"].as<float>(0.25f);
    max_tilt_=cfg["max_tilt"].as<float>(0.85f);
    max_target_speed_=cfg["max_target_speed"].as<float>(8.0f);
    max_anchor_error_=cfg["max_anchor_error"].as<float>(0.35f);
    if(!(max_start_error_>0 && max_start_error_<0.8f) ||
       !(max_tilt_>0 && max_tilt_<1.5f) || !(max_target_speed_>0 && max_target_speed_<=20) ||
       !(max_anchor_error_>0 && max_anchor_error_<=1) || std::abs(core_->config().dt-gait_->dt())>1e-6f)
        throw std::runtime_error("Invalid walking guards/timestep");
    log_directory_=param::proj_dir/cfg["log_directory"].as<std::string>("log/beyondmimic_walk");
    lowstate->set_timeout_ms(100);
    registered_checks.insert(registered_checks.begin(),{[this]{return stopped_.load();},FSMStringMap.right.at("Passive")});
    spdlog::info("BeyondMimicWalk {}: cyclic reference; forward/yaw only; centered sticks hold support",model_id_);
}

beyondmimic::Snapshot State_BeyondMimicWalk::snapshot() {
    std::lock_guard<std::mutex> lock(lowstate->mutex_);
    beyondmimic::Snapshot s;
    const auto& msg=lowstate->msg_;
    for(int i=0;i<beyondmimic::J;++i) {s.q[i]=msg.motor_state()[i].q();s.dq[i]=msg.motor_state()[i].dq();}
    const auto& q=msg.imu_state().quaternion();const auto& w=msg.imu_state().gyroscope();
    s.orientation=beyondmimic::Quat(q[0],q[1],q[2],q[3]);
    s.angular_velocity=beyondmimic::Vec(w[0],w[1],w[2]);
    beyondmimic::validate_snapshot(s);s.orientation.normalize();return s;
}

void State_BeyondMimicWalk::cache_sticks() {
    // Only the FSM thread accesses joystick.update()/axes.
    std::lock_guard<std::mutex> lock(command_mutex_);
    sticks_={lowstate->joystick.ly(),lowstate->joystick.lx(),lowstate->joystick.rx()};
    last_command_=std::chrono::steady_clock::now();
}

void State_BeyondMimicWalk::open_log() {
    std::filesystem::create_directories(log_directory_);
    const auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto path=log_directory_/(model_id_+"_"+std::to_string(stamp)+".csv");
    log_.open(path);
    if(!log_) throw std::runtime_error("Cannot open walking evidence log");
    log_<<std::setprecision(9);
    log_<<"time_s,wall_elapsed_s,phase,cycle,phase_rate,ly,lx,rx,command_vx,command_vy,command_wz,effective_vx,effective_vy,effective_wz,reference_x,reference_y,reference_yaw,estimated_x,estimated_y,estimated_z,estimated_vx_b,estimated_vy_b,estimated_vz_b,tilt,anchor_error,limited_joints";
    for(const char* group:{"q_sdk","dq_sdk","action_policy","target_sdk"})
        for(int i=0;i<beyondmimic::J;++i) log_<<","<<group<<"_"<<i;
    log_<<"\n";
    spdlog::info("BeyondMimicWalk model={} CSV={} (position/velocity are leg estimates, not ground truth)",model_id_,path.string());
}

void State_BeyondMimicWalk::enter() {
    exit();stopped_=false;
    try {
        if(lowstate->isTimeout() || lowstate->isJoystickTimeout())
            throw std::runtime_error("LowState/remote timeout at walking entry");
        cache_sticks();
        if(std::any_of(sticks_.begin(),sticks_.end(),[this](float x){return !std::isfinite(x)||std::abs(x)>deadzone_;}))
            throw std::runtime_error("Center both sticks before entering walking");
        gait_->reset(initial_support_);
        const auto ref=gait_->reference();const auto current=snapshot();const auto& cfg=core_->config();
        if(std::acos(std::clamp((current.orientation*beyondmimic::Vec::UnitZ()).z(),-1.0f,1.0f))>max_tilt_)
            throw std::runtime_error("Pelvis tilt limit at walking entry");
        float error=0;
        for(int i=0;i<beyondmimic::J;++i) error=std::max(error,std::abs(current.q[cfg.sdk_indices[i]]-ref.q[i]));
        if(error>max_start_error_) throw std::runtime_error("Initial posture mismatch; return to the model-specific FixStand");
        core_->reset(current,ref);open_log();
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            targets_=current.q;last_result_=std::chrono::steady_clock::now();
        }
        for(int i=0;i<beyondmimic::J;++i) {
            auto& motor=lowcmd->msg_.motor_cmd()[cfg.sdk_indices[i]];
            motor.q()=current.q[cfg.sdk_indices[i]];motor.kp()=cfg.kp[i];motor.kd()=cfg.kd[i];
            motor.dq()=0;motor.tau()=0;
        }
        running_=true;worker_=std::thread(&State_BeyondMimicWalk::policy_loop,this);
    } catch(const std::exception& e) {stop(e.what());damping();}
}

void State_BeyondMimicWalk::policy_loop() {
    try {
        using Clock=std::chrono::steady_clock;
        const auto dt=std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(core_->config().dt));
        const auto started=Clock::now();auto deadline=started;std::uint64_t step=0;
        while(running_) {
            if(lowstate->isTimeout()) throw std::runtime_error("LowState timeout");
            std::array<float,3> sticks;
            {
                std::lock_guard<std::mutex> lock(command_mutex_);
                if(Clock::now()-last_command_>std::chrono::milliseconds(100))
                    throw std::runtime_error("Joystick/FSM update timeout");
                sticks=sticks_;
            }
            gait_->request(beyondmimic::walking_sticks(sticks[0],sticks[1],sticks[2],*gait_,deadzone_,minimum_speed_scale_));
            const auto current=snapshot();
            const float tilt=std::acos(std::clamp((current.orientation*beyondmimic::Vec::UnitZ()).z(),-1.0f,1.0f));
            if(tilt>max_tilt_) throw std::runtime_error("Pelvis tilt limit exceeded");
            const auto ref=gait_->reference();const auto obs=core_->observe(current,ref);
            const float anchor=beyondmimic::Vec(obs[58],obs[59],obs[60]).norm();
            if(anchor>max_anchor_error_) throw std::runtime_error("Estimated torso/reference position error limit exceeded");
            const auto action=runner_->act(obs);const auto target=core_->targets(action,current,max_target_speed_);
            if(!running_) break;
            {
                std::lock_guard<std::mutex> lock(target_mutex_);
                targets_=target;last_result_=Clock::now();
            }
            log_<<step*core_->config().dt<<","<<std::chrono::duration<double>(Clock::now()-started).count()
                <<","<<gait_->phase()<<","<<gait_->cycles()<<","<<gait_->rate();
            for(float v:sticks) log_<<","<<v;
            for(float v:gait_->bounded()) log_<<","<<v;
            for(float v:gait_->effective()) log_<<","<<v;
            log_<<","<<gait_->position().x()<<","<<gait_->position().y()<<","<<gait_->yaw_angle();
            for(int k=0;k<3;++k) log_<<","<<core_->position()[k];
            for(int k=0;k<3;++k) log_<<","<<core_->velocity_b()[k];
            log_<<","<<tilt<<","<<anchor<<","<<core_->clipped();
            for(const auto& values:{current.q,current.dq,action,target}) for(float v:values) log_<<","<<v;
            log_<<"\n";
            if(!log_) throw std::runtime_error("Walking evidence log write failed");
            gait_->step();++step;deadline+=dt;
            if(Clock::now()>deadline+dt) throw std::runtime_error("Walking policy missed its 20 ms deadline");
            std::this_thread::sleep_until(deadline);
        }
    } catch(const std::exception& e) {stop(e.what());}
}

void State_BeyondMimicWalk::stop(const std::string& reason) {
    running_=false;
    if(!stopped_.exchange(true)) spdlog::warn("BeyondMimicWalk {} -> Passive: {}",model_id_,reason);
}
void State_BeyondMimicWalk::damping() {
    for(int i=0;i<beyondmimic::J;++i) {
        auto& motor=lowcmd->msg_.motor_cmd()[i];
        motor.kp()=0;motor.kd()=3;motor.dq()=0;motor.tau()=0;
    }
}
void State_BeyondMimicWalk::run() {
    if(lowstate->isJoystickTimeout()) {stop("Wireless remote timeout");damping();return;}
    cache_sticks();
    if(stopped_) {damping();return;}
    std::lock_guard<std::mutex> lock(target_mutex_);
    if(std::chrono::steady_clock::now()-last_result_>std::chrono::milliseconds(100)) {
        stop("Walking output timeout");damping();return;
    }
    for(int i=0;i<beyondmimic::J;++i) lowcmd->msg_.motor_cmd()[i].q()=targets_[i];
}
void State_BeyondMimicWalk::exit() {
    running_=false;if(worker_.joinable()) worker_.join();
    if(log_.is_open()) {log_.flush();log_.close();}
}
