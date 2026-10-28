#pragma once
#include "FSM/FSMState.h"
#include "velocity11k/runner.h"
#include <atomic>
#include <fstream>
#include <mutex>
#include <thread>

class State_Velocity11k : public FSMState {
public:
    State_Velocity11k(int state,std::string name);
    ~State_Velocity11k() override { exit(); }
    void enter() override;
    void run() override;
    void exit() override;
private:
    velocity11k::Snapshot snapshot();
    void policy_loop();
    void stop(const std::string& reason);
    void damping();
    void open_joint_log();
    std::unique_ptr<velocity11k::Runner> runner_;
    velocity11k::Core core_;
    std::thread worker_;
    std::mutex target_mutex_;
    velocity11k::Joints targets_{};
    std::chrono::steady_clock::time_point last_result_;
    std::atomic<bool> running_{false},stopped_{true};
    float max_start_error_=0.25f, max_tilt_=1.0f;
    bool joint_log_enabled_=true;
    std::ofstream joint_log_;
};
REGISTER_FSM(State_Velocity11k)
