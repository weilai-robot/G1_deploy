#pragma once
#include "FSM/FSMState.h"
#include "beyondmimic/package.h"
#include "beyondmimic/cyclic.h"
#include <atomic>
#include <fstream>
#include <mutex>
#include <thread>

// Independent of the finite dance state. Commands never reset policy or odometry.
class State_BeyondMimicWalk : public FSMState {
public:
    State_BeyondMimicWalk(int state,std::string name);
    ~State_BeyondMimicWalk() override { exit(); }
    void enter() override;
    void run() override;
    void exit() override;
private:
    beyondmimic::Snapshot snapshot();
    void cache_sticks();
    void policy_loop();
    void stop(const std::string& reason);
    void damping();
    void open_log();
    std::unique_ptr<beyondmimic::Core> core_;
    std::unique_ptr<beyondmimic::CyclicGait> gait_;
    std::unique_ptr<beyondmimic::Runner> runner_;
    std::thread worker_;
    std::atomic<bool> running_{false},stopped_{true};
    std::mutex target_mutex_,command_mutex_;
    beyondmimic::Joints targets_{};
    std::array<float,3> sticks_{};
    std::chrono::steady_clock::time_point last_result_{},last_command_{};
    std::size_t initial_support_=0;
    float deadzone_=0.1f,minimum_speed_scale_=0.5f;
    float max_start_error_=0.25f,max_tilt_=0.85f,max_target_speed_=8.0f,max_anchor_error_=0.35f;
    std::string model_id_;
    std::filesystem::path log_directory_;
    std::ofstream log_;
};
REGISTER_FSM(State_BeyondMimicWalk)
