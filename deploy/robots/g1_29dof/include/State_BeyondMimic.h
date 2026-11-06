#pragma once

#include "FSM/FSMState.h"
#include "beyondmimic/package.h"
#include <atomic>
#include <mutex>
#include <thread>

class State_BeyondMimic : public FSMState {
public:
    State_BeyondMimic(int state,std::string name);
    ~State_BeyondMimic() override { exit(); }
    void enter() override;
    void run() override;
    void exit() override;
private:
    beyondmimic::Snapshot snapshot();
    void policy_loop();
    void stop(const std::string& reason);
    void damping();
    void apply_gains(bool scaled);
    std::unique_ptr<beyondmimic::Core> core_;
    std::unique_ptr<beyondmimic::Motion> motion_;
    std::unique_ptr<beyondmimic::Runner> runner_;
    std::thread worker_;
    std::atomic<bool> running_{false},stopped_{true};
    std::mutex target_mutex_;
    beyondmimic::Joints targets_{};
    beyondmimic::Joints prepose_from_{},prepose_goal_{};  // SDK order
    std::chrono::steady_clock::time_point last_result_{};
    std::size_t start_frame_=0,end_frame_=0;
    int prepose_tick_=0,prepose_ticks_=1;
    float max_start_error_=0.25f,max_tilt_=0.85f,max_target_speed_=8.0f;
    float max_anchor_error_=0.35f;
    float prepose_s_=0.0f,prepose_kp_scale_=1.0f;
    bool prepose_only_=false;  // true: ramp to the reference start pose and hold for [RB + X]
};
REGISTER_FSM(State_BeyondMimic)
