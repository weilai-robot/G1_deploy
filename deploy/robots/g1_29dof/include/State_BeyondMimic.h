#pragma once

#include "FSM/FSMState.h"
#include "beyondmimic/package.h"
#include "beyondmimic/preparation.h"
#include <atomic>
#include <chrono>
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
    bool prepare();
    void stop(const std::string& reason);
    void damping();
    std::unique_ptr<beyondmimic::Core> core_;
    std::unique_ptr<beyondmimic::Motion> motion_;
    std::unique_ptr<beyondmimic::Runner> runner_;
    std::thread worker_;
    std::atomic<bool> running_{false},stopped_{true};
    std::mutex target_mutex_;
    beyondmimic::Joints targets_{}, target_kp_{}, target_kd_{};
    beyondmimic::Joints standing_q_{};
    beyondmimic::PreparationConfig<beyondmimic::J> preparation_cfg_;
    std::chrono::steady_clock::time_point last_result_{};
    std::size_t start_frame_=0,end_frame_=0;
    float max_start_error_=0.25f,max_tilt_=0.85f,max_target_speed_=8.0f;
    float max_anchor_error_=0.35f;
    bool preparation_enabled_=false;
    bool first_frame_test_=false;
};
REGISTER_FSM(State_BeyondMimic)

// A distinct factory type makes older binaries reject the diagnostic config
// instead of ignoring first_frame_test and accidentally playing a policy.
class State_BeyondMimicFirstFrameTest : public State_BeyondMimic {
    static std::string require_test(const std::string& name) {
        if(!param::config["FSM"][name]["first_frame_test"].as<bool>(false))
            throw std::runtime_error("First-frame test state requires first_frame_test: true");
        return name;
    }
public:
    State_BeyondMimicFirstFrameTest(int state, std::string name)
        : State_BeyondMimic(state, require_test(name)) {}
};
REGISTER_FSM(State_BeyondMimicFirstFrameTest)
