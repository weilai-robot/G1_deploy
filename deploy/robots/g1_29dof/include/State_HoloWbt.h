#pragma once

#include "FSM/FSMState.h"
#include "holo_wbt/package.h"
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

// holosoma 154 维 WBT 策略状态：50 Hz 参考动作回放 + 无里程计的观测拼装。
class State_HoloWbt : public FSMState {
public:
    State_HoloWbt(int state, std::string name);
    ~State_HoloWbt() override { exit(); }
    void enter() override;
    void run() override;
    void exit() override;
private:
    holowbt::Snapshot snapshot();
    void policy_loop();
    void stop(const std::string& reason);
    void damping();
    std::unique_ptr<holowbt::Core> core_;
    std::unique_ptr<holowbt::Motion> motion_;
    std::unique_ptr<holowbt::Runner> runner_;
    std::thread worker_;
    std::atomic<bool> running_{false}, stopped_{true}, completed_{false};
    std::mutex target_mutex_;
    holowbt::Joints targets_{};
    std::chrono::steady_clock::time_point last_result_{};
    std::size_t start_frame_ = 0, end_frame_ = 0;
    float max_start_error_ = 0.25f, max_tilt_ = 0.85f, max_target_speed_ = 8.0f;
};
REGISTER_FSM(State_HoloWbt)
