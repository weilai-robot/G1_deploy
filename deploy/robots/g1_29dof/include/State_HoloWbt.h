#pragma once

#include "FSM/FSMState.h"
#include "holo_wbt/package.h"
#include <chrono>

// holosoma 154 维 WBT 策略状态：50 Hz 参考动作回放 + 无里程计的观测拼装。
// 推理在 FSM 的 1 kHz 回调内每隔 20 ms 执行一次，不额外起线程。
class State_HoloWbt : public FSMState {
public:
    State_HoloWbt(int state, std::string name);
    ~State_HoloWbt() override = default;
    void enter() override;
    void run() override;
    void exit() override;
private:
    holowbt::Snapshot snapshot();
    void step_policy();
    void stop(const std::string& reason);
    void damping();
    bool try_enter();
    std::unique_ptr<holowbt::Core> core_;
    std::unique_ptr<holowbt::Motion> motion_;
    std::unique_ptr<holowbt::Runner> runner_;
    bool stopped_ = true, completed_ = false;
    holowbt::Joints targets_{};
    holowbt::Joints last_command_{};
    holowbt::Joints prepose_from_{};  // 预摆位起点（进入姿态，SDK 顺序）
    holowbt::Joints prepose_goal_{};  // 预摆位终点（参考首帧，SDK 顺序）
    std::chrono::steady_clock::time_point last_result_{};
    std::chrono::steady_clock::time_point next_step_{};
    std::size_t start_frame_ = 0, end_frame_ = 0, frame_ = 0;
    int blend_tick_ = 0, blend_limit_ticks_ = 1;
    int prepose_tick_ = 0, prepose_ticks_ = 1;
    bool blending_ = false, preposing_ = false, preposed_ = false;
    float max_start_error_ = 0.25f, max_tilt_ = 0.85f, max_target_speed_ = 8.0f;
    float blend_s_ = 2.5f, blend_rate_ = 1.5f, prepose_s_ = 3.0f;
    bool prepose_only_ = false;  // true: 只做预摆位并保持，等按键再进入动作状态
};
REGISTER_FSM(State_HoloWbt)
