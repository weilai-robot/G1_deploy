#pragma once

// Standalone HugWBC FSM state (R41 reference-arms / R42 adaptation packages).
// Semantics follow the frozen HugWBC training contract, NOT State_RLBase or
// State_BeyondMimic: 23-dim hybrid actions, 5x88 history, 50 Hz policy ticks
// inside the 1 kHz FSM send loop, eight reference arm joints and six held
// wrists. The variant (R41/R42) is selected purely by the config file loaded
// with --config; nothing here auto-picks "newest" models.

#include "FSM/FSMState.h"
#include "hugwbc/package.h"
#include "hugwbc/runner.h"
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>

class State_HugWBC : public FSMState {
public:
    State_HugWBC(int state, std::string name);
    ~State_HugWBC() override { exit(); }
    void enter() override;
    void run() override;
    void exit() override;
private:
    using Clock = std::chrono::steady_clock;
    hugwbc::Snapshot snapshot();
    void policy_loop();
    void stop(const std::string& reason);
    void damping();
    void open_tick_log();
    void log_tick(const hugwbc::Tick& tick, Clock::time_point sensor_time,
                  Clock::time_point inference_done, Clock::time_point target_ready_time);

    std::unique_ptr<hugwbc::Core> core_;
    std::unique_ptr<hugwbc::Runner> runner_;
    std::string checkpoint_sha_;
    std::string policy_sha_;
    std::thread worker_;
    std::atomic<bool> running_{false}, stopped_{true};
    std::mutex target_mutex_;
    hugwbc::Motors targets_{};   // full 29-DOF position package, SDK order
    std::chrono::steady_clock::time_point last_result_{};
    std::array<float, hugwbc::CMD> command_{};  // fixed test profile from config
    bool standing_ = false;
    float max_tilt_ = 0.85f;
    float max_start_error_ = 0.6f;  // un-validated placeholder, see runbook
    // Per-tick record (plan section 11), written from the policy thread only.
    bool tick_log_enabled_ = true;
    std::ofstream tick_log_;
    hugwbc::Snapshot last_snapshot_{};
    hugwbc::Vec last_pelvis_gyro_ = hugwbc::Vec::Zero();
    hugwbc::Vec last_pelvis_gravity_ = hugwbc::Vec::Zero();
};
REGISTER_FSM(State_HugWBC)
