// Deterministic checks of the real C++ deployment core. No DDS or robot I/O.
#include "hugwbc/package.h"
#include <iostream>

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

hugwbc::Snapshot nominal(const hugwbc::Config& cfg) {
    hugwbc::Snapshot s;
    for (int i = 0; i < hugwbc::J; ++i) s.q[cfg.model_to_sdk[i]] = cfg.default_q[i];
    return s;
}

int main(int argc, char** argv) {
    try {
        using namespace hugwbc;
        if (argc != 2) throw std::runtime_error("Usage: hugwbc_core_checks PACKAGE_DIRECTORY");
        auto cfg = load_package(argv[1]).cfg;
        const std::array<float, CMD> cmd{0.5f, 0, 0, 1.5f, 0.5f, 0.5f, 0.1f, 0, 0, 0, 0};
        Joints raw{};
        Core core(cfg);
        core.reset();
        auto s = nominal(cfg);
        Joints previous{};
        for (int t = 1; t <= 8; ++t) {
            s.angular_velocity.x() = 4.f * t;
            auto frame = core.prepare(s, cmd, false);
            for (int j = 0; j < J; ++j)
                require(std::abs(frame.obs[52 + j] - previous[j]) < 1e-6f,
                        "previous hybrid action mismatch");
            auto tick = core.complete(s, cmd, false, raw);
            for (int h = 0; h < HIST; ++h)
                require(std::abs(tick.history[h * OBS] - std::max(1, t - 4 + h)) < 1e-6f,
                        "history does not drop oldest frame");
            previous = tick.hybrid_action;
        }
        auto standing = core.prepare(s, cmd, true);
        require(standing.clock_left == 0 && standing.clock_right == 0, "standing clock mismatch");
        core.reset();
        require(core.tick_index() == 0 && !core.seeded(), "reset did not clear state");
        auto reset_frame = core.prepare(s, cmd, false);
        for (int j = 0; j < J; ++j)
            require(reset_frame.obs[52 + j] == 0, "reset retained previous actions");

        cfg.imu_source = ImuSource::Torso;
        Core torso(cfg);
        s = nominal(cfg);
        s.q[12] = 0.3f; s.q[13] = -0.2f; s.q[14] = 0.4f;
        s.dq[12] = 0.7f; s.dq[13] = -0.8f; s.dq[14] = 0.9f;
        const Quat yaw(Eigen::AngleAxisf(s.q[12], Vec::UnitZ()));
        const Quat roll(Eigen::AngleAxisf(s.q[13], Vec::UnitX()));
        const Quat pitch(Eigen::AngleAxisf(s.q[14], Vec::UnitY()));
        const Quat waist = yaw * roll * pitch;
        const Quat pelvis = Quat(0.97f, 0.1f, 0.15f, -0.05f).normalized();
        const Vec expected_gyro(0.2f, -0.3f, 0.4f);
        const Vec relative = Vec::UnitZ() * s.dq[12] + yaw * (Vec::UnitX() * s.dq[13]) +
                             (yaw * roll) * (Vec::UnitY() * s.dq[14]);
        s.orientation = pelvis * waist;
        s.angular_velocity = waist.conjugate() * (expected_gyro + relative);
        Quat measured;
        Vec gyro, gravity;
        torso.pelvis_frame(s, measured, gyro, gravity);
        require((gyro - expected_gyro).norm() < 1e-6f, "three-axis torso gyro mismatch");
        require((gravity - pelvis.conjugate() * Vec(0, 0, -1)).norm() < 1e-6f,
                "torso gravity mismatch");

        cfg.imu_source = ImuSource::Pelvis;
        for (int i = 0; i < WRISTS; ++i) cfg.wrist_target[cfg.wrist_sdk_indices[i]] = 0.1f * (i + 1);
        Core wrists(cfg);
        s = nominal(cfg);
        wrists.prepare(s, cmd, false);
        auto tick = wrists.complete(s, cmd, false, raw);
        for (int w : cfg.wrist_sdk_indices)
            require(tick.sdk_targets[w] == cfg.wrist_target[w], "wrist target mismatch");

        Core guard(cfg);
        s.q[18] = cfg.upper[18] - 0.01f;
        s.dq[18] = 30.f;
        guard.prepare(s, cmd, false);
        bool rejected = false;
        try { guard.complete(s, cmd, false, raw); }
        catch (const std::runtime_error& e) {
            rejected = std::string(e.what()).find("Empty target constraint intersection") != std::string::npos;
        }
        require(rejected, "infeasible tick returned a target package");
        guard.reset();
        s = nominal(cfg);
        guard.prepare(s, cmd, false);
        guard.complete(s, cmd, false, raw);

        cfg.max_target_speed = 1.f;
        Core speed(cfg);
        speed.prepare(s, cmd, false);
        raw[0] = 2.f;
        auto first = speed.complete(s, cmd, false, raw);
        require(std::abs(first.sdk_targets[0] - 0.4f) < 1e-6f, "first target incorrectly rate limited");
        speed.prepare(s, cmd, false);
        raw[0] = -2.f;
        auto second = speed.complete(s, cmd, false, raw);
        require(std::abs(second.sdk_targets[0] - (first.sdk_targets[0] - cfg.dt)) < 1e-6f,
                "subsequent target not rate limited");
        std::cout << "PASS: history, hybrid actions, reset, standing, three-axis IMU, wrists, "
                     "infeasible tick rejection, speed-limit initialization\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
