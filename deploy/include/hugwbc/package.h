#pragma once

// HugWBC deployment package: configuration loading, model identity checks and
// ONNX Runtime inference. Strict by design: anything that does not match the
// frozen R41/R42 contract is rejected before any control happens.

#include "hugwbc/core.h"
#include <yaml-cpp/yaml.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hugwbc {

// Canonical frozen contract, verified against the loaded deploy.yaml.
constexpr const char* FORMAT_ID = "hugwbc_g1_23_88_v1";
inline const std::array<const char*, J>& training_joint_names() {
    static const std::array<const char*, J> names = {
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint",
    };
    return names;
}

// Compact public-domain-style SHA-256 (FIPS 180-4), used for on-robot identity
// checks of the loaded policy file. Kept dependency-free on purpose.
class Sha256 {
public:
    Sha256() { reset(); }
    void reset() {
        h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        len_ = 0; buf_len_ = 0;
    }
    void update(const unsigned char* data, std::size_t size) {
        len_ += size;
        while (size > 0) {
            const std::size_t take = std::min(size, std::size_t(64) - buf_len_);
            std::memcpy(buffer_ + buf_len_, data, take);
            buf_len_ += take; data += take; size -= take;
            if (buf_len_ == 64) { block(buffer_); buf_len_ = 0; }
        }
    }
    std::string hex() {
        unsigned char pad = 0x80;
        const std::uint64_t bits = len_ * 8;
        update(&pad, 1);
        const unsigned char zero = 0;
        while (buf_len_ != 56) update(&zero, 1);
        unsigned char length[8];
        for (int i = 0; i < 8; ++i) length[i] = unsigned(bits >> (56 - 8 * i));
        update(length, 8);
        std::string out;
        char cell[9];
        for (int i = 0; i < 8; ++i) {
            std::snprintf(cell, sizeof(cell), "%08x", h_[i]);
            out += cell;
        }
        return out;
    }
private:
    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const unsigned char* p) {
        static const std::uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(p[4*i]) << 24) | (std::uint32_t(p[4*i+1]) << 16) |
                   (std::uint32_t(p[4*i+2]) << 8) | std::uint32_t(p[4*i+3]);
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const std::uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        std::uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + s1 + ch + K[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
        h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }
    std::array<std::uint32_t, 8> h_{};  // std::array: braced assignment in reset()
    unsigned char buffer_[64];
    std::uint64_t len_ = 0;
    std::size_t buf_len_ = 0;
};

inline std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + path.string());
    Sha256 sha;
    char chunk[65536];
    while (file.read(chunk, sizeof(chunk)) || file.gcount() > 0)
        sha.update(reinterpret_cast<unsigned char*>(chunk), std::size_t(file.gcount()));
    return sha.hex();
}

template<std::size_t N>
std::array<float, N> read_array(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != N) throw std::runtime_error("Invalid package vector length");
    std::array<float, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = node[i].as<float>();
        if (!std::isfinite(result[i])) throw std::runtime_error("Non-finite package parameter");
    }
    return result;
}

struct Package {
    Config cfg;
    std::string checkpoint_sha256;  // source .pt identity, also printed at startup
    std::string policy_sha256;      // loaded policy.onnx identity
};

inline Package load_package(const std::filesystem::path& path) {
    const auto yaml = YAML::LoadFile((path / "deploy.yaml").string());
    if (!yaml["format"] || yaml["format"].as<std::string>() != FORMAT_ID)
        throw std::runtime_error("Incompatible HugWBC package format");
    if (yaml["model"].as<std::string>() != "policy.onnx")
        throw std::runtime_error("Unexpected policy file name");
    const int observation_dim = yaml["observation_dim"].as<int>();
    const int history_steps = yaml["history_steps"].as<int>();
    const int action_dim = yaml["action_dim"].as<int>();
    const int proprioception_dim = yaml["proprioception_dim"].as<int>();
    const int command_dim = yaml["command_dim"].as<int>();
    const int clock_dim = yaml["clock_dim"].as<int>();
    if (observation_dim != OBS || history_steps != HIST || action_dim != J ||
        proprioception_dim != PRO || command_dim != CMD || clock_dim != CLK)
        throw std::runtime_error("Incompatible HugWBC observation/action contract");
    if (!yaml["checkpoint_sha256"] || !yaml["policy_sha256"])
        throw std::runtime_error("Package identity hashes missing");

    Package package;
    package.checkpoint_sha256 = yaml["checkpoint_sha256"].as<std::string>();
    package.policy_sha256 = yaml["policy_sha256"].as<std::string>();

    Config& cfg = package.cfg;
    cfg.dt = yaml["step_dt"].as<float>();
    auto names = yaml["joint_names"];
    if (!names.IsSequence() || names.size() != J) throw std::runtime_error("Invalid joint names");
    for (int i = 0; i < J; ++i)
        if (names[i].as<std::string>() != training_joint_names()[i])
            throw std::runtime_error("Joint order does not match the frozen training contract");
    if (yaml["sdk_indices"].size() != J) throw std::runtime_error("Invalid SDK mapping");
    for (int i = 0; i < J; ++i) cfg.model_to_sdk[i] = yaml["sdk_indices"][i].as<int>();
    cfg.default_q = read_array<J>(yaml["default_joint_pos"]);
    cfg.kp = read_array<J>(yaml["kp"]);
    cfg.kd = read_array<J>(yaml["kd"]);
    cfg.action_scale = read_array<J>(yaml["action_scale"]);
    cfg.torque_limit = read_array<J>(yaml["torque_limits"]);
    cfg.lower = read_array<J>(yaml["joint_lower"]);
    cfg.upper = read_array<J>(yaml["joint_upper"]);
    cfg.obs_ang_vel_scale = yaml["obs_scales"]["ang_vel"].as<float>();
    cfg.obs_dof_vel_scale = yaml["obs_scales"]["dof_vel"].as<float>();
    if (std::abs(cfg.obs_ang_vel_scale - 0.25f) > 1e-6f ||
        std::abs(cfg.obs_dof_vel_scale - 0.05f) > 1e-6f)
        throw std::runtime_error("Observation scales do not match the training contract");
    cfg.command_scales = read_array<CMD>(yaml["command_scales"]);
    cfg.clip_obs = yaml["clip_observations"].as<float>();
    cfg.clip_action = yaml["clip_actions"].as<float>();
    if (std::abs(cfg.clip_obs - 100.f) > 1e-6f || std::abs(cfg.clip_action - 100.f) > 1e-6f)
        throw std::runtime_error("Clip values do not match the training contract");

    auto arm = yaml["arm_reference"];
    cfg.arm_model_indices = {15, 16, 17, 18, 19, 20, 21, 22};
    if (arm["model_indices"].size() != ARMS) throw std::runtime_error("Invalid arm reference indices");
    for (int i = 0; i < ARMS; ++i)
        if (arm["model_indices"][i].as<int>() != cfg.arm_model_indices[i])
            throw std::runtime_error("Arm reference indices do not match the training contract");
    cfg.arm_pitch_bias = arm["pitch_bias_rad"].as<float>();
    cfg.arm_shoulder_roll = arm["shoulder_roll_rad"].as<float>();
    cfg.arm_elbow = arm["elbow_rad"].as<float>();
    cfg.arm_amplitude = arm["swing_pitch_amplitude"].as<float>();
    cfg.arm_phase_offset = arm["swing_phase_offset"].as<float>();
    cfg.arm_speed_reference = arm["swing_speed_reference"].as<float>();
    cfg.arm_velocity_feedforward = arm["velocity_feedforward"].as<bool>();
    if (cfg.arm_velocity_feedforward)
        throw std::runtime_error("The accepted execution used reference_velocity_feedforward=false");

    auto wrist = yaml["wrist_hold"];
    if (wrist["sdk_indices"].size() != WRISTS) throw std::runtime_error("Invalid wrist hold mapping");
    for (int i = 0; i < WRISTS; ++i) cfg.wrist_sdk_indices[i] = wrist["sdk_indices"][i].as<int>();
    const auto wrist_target = read_array<WRISTS>(wrist["target_rad"]);
    const float wrist_kp = wrist["kp"].as<float>(), wrist_kd = wrist["kd"].as<float>();
    for (int i = 0; i < WRISTS; ++i) {
        const int s = cfg.wrist_sdk_indices[i];
        cfg.wrist_target[s] = wrist_target[i];
        cfg.wrist_kp[s] = wrist_kp;
        cfg.wrist_kd[s] = wrist_kd;
    }

    const std::string imu_source = yaml["imu_source"].as<std::string>();
    if (imu_source == "pelvis") cfg.imu_source = ImuSource::Pelvis;
    else if (imu_source == "torso") cfg.imu_source = ImuSource::Torso;
    else throw std::runtime_error("Unknown imu_source: " + imu_source);
    cfg.torque_guard = yaml["torque_guard"].as<bool>();
    cfg.max_target_speed = yaml["max_target_speed"].as<float>();
    cfg.validate();
    return package;
}

}  // namespace hugwbc
