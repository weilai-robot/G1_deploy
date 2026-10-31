#pragma once

// 154 维 whole-body-tracking 运行时：holosoma 训练契约（无里程计、无参考位置）。
// 参考动作与推理逻辑均不依赖 DDS，真机控制器与离线校验共用同一份代码。
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

namespace holowbt {
constexpr int J = 29, B = 14, O = 154, RecordSize = 2 * J + 7 * B;
inline constexpr const char* Format = "holo_wbt_g1_154_v1";
using Vec = Eigen::Vector3f;
using Quat = Eigen::Quaternionf;
using Joints = std::array<float, J>;

// 一帧机器人状态：关节角/角速度按 SDK（硬件）顺序，IMU 为骨盆坐标系。
struct Snapshot {
    Joints q{}, dq{};
    Quat orientation = Quat::Identity();
    Vec angular_velocity = Vec::Zero();
};

// 运动链一段：motor_index < 0 表示固定关节。
struct Segment {
    int motor_index = -1;
    Vec xyz = Vec::Zero(), rpy = Vec::Zero(), axis = Vec::UnitZ();
};
using Chain = std::vector<Segment>;

inline Eigen::Isometry3f forward_kinematics(const Chain& chain, const Joints& q) {
    auto pose = Eigen::Isometry3f::Identity();
    for (const auto& joint : chain) {
        pose.translate(joint.xyz);
        pose.rotate(Eigen::AngleAxisf(joint.rpy.z(), Vec::UnitZ()) *
                    Eigen::AngleAxisf(joint.rpy.y(), Vec::UnitY()) *
                    Eigen::AngleAxisf(joint.rpy.x(), Vec::UnitX()));
        if (joint.motor_index >= 0) {
            pose.rotate(Eigen::AngleAxisf(q.at(joint.motor_index), joint.axis.normalized()));
        }
    }
    return pose;
}

inline void validate_snapshot(const Snapshot& s) {
    if (!s.orientation.coeffs().allFinite() || std::abs(s.orientation.norm() - 1) > 0.05f ||
        !s.angular_velocity.allFinite()) throw std::runtime_error("Invalid pelvis IMU data");
    for (int i = 0; i < J; ++i) {
        if (!std::isfinite(s.q[i]) || !std::isfinite(s.dq[i])) throw std::runtime_error("Invalid motor data");
    }
}

// 参考帧：策略顺序关节角/角速度 + 14 个刚体的世界位姿（仅 anchor 与脚踝被使用）。
struct Reference {
    Joints q{}, dq{};
    std::array<Vec, B> positions;
    std::array<Quat, B> orientations;
};

// BMG1V1 参考表：24 字节头 + 每帧 156 个 float，四元数为 wxyz。
class Motion {
public:
    explicit Motion(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open reference: " + path);
        const auto size = file.tellg();
        file.seekg(0);
        char magic[8];
        std::uint32_t joints = 0, bodies = 0;
        file.read(magic, 8);
        file.read(reinterpret_cast<char*>(&frames_), 4);
        file.read(reinterpret_cast<char*>(&joints), 4);
        file.read(reinterpret_cast<char*>(&bodies), 4);
        file.read(reinterpret_cast<char*>(&dt_), 4);
        const std::uint16_t endian = 1;
        if (*reinterpret_cast<const char*>(&endian) != 1 || std::memcmp(magic, "BMG1V1\0\0", 8) ||
            joints != J || bodies != B || frames_ < 2 || frames_ > 10000000 ||
            std::abs(dt_ - 0.02f) > 1e-6f ||
            size != static_cast<std::streamoff>(24 + std::uint64_t(frames_) * RecordSize * sizeof(float)))
            throw std::runtime_error("Invalid/incompatible reference table");
        data_.resize(std::size_t(frames_) * RecordSize);
        file.read(reinterpret_cast<char*>(data_.data()), data_.size() * sizeof(float));
        if (!file) throw std::runtime_error("Truncated reference table");
        for (float value : data_) if (!std::isfinite(value)) throw std::runtime_error("Non-finite reference");
    }
    Reference frame(std::size_t index) const {
        if (index >= frames_) throw std::out_of_range("Reference motion ended");
        const float* p = data_.data() + index * RecordSize;
        Reference result;
        std::copy(p, p + J, result.q.begin()); p += J;
        std::copy(p, p + J, result.dq.begin()); p += J;
        for (auto& pos : result.positions) { pos = Vec(p[0], p[1], p[2]); p += 3; }
        for (auto& quat : result.orientations) {
            quat = Quat(p[0], p[1], p[2], p[3]);
            if (std::abs(quat.norm() - 1) > 0.05f) throw std::runtime_error("Invalid reference quaternion");
            quat.normalize();
            p += 4;
        }
        return result;
    }
    std::size_t frames() const { return frames_; }
    float dt() const { return dt_; }
private:
    std::uint32_t frames_ = 0;
    float dt_ = 0;
    std::vector<float> data_;
};

struct Config {
    std::array<int, J> sdk_indices{};
    Joints default_q{}, kp{}, kd{}, action_scale{}, lower{}, upper{}, effort{};
    Chain torso;
    float dt = 0.02f;
    void validate() const {
        std::set<int> ids(sdk_indices.begin(), sdk_indices.end());
        if (ids.size() != J || *ids.begin() != 0 || *ids.rbegin() != J - 1 || std::abs(dt - 0.02f) > 1e-6f)
            throw std::runtime_error("Invalid SDK joint permutation or policy dt");
        for (int i = 0; i < J; ++i) {
            if (!(kp[i] > 0 && kd[i] >= 0 && action_scale[i] > 0 && effort[i] > 0 && lower[i] < upper[i]))
                throw std::runtime_error("Invalid joint control parameters");
        }
        if (torso.empty()) throw std::runtime_error("Missing torso kinematic chain");
        for (const auto& segment : torso) {
            if (segment.motor_index < -1 || segment.motor_index >= J || !segment.xyz.allFinite() ||
                !segment.rpy.allFinite() || !segment.axis.allFinite() || segment.axis.norm() < 0.9f)
                throw std::runtime_error("Invalid torso kinematic segment");
        }
    }
};

template <std::size_t N>
inline std::array<float, N> read_array(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != N) throw std::runtime_error("Invalid package vector length");
    std::array<float, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = node[i].as<float>();
        if (!std::isfinite(result[i])) throw std::runtime_error("Non-finite package parameter");
    }
    return result;
}

inline Config load_config(const std::filesystem::path& path) {
    const auto yaml = YAML::LoadFile(path.string());
    if (yaml["format"].as<std::string>() != Format ||
        yaml["observation_dim"].as<int>() != O || yaml["action_dim"].as<int>() != J ||
        yaml["anchor_body_name"].as<std::string>() != "torso_link" ||
        !yaml["normalizer"]["embedded"].as<bool>()) throw std::runtime_error("Incompatible holo_wbt package");
    Config cfg;
    cfg.dt = yaml["step_dt"].as<float>();
    cfg.default_q = read_array<J>(yaml["default_joint_pos"]);
    cfg.kp = read_array<J>(yaml["kp"]);
    cfg.kd = read_array<J>(yaml["kd"]);
    cfg.action_scale = read_array<J>(yaml["action_scale"]);
    cfg.effort = read_array<J>(yaml["effort_limit"]);
    cfg.lower = read_array<J>(yaml["joint_lower"]);
    cfg.upper = read_array<J>(yaml["joint_upper"]);
    if (yaml["sdk_indices"].size() != J) throw std::runtime_error("Invalid SDK mapping");
    for (int i = 0; i < J; ++i) cfg.sdk_indices[i] = yaml["sdk_indices"][i].as<int>();
    for (const auto& segment : yaml["kinematics"]["torso"]) {
        auto xyz = read_array<3>(segment["xyz"]), rpy = read_array<3>(segment["rpy"]), axis = read_array<3>(segment["axis"]);
        cfg.torso.push_back({segment["motor_index"].as<int>(), Vec(xyz.data()), Vec(rpy.data()), Vec(axis.data())});
    }
    cfg.validate();
    return cfg;
}

// 单输入 [1,154] / 单输出 [1,29] 的 ONNX 封装；归一化内嵌在图中。
class Runner {
public:
    explicit Runner(const std::filesystem::path& path) : env_(ORT_LOGGING_LEVEL_WARNING, "holo_wbt") {
        options_.SetIntraOpNumThreads(1);
        options_.SetInterOpNumThreads(1);
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, path.c_str(), options_);
        Ort::AllocatorWithDefaultOptions allocator;
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1 ||
            std::string(session_->GetInputNameAllocated(0, allocator).get()) != "obs" ||
            std::string(session_->GetOutputNameAllocated(0, allocator).get()) != "actions" ||
            session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1, O} ||
            session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1, J} ||
            session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Model must be the packaged raw-obs [1,154] -> actions [1,29] ONNX");
    }
    Joints act(const std::array<float, O>& obs) {
        const std::array<int64_t, 2> shape{1, O};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(obs.data()), O, shape.data(), shape.size());
        const char* inputs[]{"obs"};
        const char* outputs[]{"actions"};
        auto result = session_->Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
        Joints action{};
        std::copy_n(result[0].GetTensorData<float>(), J, action.begin());
        return action;
    }
private:
    Ort::Env env_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
};

// 观测拼装 + 动作换算。holosoma 的 concatenate 组按术语名字母序拼接，因此顺序是：
// actions(29) base_ang_vel(3) dof_pos(29) dof_vel(29) motion_command(58) motion_ref_ori_b(6)。
class Core {
public:
    explicit Core(Config cfg) : cfg_(std::move(cfg)) { cfg_.validate(); }
    void reset(const Snapshot& s) {
        validate_snapshot(s);
        last_action_.fill(0);
        last_target_ = s.q;
    }
    std::array<float, O> observe(const Snapshot& s, const Reference& ref) {
        validate_snapshot(s);
        const auto torso = forward_kinematics(cfg_.torso, s.q);
        const Quat torso_q = s.orientation * Quat(torso.rotation());
        const Quat ref_q = ref.orientations[7];  // anchor body: torso_link
        const auto rot = (torso_q.conjugate() * ref_q).toRotationMatrix();
        std::array<float, O> obs{};
        int k = 0;
        auto append = [&](const auto& values) { for (float v : values) obs[k++] = v; };
        append(last_action_);
        for (int i = 0; i < 3; ++i) obs[k++] = s.angular_velocity[i];
        for (int i = 0; i < J; ++i) obs[k++] = s.q[cfg_.sdk_indices[i]] - cfg_.default_q[i];
        for (int i = 0; i < J; ++i) obs[k++] = s.dq[cfg_.sdk_indices[i]];
        append(ref.q);
        append(ref.dq);
        // 与 Isaac Lab 的 matrix[..., :2].reshape 一致：行优先取前两列。
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 2; ++j) obs[k++] = rot(i, j);
        for (float v : obs) if (!std::isfinite(v)) throw std::runtime_error("Invalid observation");
        return obs;
    }
    Joints targets(const Joints& action, const Snapshot& state, float max_speed = 8.0f) {
        Joints sdk{};
        clipped_ = 0;
        for (int i = 0; i < J; ++i) {
            const int id = cfg_.sdk_indices[i];
            if (!std::isfinite(action[i]) || std::abs(action[i]) > 100)
                throw std::runtime_error("Invalid policy action");
            float desired = cfg_.default_q[i] + cfg_.action_scale[i] * action[i];
            float limited = std::clamp(desired, cfg_.lower[i] + 0.01f, cfg_.upper[i] - 0.01f);
            limited = std::clamp(limited, last_target_[id] - max_speed * cfg_.dt, last_target_[id] + max_speed * cfg_.dt);
            // 用训练时的执行器力矩上限约束 PD 需求。
            const float center = state.q[id] + cfg_.kd[i] * state.dq[id] / cfg_.kp[i];
            const float radius = cfg_.effort[i] / cfg_.kp[i];
            limited = std::clamp(limited, center - radius, center + radius);
            limited = std::clamp(limited, cfg_.lower[i] + 0.01f, cfg_.upper[i] - 0.01f);
            clipped_ += std::abs(desired - limited) > 1e-5f;
            sdk[id] = limited;
        }
        last_target_ = sdk;
        last_action_ = action;  // 训练观测到的是上一拍原始动作，不是限幅后的目标
        return sdk;
    }
    int clipped() const { return clipped_; }
    const Config& config() const { return cfg_; }
private:
    Config cfg_;
    Joints last_action_{}, last_target_{};
    int clipped_ = 0;
};
}  // namespace holowbt
