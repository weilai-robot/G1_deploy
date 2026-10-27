#pragma once

#include "beyondmimic/core.h"
#include <yaml-cpp/yaml.h>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <memory>

namespace beyondmimic {
template<std::size_t N>
std::array<float,N> read_array(const YAML::Node& node) {
    if (!node.IsSequence() || node.size()!=N) throw std::runtime_error("Invalid package vector length");
    std::array<float,N> result{};
    for(std::size_t i=0;i<N;++i) {
        result[i] = node[i].as<float>();
        if (!std::isfinite(result[i])) throw std::runtime_error("Non-finite package parameter");
    }
    return result;
}
inline Config load_config(const std::filesystem::path& path) {
    const auto yaml = YAML::LoadFile(path.string());
    if (yaml["format"].as<std::string>()!="beyondmimic_g1_160_v1" ||
        yaml["observation_dim"].as<int>()!=O || yaml["action_dim"].as<int>()!=J ||
        yaml["anchor_body_name"].as<std::string>()!="torso_link" ||
        !yaml["normalizer"]["embedded"].as<bool>()) throw std::runtime_error("Incompatible BeyondMimic package");
    Config cfg;
    cfg.dt = yaml["step_dt"].as<float>();
    cfg.default_q = read_array<J>(yaml["default_joint_pos"]);
    cfg.kp = read_array<J>(yaml["kp"]);
    cfg.kd = read_array<J>(yaml["kd"]);
    cfg.action_scale = read_array<J>(yaml["action_scale"]);
    cfg.effort = read_array<J>(yaml["effort_limit"]);
    cfg.lower = read_array<J>(yaml["joint_lower"]);
    cfg.upper = read_array<J>(yaml["joint_upper"]);
    if (yaml["sdk_indices"].size()!=J) throw std::runtime_error("Invalid SDK mapping");
    for(int i=0;i<J;++i) cfg.sdk_indices[i]=yaml["sdk_indices"][i].as<int>();
    auto chain = [&](const std::string& name) {
        Chain result;
        for(const auto& n : yaml["kinematics"][name]) {
            auto xyz=read_array<3>(n["xyz"]), rpy=read_array<3>(n["rpy"]), axis=read_array<3>(n["axis"]);
            result.push_back({n["motor_index"].as<int>(), Vec(xyz.data()), Vec(rpy.data()), Vec(axis.data())});
        }
        return result;
    };
    cfg.torso=chain("torso"); cfg.left_foot=chain("left_foot"); cfg.right_foot=chain("right_foot");
    cfg.validate();
    return cfg;
}

class Runner {
public:
    explicit Runner(const std::filesystem::path& path) : env_(ORT_LOGGING_LEVEL_WARNING,"beyondmimic") {
        options_.SetIntraOpNumThreads(1);
        options_.SetInterOpNumThreads(1);
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_,path.c_str(),options_);
        Ort::AllocatorWithDefaultOptions allocator;
        if (session_->GetInputCount()!=1 || session_->GetOutputCount()!=1 ||
            std::string(session_->GetInputNameAllocated(0,allocator).get())!="obs" ||
            std::string(session_->GetOutputNameAllocated(0,allocator).get())!="actions" ||
            session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,O} ||
            session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,J} ||
            session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Model must be the packaged raw-obs [1,160] -> actions [1,29] ONNX");
    }
    Joints act(std::array<float,O> obs) {
        const std::array<int64_t,2> shape{1,O};
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto input=Ort::Value::CreateTensor<float>(memory,obs.data(),O,shape.data(),shape.size());
        const char* inputs[]{"obs"}; const char* outputs[]{"actions"};
        auto result=session_->Run(Ort::RunOptions{nullptr},inputs,&input,1,outputs,1);
        Joints action{};
        std::copy_n(result[0].GetTensorData<float>(),J,action.begin());
        return action;
    }
private:
    Ort::Env env_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
};
} // namespace beyondmimic
