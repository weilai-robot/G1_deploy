#pragma once

// HugWBC ONNX Runtime inference with strict contract checks. Loading verifies
// the file hash against the package manifest value, so a swapped or truncated
// policy is rejected before any robot command is produced.

#include "hugwbc/package.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>

namespace hugwbc {

class Runner {
public:
    explicit Runner(const std::filesystem::path& path, const std::string& expected_sha)
        : env_(ORT_LOGGING_LEVEL_WARNING, "hugwbc") {
        if (!expected_sha.empty()) {
            const std::string actual = sha256_file(path);
            if (actual != expected_sha)
                throw std::runtime_error("policy.onnx hash mismatch: expected " + expected_sha +
                                         ", got " + actual);
        }
        options_.SetIntraOpNumThreads(1);
        options_.SetInterOpNumThreads(1);
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, path.c_str(), options_);
        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_shape =
            session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const auto output_shape =
            session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1 ||
            std::string(session_->GetInputNameAllocated(0, allocator).get()) != "obs_history" ||
            std::string(session_->GetOutputNameAllocated(0, allocator).get()) != "actions" ||
            input_shape != std::vector<std::int64_t>{1, HIST, OBS} ||
            output_shape != std::vector<std::int64_t>{1, J} ||
            session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Model must be the HugWBC [1,5,88] -> [1,23] float ONNX");
    }

    Joints act(const std::array<float, OBS * HIST>& history) {
        std::array<float, OBS * HIST> input = history;  // session reads, never writes
        const std::array<std::int64_t, 3> shape{1, HIST, OBS};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory, input.data(), input.size(), shape.data(), shape.size());
        const char* inputs[]{"obs_history"};
        const char* outputs[]{"actions"};
        auto result = session_->Run(Ort::RunOptions{nullptr}, inputs, &input_tensor, 1, outputs, 1);
        Joints action{};
        std::copy_n(result[0].GetTensorData<float>(), J, action.begin());
        return action;
    }

private:
    Ort::Env env_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
};

}  // namespace hugwbc