#pragma once

#include "velocity11k/core.h"
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace velocity11k {
class Runner {
public:
    explicit Runner(const std::filesystem::path& model) : env_(ORT_LOGGING_LEVEL_WARNING,"velocity11k") {
        options_.SetIntraOpNumThreads(1);
        options_.SetInterOpNumThreads(1);
        options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_=std::make_unique<Ort::Session>(env_,model.c_str(),options_);
        Ort::AllocatorWithDefaultOptions allocator;
        if(session_->GetInputCount()!=1 || session_->GetOutputCount()!=1 ||
           std::string(session_->GetInputNameAllocated(0,allocator).get())!="obs" ||
           std::string(session_->GetOutputNameAllocated(0,allocator).get())!="actions" ||
           session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,O} ||
           session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,J} ||
           session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
           session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("11k model must have float obs [1,480] -> actions [1,29]");
        const auto metadata=session_->GetModelMetadata();
        auto checkpoint=metadata.LookupCustomMetadataMapAllocated("checkpoint_sha256",allocator);
        auto contract=metadata.LookupCustomMetadataMapAllocated("contract_sha256",allocator);
        if(!checkpoint || !contract || std::string(checkpoint.get())!=CHECKPOINT_SHA256 || std::string(contract.get())!=CONTRACT_SHA256)
            throw std::runtime_error("11k model provenance does not match this controller's compiled contract");
    }
    Joints act(Observation obs) {
        finite(obs);
        const std::array<int64_t,2> shape{1,O};
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto input=Ort::Value::CreateTensor<float>(memory,obs.data(),O,shape.data(),shape.size());
        const char* inputs[]{"obs"}; const char* outputs[]{"actions"};
        auto result=session_->Run(Ort::RunOptions{nullptr},inputs,&input,1,outputs,1);
        Joints actions{};
        std::copy_n(result[0].GetTensorData<float>(),J,actions.begin());
        finite(actions);
        return actions;
    }
private:
    Ort::Env env_;
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
};
} // namespace velocity11k
