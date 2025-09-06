#pragma once

#include <onnxruntime_cxx_api.h>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

namespace mmm {

class IModel {
public:
virtual ~IModel() = default;
virtual std::vector<float> forward(const std::vector<int64_t>& input_ids) = 0;
};

class CausalLM : IModel {
public:
    CausalLM(const std::string& model_path, int vocab_size, int pad_token_id);

    // Forward pass (batch size = 1 for simplicity)
    std::vector<float> forward(const std::vector<int64_t>& input_ids) override;

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

    int vocab_size;
    int pad_token_id;

};

class CausalLMTorch : IModel {
public:
    CausalLMTorch(const std::string& model_path, int vocab_size, int pad_token_id);

    // Forward pass (batch size = 1 for simplicity)
    std::vector<float> forward(const std::vector<int64_t>& input_ids) override;

private:
    torch::jit::script::Module model;

    int vocab_size;
    int pad_token_id;

};


}