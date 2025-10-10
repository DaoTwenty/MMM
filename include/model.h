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

struct ModelConfig {
    std::string path;
    std::string type = "ONNX";
    bool cached = false;
    bool coreml = false;
    int vocab_size = 16000;
};

class IModel {
public:
virtual ~IModel() = default;
virtual std::vector<float> forward(const std::vector<int64_t>& input_ids) = 0;
virtual bool is_cached() {return false;}
virtual void reset_cache() {}
};

class CausalLM : public IModel {
public:

    CausalLM(const std::string& model_path, int vocab_size, bool coreML = false);

    // Forward pass (batch size = 1 for simplicity)
    std::vector<float> forward(const std::vector<int64_t>& input_ids) override;

protected:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

    int vocab_size;

};

class CausalLMCached : public CausalLM{
public:

    CausalLMCached(const std::string& model_path, int vocab_size, bool coreML = false);

    std::vector<float> forward(const std::vector<int64_t>& input_ids) override;

    void reset_cache() override;

    bool is_cached() override {return true;}

private:

    std::vector<const char*> main_input_names;
    std::vector<const char*> past_input_names;
    std::vector<const char*> past_output_names;
    const char* logits_output_name = nullptr;

    std::vector<Ort::Value> past_key_values;

    bool has_input(const std::string& name) const {
        return std::any_of(main_input_names.begin(), main_input_names.end(),
                           [&](const char* n) { return name == n; });
    }

};

class CausalLMTorch : public IModel {
public:
    CausalLMTorch(const std::string& model_path, int vocab_size);

    // Forward pass (batch size = 1 for simplicity)
    std::vector<float> forward(const std::vector<int64_t>& input_ids) override;

protected:
    torch::jit::script::Module model;

    int vocab_size;

};

class CausalLMTorchCached : public CausalLMTorch {
public:
    CausalLMTorchCached(const std::string& model_path, int vocab_size);

    std::vector<float> forward(const std::vector<int64_t>& input_ids) override;

    void reset_cache() override;

    bool is_cached() override {return true;}

private:
    std::vector<c10::IValue> past_key_values;
};


}