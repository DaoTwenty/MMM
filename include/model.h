#pragma once

#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#elif defined(USE_TORCH)
#include <torch/torch.h>
#include <torch/script.h>
#else
#error "You must define either USE_ONNX or USE_TORCH"
#endif

namespace mmm {

struct ModelConfig {
    std::string path;
    bool cached = false;
#ifdef USE_ONNX
    bool coreml = false;
#endif
    int vocab_size = 16000;
};

class IModel {
public:
    virtual ~IModel() = default;
    virtual std::vector<float> forward(const std::vector<int64_t>& input_ids) = 0;
    virtual bool is_cached() { return false; }
    virtual void reset_cache() {}
};

#ifdef USE_ONNX
// -------------------- ONNX IMPLEMENTATION --------------------
class CausalLM : public IModel {
public:
    CausalLM(const std::string& model_path, int vocab_size, bool coreML = false);
    std::vector<float> forward(const std::vector<int64_t>& input_ids) override;

protected:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
    int vocab_size;
};

class CausalLMCached : public CausalLM {
public:
    CausalLMCached(const std::string& model_path, int vocab_size, bool coreML = false);
    std::vector<float> forward(const std::vector<int64_t>& input_ids) override;
    void reset_cache() override;
    bool is_cached() override { return true; }

private:
    std::vector<std::string> main_input_names;
    std::vector<std::string> past_input_names;
    std::vector<std::string> past_output_names;
    std::string logits_output_name;

    std::vector<Ort::Value> past_key_values;

    bool has_input(const std::string& name) const {
        return std::any_of(
            main_input_names.begin(),
            main_input_names.end(),
            [&](const std::string& n) { return name == n; }
        );
    }
};
#endif // USE_ONNX


#ifdef USE_TORCH
// -------------------- TORCH IMPLEMENTATION --------------------
class CausalLM : public IModel {
public:
    CausalLM(const std::string& model_path, int vocab_size);
    std::vector<float> forward(const std::vector<int64_t>& input_ids) override;

protected:
    torch::jit::script::Module model;
    int vocab_size;
};

class CausalLMCached : public CausalLM {
public:
    CausalLMCached(const std::string& model_path, int vocab_size);
    std::vector<float> forward(const std::vector<int64_t>& input_ids) override;
    void reset_cache() override;
    bool is_cached() override { return true; }

private:
    std::vector<c10::IValue> past_key_values;
};
#endif // USE_TORCH

} // namespace mmm
