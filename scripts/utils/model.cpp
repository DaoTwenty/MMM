#include "model.h"

namespace mmm {

CausalLM::CausalLM(const std::string& model_path, int vocab_size, int pad_token_id)
    : env(ORT_LOGGING_LEVEL_WARNING, "CausalLM"),
        session_options(),
        session(env, model_path.c_str(), session_options),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        vocab_size(vocab_size),
        pad_token_id(pad_token_id) {
    session_options.SetIntraOpNumThreads(1);
}

CausalLMTorch::CausalLMTorch(const std::string& model_path, int vocab_size, int pad_token_id)
    : model(torch::jit::load(model_path)),  // initialize the reference here
      vocab_size(vocab_size),
      pad_token_id(pad_token_id) {
}

std::vector<float> CausalLM::forward(const std::vector<int64_t>& input_ids) {
    std::vector<int64_t> input_shape = {1, (int64_t)input_ids.size()};

    // 1. input_ids tensor
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t*>(input_ids.data()), input_ids.size(),
        input_shape.data(), input_shape.size());

    // 2. attention_mask tensor (all ones if no padding)
    std::vector<int64_t> attention_mask(input_ids.size(), 1);
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        attention_mask.data(), attention_mask.size(),
        input_shape.data(), input_shape.size());

    // 3. position_ids tensor
    std::vector<int64_t> position_ids(input_ids.size());
    for (size_t i = 0; i < input_ids.size(); ++i) position_ids[i] = i;
    Ort::Value position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        position_ids.data(), position_ids.size(),
        input_shape.data(), input_shape.size());

    // Names
    const char* input_names[] = {"input_ids", "attention_mask", "position_ids"};

    // Run session
    const char* output_names[] = {"logits"};
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_ids_tensor));
    input_tensors.push_back(std::move(attention_mask_tensor));
    input_tensors.push_back(std::move(position_ids_tensor));

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), input_tensors.size(),
        output_names, 1);

    // Extract last-token logits
    float* logits_ptr = output_tensors.front().GetTensorMutableData<float>();
    size_t seq_len = input_ids.size();
    return std::vector<float>(logits_ptr + (seq_len - 1) * vocab_size,
                            logits_ptr + seq_len * vocab_size);
}

std::vector<float> CausalLMTorch::forward(const std::vector<int64_t>& input_ids) {
    torch::NoGradGuard no_grad;

    int64_t seq_len = input_ids.size();
    auto device = torch::kCPU; // or torch::kCUDA if GPU

    // 1. input_ids tensor
    torch::Tensor input_ids_tensor = torch::from_blob(
        const_cast<int64_t*>(input_ids.data()),
        {1, seq_len}, torch::kInt64).to(device);

    // 2. attention_mask (all ones if no padding)
    torch::Tensor attention_mask_tensor = torch::ones({1, seq_len}, torch::kInt64).to(device);

    // 3. position_ids
    std::vector<int64_t> pos(seq_len);
    std::iota(pos.begin(), pos.end(), 0);
    torch::Tensor position_ids_tensor = torch::from_blob(pos.data(), {1, seq_len}, torch::kInt64).to(device);

    // Forward
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_ids_tensor);
    inputs.push_back(attention_mask_tensor);
    inputs.push_back(position_ids_tensor);

    auto output = model.forward(inputs).toTensor(); // shape [1, seq_len, vocab_size]

    // Get last-token logits
    torch::Tensor last_token_logits = output.index({0, seq_len - 1, torch::indexing::Slice()});

    // Convert to std::vector<float>
    std::vector<float> logits(last_token_logits.data_ptr<float>(), last_token_logits.data_ptr<float>() + vocab_size);
    return logits;
}

}