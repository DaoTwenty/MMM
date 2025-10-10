#include "model.h"

namespace mmm {

#if defined(USE_ONNX)
// ============================================================================
// ONNX Runtime backend
// ============================================================================
static Ort::SessionOptions make_options(bool useCoreML) {
    Ort::SessionOptions opts;
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    opts.DisableMemPattern();
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    if (useCoreML) {
        std::unordered_map<std::string, std::string> provider_options;
        provider_options["ModelFormat"] = "MLProgram";
        provider_options["MLComputeUnits"] = "ALL";
        provider_options["RequireStaticInputShapes"] = "0";
        provider_options["EnableOnSubgraphs"] = "0";
        opts.AppendExecutionProvider("CoreML", provider_options);
    }

    return opts;
}

CausalLM::CausalLM(const std::string& model_path, int vocab_size, bool coreML)
    : env(ORT_LOGGING_LEVEL_WARNING, "CausalLM"),
      session_options(make_options(coreML)),
      session(env, model_path.c_str(), session_options),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      vocab_size(vocab_size) {}

CausalLMCached::CausalLMCached(const std::string& model_path, int vocab_size, bool coreML)
    : CausalLM(model_path, vocab_size, coreML) {
    Ort::AllocatorWithDefaultOptions allocator;

    // Inspect inputs
    size_t num_inputs = session.GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        std::string iname = session.GetInputNameAllocated(i, allocator).get();
        if (iname == "input_ids" || iname == "attention_mask" || iname == "position_ids")
            main_input_names.push_back(strdup(iname.c_str()));
        else if (iname.find("past") != std::string::npos)
            past_input_names.push_back(strdup(iname.c_str()));
    }

    // Inspect outputs
    size_t num_outputs = session.GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        std::string oname = session.GetOutputNameAllocated(i, allocator).get();
        if (oname == "logits")
            logits_output_name = strdup(oname.c_str());
        else if (oname.find("present") != std::string::npos ||
                 oname.find("past") != std::string::npos)
            past_output_names.push_back(strdup(oname.c_str()));
    }
}

void CausalLMCached::reset_cache() { past_key_values.clear(); }

std::vector<float> CausalLMCached::forward(const std::vector<int64_t>& input_ids_raw) {
    const size_t batch_size = 1;
    const size_t seq_len = input_ids_raw.size();

    // 1. Convert to tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(seq_len)};
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(input_ids_raw.data()),
        input_ids_raw.size(), input_shape.data(), input_shape.size());

    // 2. Past key values
    int64_t past_len = 0;
    if (!past_key_values.empty()) {
        const auto& past_shape = past_key_values[0].GetTensorTypeAndShapeInfo().GetShape();
        if (past_shape.size() >= 4) past_len = past_shape[3];
    } else {
        constexpr int64_t num_heads = 8, head_size = 64;
        past_key_values.clear();
        for (size_t i = 0; i < past_input_names.size(); ++i) {
            std::vector<int64_t> shape = {2, 1, num_heads, 0, head_size};
            Ort::Value empty_tensor = Ort::Value::CreateTensor<float>(
                memory_info, nullptr, 0, shape.data(), shape.size());
            past_key_values.push_back(std::move(empty_tensor));
        }
    }

    // 3. Optional tensors
    std::vector<const char*> input_names{"input_ids"};
    std::vector<Ort::Value> run_inputs;
    run_inputs.push_back(std::move(input_ids_tensor));

    auto make_tensor = [&](const std::string& name, std::vector<int64_t>& data,
                           const std::vector<int64_t>& shape) {
        if (has_input(name)) {
            input_names.push_back(name.c_str());
            run_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, data.data(), data.size(), shape.data(), shape.size()));
        }
    };

    // Position IDs
    if (has_input("position_ids")) {
        std::vector<int64_t> pos(seq_len);
        for (size_t i = 0; i < seq_len; ++i) pos[i] = past_len + i;
        make_tensor("position_ids", pos, input_shape);
    }

    // Attention mask
    if (has_input("attention_mask")) {
        const int64_t total_len = past_len + seq_len;
        std::vector<int64_t> mask(total_len, 1);
        std::vector<int64_t> mask_shape = {1, total_len};
        make_tensor("attention_mask", mask, mask_shape);
    }

    // Past
    for (size_t i = 0; i < past_input_names.size(); ++i) {
        input_names.push_back(past_input_names[i]);
        run_inputs.push_back(std::move(past_key_values[i]));
    }

    // 4. Outputs
    std::vector<const char*> output_names{logits_output_name};
    output_names.insert(output_names.end(), past_output_names.begin(), past_output_names.end());

    auto outputs = session.Run(
        Ort::RunOptions{nullptr}, input_names.data(), run_inputs.data(),
        run_inputs.size(), output_names.data(), output_names.size());

    // 5. Update cache
    past_key_values.clear();
    for (size_t i = 1; i < outputs.size(); ++i)
        past_key_values.push_back(std::move(outputs[i]));

    // 6. Extract last token logits
    Ort::Value& logits_tensor = outputs[0];
    float* data = logits_tensor.GetTensorMutableData<float>();
    auto shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
    int64_t vocab = shape.back();
    int64_t offset = (shape.size() == 3) ? (shape[1] - 1) * vocab : 0;

    return std::vector<float>(data + offset, data + offset + vocab);
}

std::vector<float> CausalLM::forward(const std::vector<int64_t>& input_ids) {
    std::vector<int64_t> shape = {1, static_cast<int64_t>(input_ids.size())};

    // Inputs
    Ort::Value ids = Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(input_ids.data()),
        input_ids.size(), shape.data(), shape.size());

    std::vector<int64_t> mask(input_ids.size(), 1);
    Ort::Value am = Ort::Value::CreateTensor<int64_t>(
        memory_info, mask.data(), mask.size(), shape.data(), shape.size());

    std::vector<int64_t> pos(input_ids.size());
    std::iota(pos.begin(), pos.end(), 0);
    Ort::Value pid = Ort::Value::CreateTensor<int64_t>(
        memory_info, pos.data(), pos.size(), shape.data(), shape.size());

    const char* in_names[] = {"input_ids", "attention_mask", "position_ids"};
    const char* out_names[] = {"logits"};

    std::vector<Ort::Value> ins;
    ins.push_back(std::move(ids));
    ins.push_back(std::move(am));
    ins.push_back(std::move(pid));

    auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, ins.data(),
                            ins.size(), out_names, 1);

    float* logits = outs[0].GetTensorMutableData<float>();
    const size_t len = input_ids.size();
    return std::vector<float>(logits + (len - 1) * vocab_size,
                              logits + len * vocab_size);
}

#elif defined(USE_TORCH)
// ============================================================================
// Torch backend
// ============================================================================
CausalLM::CausalLM(const std::string& model_path, int vocab_size)
    : model(torch::jit::load(model_path)), vocab_size(vocab_size) {}

std::vector<float> CausalLM::forward(const std::vector<int64_t>& input_ids) {
    torch::NoGradGuard no_grad;
    const int64_t seq_len = input_ids.size();

    auto ids = torch::from_blob((void*)input_ids.data(), {1, seq_len}, torch::kInt64).clone();
    auto mask = torch::ones({1, seq_len}, torch::kInt64);

    auto out = model.forward({ids, mask}).toTuple();
    torch::Tensor logits = out->elements()[0].toTensor().index({0, seq_len - 1});

    std::vector<float> result(logits.numel());
    std::memcpy(result.data(), logits.data_ptr<float>(), result.size() * sizeof(float));
    return result;
}

CausalLMCached::CausalLMCached(const std::string& model_path, int vocab_size)
    : CausalLM(model_path, vocab_size) {}

void CausalLMCached::reset_cache() { past_key_values.clear(); }

std::vector<float> CausalLMCached::forward(const std::vector<int64_t>& input_ids_vec) {
    torch::NoGradGuard no_grad;
    const int64_t seq_len = input_ids_vec.size();
    const int batch = 1, num_layers = 8, n_head = 8, head_dim = 64;

    auto ids = torch::from_blob((void*)input_ids_vec.data(), {batch, seq_len}, torch::kInt64).clone();
    int64_t past_len = 0;

    if (!past_key_values.empty()) {
        auto tup = past_key_values[0].toTuple();
        past_len = tup->elements()[0].toTensor().size(2);
    } else {
        past_key_values.reserve(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            auto key = torch::empty({batch, n_head, 0, head_dim});
            auto val = torch::empty({batch, n_head, 0, head_dim});
            past_key_values.push_back(c10::ivalue::Tuple::create({key, val}));
        }
    }

    auto mask = torch::ones({batch, past_len + seq_len}, torch::kInt64);
    c10::IValue past_tuple = c10::ivalue::Tuple::create(past_key_values);

    auto out = model.forward({ids, mask, past_tuple}).toTuple();
    auto logits = out->elements()[0].toTensor().index({0, seq_len - 1});

    std::vector<float> result(logits.numel());
    std::memcpy(result.data(), logits.data_ptr<float>(), result.size() * sizeof(float));

    past_key_values.clear();
    for (auto &layer : out->elements()[1].toTuple()->elements())
        past_key_values.push_back(layer);

    return result;
}
#endif // USE_ONNX / USE_TORCH

} // namespace mmm
