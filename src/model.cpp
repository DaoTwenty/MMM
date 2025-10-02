#include "model.h"

namespace mmm {

static Ort::SessionOptions make_options(bool useCoreML) {
    Ort::SessionOptions opts;
    //opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    //opts.SetLogSeverityLevel(0);  // verbose
    //opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    //opts.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    opts.SetInterOpNumThreads(1);
    //opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    opts.DisableMemPattern();               // session_options.DisableMemPattern();  (C API: OrtSessionOptionsDisableMemPattern)
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    std::unordered_map<std::string, std::string> provider_options;
    if (useCoreML) {
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
            Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(i, allocator);
            std::string iname(input_name_ptr.get());

            if (iname == "input_ids" ||
                iname == "attention_mask" ||
                iname == "position_ids") {
                if (!iname.empty())
                    main_input_names.push_back(strdup(iname.c_str()));
            } else if (iname.find("past") != std::string::npos) {
                if (!iname.empty())
                    past_input_names.push_back(strdup(iname.c_str()));
            }
        }

        // Inspect outputs
        size_t num_outputs = session.GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(i, allocator);
            std::string oname(output_name_ptr.get());

            if (oname == "logits") {
                logits_output_name = strdup(oname.c_str());
            } else if (oname.find("present") != std::string::npos ||
                       oname.find("past") != std::string::npos) {
                past_output_names.push_back(strdup(oname.c_str()));
            }
        }

        std::cout << "✅ Loaded model with "
                  << main_input_names.size() << " main inputs, "
                  << past_input_names.size() << " past inputs, "
                  << "and " << past_output_names.size() << " past outputs.\n";
}

CausalLMTorch::CausalLMTorch(const std::string& model_path, int vocab_size)
    : model(torch::jit::load(model_path)),  // initialize the reference here
      vocab_size(vocab_size) {
}

CausalLMTorchCached::CausalLMTorchCached(const std::string& model_path, int vocab_size)
    : CausalLMTorch(model_path, vocab_size) 
    {}

void CausalLMCached::reset_cache() {
    past_key_values.clear();
}

void CausalLMTorchCached::reset_cache() {
    past_key_values.clear();
}

std::vector<float> CausalLMCached::forward(const std::vector<int64_t>& input_ids_raw) {
    // NOTE: model expects int32 for ids/masks/positions
    size_t batch_size = 1;
    size_t seq_len = input_ids_raw.size();

    // Convert input_ids to int32 (the model expects int32)
    std::vector<int32_t> input_ids(seq_len);
    for (size_t i = 0; i < seq_len; ++i) input_ids[i] = static_cast<int32_t>(input_ids_raw[i]);

    std::cout << "=== Forward call ===\n";
    std::cout << "input_ids: [";
    for (auto id : input_ids) std::cout << id << " ";
    std::cout << "]\n";

    // -------------------------
    // 1. input_ids tensor (int32)
    // -------------------------
    std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len)};
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int32_t>(
        memory_info,
        input_ids.data(), input_ids.size(),
        input_shape.data(), input_shape.size());

    // -------------------------
    // 2. past key/value tensors (float32)
    //    Model past shape: [2, batch, num_heads, past_seq_len, head_size]
    // -------------------------
    int64_t past_len = 0;
    if (!past_key_values.empty()) {
        auto past_shape = past_key_values[0].GetTensorTypeAndShapeInfo().GetShape();
        // past_shape: [2, batch, num_heads, past_seq_len, head_size]
        if (past_shape.size() >= 4) past_len = past_shape[3];
    }

    // initialize empty past tensors on first pass (only if vector truly empty)
    if (past_key_values.empty()) {
        std::cout << "Initializing empty past tensors (first pass)\n";
        past_key_values.clear();
        // assume num_heads=8, head_size=64 per model description; batch_size=1, outer dim=2
        for (size_t i = 0; i < past_input_names.size(); ++i) {
            std::vector<int64_t> zero_shape = {
                2,                              // two tensors-per-present dimension (maybe key/value dims)
                static_cast<int64_t>(batch_size),
                8,                              // num_heads
                0,                              // past_seq_len = 0
                64                              // head_size
            };
            Ort::Value empty_tensor = Ort::Value::CreateTensor<float>(
                memory_info, nullptr, 0, zero_shape.data(), zero_shape.size());
            past_key_values.push_back(std::move(empty_tensor));
        }
    }

    // debug past shapes
    for (size_t i = 0; i < past_key_values.size(); ++i) {
        auto shape = past_key_values[i].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "past_key_values[" << i << "] shape: [";
        for (auto dim : shape) std::cout << dim << " ";
        std::cout << "]\n";
    }

    // -------------------------
    // 4. position_ids (int32)
    //    Model expects shape [batch, seq_len] (per your Netron snippet)
    // -------------------------
    Ort::Value pos_tensor(nullptr);
    if (has_input("position_ids")) {
        // position ids for current seq. If model expects absolute positions you may use past_len + i
        std::vector<int32_t> position_ids(seq_len);
        for (size_t i = 0; i < seq_len; ++i) {
            int64_t pos = past_len + static_cast<int64_t>(i);
            // optional clamp if model has MAX_POS
            constexpr int64_t MAX_POS = 8192;
            if (pos >= MAX_POS) pos = MAX_POS - 1;
            position_ids[i] = static_cast<int32_t>(pos);
        }

        pos_tensor = Ort::Value::CreateTensor<int32_t>(
            memory_info, position_ids.data(), position_ids.size(),
            input_shape.data(), input_shape.size());

        std::cout << "position_ids: [";
        for (auto p : position_ids) std::cout << p << " ";
        std::cout << "]\n";
    }

    // -------------------------
    // 3. attention_mask (int32)
    //    Model expects shape [batch, total_seq_len] where total_seq_len = past_len + seq_len
    // -------------------------
    Ort::Value mask_tensor(nullptr);
    if (has_input("attention_mask")) {
        int64_t total_len = past_len + static_cast<int64_t>(seq_len);
        std::vector<int32_t> attention_mask(total_len, 1);
        std::vector<int64_t> mask_shape = {static_cast<int64_t>(batch_size), total_len};
        mask_tensor = Ort::Value::CreateTensor<int32_t>(
            memory_info, attention_mask.data(), attention_mask.size(),
            mask_shape.data(), mask_shape.size());

        std::cout << "attention_mask length = " << attention_mask.size()
                  << " (past_len=" << past_len << ", seq_len=" << seq_len << ")\n";
    }

    // -------------------------
    // 5. build input names & run_inputs in Netron order:
    //    input_ids, past_0..past_7, attention_mask, position_ids
    // -------------------------
    std::vector<const char*> input_names;
    std::vector<Ort::Value> run_inputs;

    // 1) input_ids first
    input_names.push_back("input_ids");
    run_inputs.push_back(std::move(input_ids_tensor));

    // 4) position_ids
    if (has_input("position_ids")) {
        input_names.push_back("position_ids");
        run_inputs.push_back(std::move(pos_tensor));
    }

    // 3) attention_mask
    if (has_input("attention_mask")) {
        input_names.push_back("attention_mask");
        run_inputs.push_back(std::move(mask_tensor));
    }

    // 2) all past_key_values in exact order (the names in past_input_names should match model)
    // Ensure past_input_names ordering matches model's past_0..past_7
    for (size_t i = 0; i < past_input_names.size(); ++i) {
        input_names.push_back(past_input_names[i]);
        // move the stored past tensor into run inputs
        run_inputs.push_back(std::move(past_key_values[i]));
    }

    // -------------------------
    // 6. debug print: input names, types, shapes
    // -------------------------
    for (size_t i = 0; i < run_inputs.size(); ++i) {
        auto info = run_inputs[i].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        std::cout << (i < input_names.size() && input_names[i] ? input_names[i] : "<noname>")
                  << ": type=" << info.GetElementType() << " shape=[";
        for (auto d : shape) std::cout << d << " ";
        std::cout << "]\n";
    }

    // -------------------------
    // 7. build output names
    //    first output logits, then present_0..present_7
    // -------------------------
    std::vector<const char*> output_names;
    output_names.push_back(logits_output_name);
    output_names.insert(output_names.end(), past_output_names.begin(), past_output_names.end());

    // -------------------------
    // 8. run session
    // -------------------------
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(), run_inputs.data(), run_inputs.size(),
        output_names.data(), output_names.size());

    // -------------------------
    // 9. update past_key_values (move outputs[1..] into cache)
    //    outputs[0] = logits, outputs[1..] = present_0..present_7
    // -------------------------
    past_key_values.clear();
    for (size_t i = 1; i < outputs.size(); ++i) {
        past_key_values.push_back(std::move(outputs[i]));
    }

    std::vector<float> logits;
    {
        Ort::Value& logits_tensor = outputs[0]; // [1, seq_len, vocab_size] or [1, vocab_size] for last token
        float* logits_ptr = logits_tensor.GetTensorMutableData<float>();

        // Get total number of elements
        Ort::TensorTypeAndShapeInfo shape_info = logits_tensor.GetTensorTypeAndShapeInfo();
        size_t num_elements = shape_info.GetElementCount();

        logits.assign(logits_ptr, logits_ptr + num_elements);
    }

    return logits;
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

    int64_t seq_len = static_cast<int64_t>(input_ids.size());
    auto device = torch::kCPU; // or torch::kCUDA

    // Build tensors (clone to own memory)
    torch::Tensor input_ids_tensor = torch::from_blob(
        const_cast<int64_t*>(input_ids.data()), {1, seq_len}, torch::kInt64).clone().to(device);
    torch::Tensor attention_mask_tensor = torch::ones({1, seq_len}, torch::kInt64).to(device);

    // Prepare inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_ids_tensor);
    inputs.push_back(attention_mask_tensor);

    // Call model
    torch::IValue output_iv = model.forward(inputs);
    auto output_tuple = output_iv.toTuple();

    // Extract logits: shape [1, seq_len, vocab_size]
    torch::Tensor logits_tensor = output_tuple->elements()[0].toTensor();

    // ✅ Select only last token logits: shape [vocab_size]
    torch::Tensor last_logits = logits_tensor.index({0, seq_len - 1});

    // Copy to std::vector<float>
    std::vector<float> logits(last_logits.numel());
    std::memcpy(
        logits.data(),
        last_logits.data_ptr<float>(),
        last_logits.numel() * sizeof(float)
    );

    return logits;
}



std::vector<float> CausalLMTorchCached::forward(const std::vector<int64_t>& input_ids_vec) {
    torch::NoGradGuard no_grad;
    auto device = torch::kCPU; // change to torch::kCUDA if needed

    const int num_layers = 8;
    const int n_head = 8;
    const int head_dim = 64;
    const int batch_size = 1;

    int64_t seq_len = static_cast<int64_t>(input_ids_vec.size());

    // 1) Build input_ids tensor
    torch::Tensor input_ids = torch::from_blob(
        const_cast<int64_t*>(input_ids_vec.data()), {batch_size, seq_len}, torch::kInt64
    ).clone().to(device);

    // 2) Determine past length
    int64_t past_len = 0;
    if (!past_key_values.empty()) {
        auto tup0 = past_key_values[0].toTuple();
        torch::Tensor key0 = tup0->elements()[0].toTensor();
        past_len = key0.size(2);
    }

    // 3) Build attention_mask
    torch::Tensor attention_mask = torch::ones({batch_size, past_len + seq_len}, torch::kInt64).to(device);

    // 4) Initialize past_key_values if empty
    if (past_key_values.empty()) {
        past_key_values.reserve(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            torch::Tensor key = torch::empty({batch_size, n_head, 0, head_dim}, torch::kFloat32).to(device);
            torch::Tensor value = torch::empty({batch_size, n_head, 0, head_dim}, torch::kFloat32).to(device);
            auto layer_tuple = c10::ivalue::Tuple::create({ key, value });
            past_key_values.push_back(layer_tuple);
        }
    }

    // 5) Build outer tuple of past_key_values
    c10::IValue past_ivalue = c10::ivalue::Tuple::create(past_key_values);

    // 6) Call forward
    std::vector<c10::IValue> inputs{ input_ids, attention_mask, past_ivalue };
    auto out_iv = model.forward(inputs).toTuple();

    // 7) Extract logits (shape [1, seq_len, vocab_size])
    torch::Tensor logits_tensor = out_iv->elements()[0].toTensor();

    // ✅ Select only last token logits [vocab_size]
    torch::Tensor last_logits = logits_tensor.index({0, seq_len - 1});

    // Copy to std::vector<float>
    std::vector<float> logits(last_logits.numel());
    std::memcpy(
        logits.data(),
        last_logits.data_ptr<float>(),
        last_logits.numel() * sizeof(float)
    );

    // 8) Update past_key_values with new present
    auto present_tuple = out_iv->elements()[1].toTuple();
    past_key_values.clear();
    for (auto &layer_iv : present_tuple->elements()) {
        past_key_values.push_back(layer_iv);
    }

    return logits;
}




}