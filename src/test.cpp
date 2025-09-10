#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <string>
#include <nlohmann/json.hpp>

#include "model.h"            
#include "engine.h" 
#include "sampler.h" 
#include "logitsprocessor.h" 
#include "logitswarper.h"
#include "profiler.h"       
#include "config.h"  

// LibTok
#include "mmm.h"

// Helper to compute mean/std
struct Stats {
    double mean;
    double min;
    double max;
    double stddev;
};

Stats compute_stats(const std::vector<double>& times) {
    if (times.empty()) return {0,0,0,0};

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    double min = *std::min_element(times.begin(), times.end());
    double max = *std::max_element(times.begin(), times.end());

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / times.size() - mean * mean);

    return {mean, min, max, stddev};
}

// Simple console progress bar
void print_progress(int current, int total, int bar_width = 40) {
    float progress = float(current) / float(total);
    int pos = int(bar_width * progress);

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

struct BenchmarkModel {
    std::string name;
    enum class Type {ONNX, TORCHSCRIPT} type;
    std::string path;
    bool cached;
    bool coreml;
    mmm::CausalLM* onnx_model = nullptr;
    mmm::CausalLMTorch* torch_model = nullptr;
    std::vector<double> times;
};

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cout << "Usage: " << argv[0] << " <config.json> <tokenizer.json> ( optional <file.mid> )\n";
        return 1;
    }

    // Load JSON config
    std::ifstream f(argv[1]);
    if (!f.is_open()) {
        std::cerr << "Error opening config file: " << argv[1] << "\n";
        return 1;
    }
    nlohmann::json j;
    f >> j;

    // Extract parameters
    int num_passes  = j.value("num_passes", 5);
    int seq_len     = j.value("seq_len", 32);
    int num_gen     = j.value("num_gen", 10);
    int vocab_size  = j.value("vocab_size", 16000);

    std::vector<int64_t> input_ids;
    std::string midi_file = "None/Random Input";
    if (argc == 4) { 
        std::cout << "Using MIDI file :: " << argv[3] << std::endl;
        std::string midiStr = argv[3];
        midi_file = midiStr;
        std::string tokenizerStr = argv[2];
        std::filesystem::path midiPath(midiStr);
        std::filesystem::path tokenizerPath(tokenizerStr);
        LibTok::MMM tokenizer = LibTok::MMM(tokenizerPath);
        auto tokens = tokenizer.encode(midiPath);
        LibTok::TokSequence tokSeq = std::get<LibTok::TokSequence>(tokens);
        std::vector<std::string> tokenVec = tokSeq.tokens;
        std::vector<int> token_ids = tokSeq.ids;
        input_ids.reserve(token_ids.size());

        for (int id : token_ids)
            input_ids.push_back(static_cast<int64_t>(id));
    } else {
        std::cout << "Using random inputs :: sequence size " << seq_len << std::endl;
        // Prepare random input ids
        std::vector<int64_t> input_ids(seq_len);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> dist(0, vocab_size - 1);
        for (auto &id : input_ids) id = dist(gen);
    }

    // Parse and load models
    std::vector<BenchmarkModel> models;
    for (auto& jm : j["models"]) {
        BenchmarkModel m;
        m.name   = jm.value("name", "unnamed");
        m.path   = jm.value("path", "");
        m.cached = jm.value("cache", false);
        m.coreml = jm.value("coreml", false);
        std::string type = jm.value("type", "ONNX");

        m.type = (type == "ONNX") ? BenchmarkModel::Type::ONNX
                                  : BenchmarkModel::Type::TORCHSCRIPT;

        if (m.type == BenchmarkModel::Type::ONNX) {
            m.onnx_model = m.cached ? new mmm::CausalLMCached(m.path, vocab_size, m.coreml)
                                    : new mmm::CausalLM(m.path, vocab_size, m.coreml);
        } else {
            m.torch_model = m.cached ? new mmm::CausalLMTorchCached(m.path, vocab_size)
                                     : new mmm::CausalLMTorch(m.path, vocab_size);
        }
        models.push_back(std::move(m));
    }

    seq_len = input_ids.size();
    int n_models   = models.size();
    int tot_passes = n_models * num_passes;
    int cum_pass   = 0;

    std::cout << "Benchmarking " << n_models
              << " models :: " << num_passes
              << " passes :: input data - " << midi_file 
              << " :: " << seq_len
              << " initial sequence length :: " << num_gen
              << " generated tokens\n";

    // Prepare SamplingEngine (model-agnostic)
    mmm::sampling::GenerationConfig config;
    config.max_new_tokens = num_gen;

    mmm::sampling::Sampler sampler;

    // Prepare LogitsProcessorList
    mmm::sampling::LogitsProcessorList processors;
    // Example: you could push processors here (temperature, top-k, etc.)
    // processors.add(std::make_shared<TemperatureLogitsProcessor>(config.temperature));
    mmm::sampling::LogitsWarperList warpers;

    mmm::utils::Profiler profiler;

    mmm::sampling::SamplingEngine engine(
        config, 
        processors, 
        warpers, 
        sampler, 
        &profiler
    );

    // Benchmark loop
    for (int pass = 0; pass < num_passes; ++pass) {
        for (auto& m : models) {
            cum_pass++;
            print_progress(cum_pass + 1, tot_passes);

            mmm::IModel* model = (m.type == BenchmarkModel::Type::ONNX)
                        ? static_cast<mmm::IModel*>(m.onnx_model)
                        : static_cast<mmm::IModel*>(m.torch_model);

            if (model->is_cached()) {
                model->reset_cache();
            }

            profiler.reset();

            // Use sampling engine for generation
            std::vector<int64_t> generated =
                engine.generate(input_ids, model);

            m.times.push_back(profiler.total_time());

        }
    }

    std::cout << "\n=== Benchmark Summary over " << num_passes << " passes ===\n";
    for (auto& m : models) {
        Stats s = compute_stats(m.times);
        std::cout << m.name << " (" << m.path << ")"
                  << " :: mean = " << s.mean
                  << " ms, min = " << s.min
                  << " ms, max = " << s.max
                  << " ms, stddev = " << s.stddev << " ms\n";
    }

    // Compute speedups relative to first model
    if (!models.empty()) {
        double ref_mean = compute_stats(models[0].times).mean;
        std::cout << "\nSpeedup relative to " << models[0].name << ":\n";
        for (auto& m : models) {
            double mean = compute_stats(m.times).mean;
            double speedup = ref_mean / mean;
            std::cout << m.name << " : " << speedup << "x\n";
        }
    }

    // Cleanup
    for (auto& m : models) {
        delete m.onnx_model;
        delete m.torch_model;
    }

    return 0;
}