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
#include "generationconfig.h"  
#include "inference.h"
#include "utils.h"

// LibTok
#include "mmm.h"
#include "utility_functions.h"

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
    if (argc != 7) {
        std::cout << "Usage: " << argv[0] << " <config.json> <tokenizer.json> <gen_config.json> <prompt.json> <file.mid> <verbose>\n";
        return 1;
    }

    std::string verbose_arg = argv[6];
    std::transform(verbose_arg.begin(), verbose_arg.end(), verbose_arg.begin(), ::tolower); // case-insensitive
    bool verbose = (verbose_arg == "true" || verbose_arg == "1" || verbose_arg == "yes");

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
    int vocab_size  = j.value("vocab_size", 16000);

    mmm::sampling::GenerationConfig gen_cfg;
    mmm::inference::PromptConfig prompt_cfg;

    std::string gen_cfg_path = argv[3];
    std::string prompt_cfg_path = argv[4];

    try {
        mmm::utils::loadGenerationConfigFromJson(gen_cfg_path, gen_cfg);
    } catch (std::exception& e) {
        std::cerr << "Error loading generation config: " << e.what() << "\n";
        return 1;
    }

    try {
        mmm::utils::loadPromptConfigFromJson(prompt_cfg_path, prompt_cfg);
    } catch (std::exception& e) {
        std::cerr << "Error loading prompt: " << e.what() << "\n";
        return 1;
    }

        std::string prompt_mode;
    if (prompt_cfg.bar_infilling()) {
        prompt_mode = "infill";
    } else {
        prompt_mode = "sample";
    }

    std::string midiStr = argv[5];
    std::string tokenizerStr = argv[2];
    std::filesystem::path midiPath(midiStr);
    std::string fname = midiPath.stem().string();
    std::filesystem::path folder = midiPath.parent_path();
    std::filesystem::path tokenizerPath(tokenizerStr);
    std::unique_ptr<LibTok::MMM> tokenizer;
    //tokenizer->tokenizerConfig.saveToJson("configs/tokenizer_export.json");

    try {
        tokenizer = std::make_unique<LibTok::MMM>(tokenizerPath, false);
    } catch (std::exception& e) {
        std::cerr << "Error loading tokenizer: " << e.what() << "\n";
        return 1;
    }

    LibTok::ScoreType score;
    try {
        score = LibTokUtils::loadScoreFromMidi(midiPath);
    } catch (std::exception& e) {
        std::cerr << "Error loading MIDI file: " << e.what() << "\n";
        return 1;
    }

    mmm::sampling::SamplingEngine engine = mmm::inference::createEngine(gen_cfg, *tokenizer, -1, verbose);

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

    int n_models   = models.size();
    int tot_passes = n_models * num_passes;
    int cum_pass   = 0;

    std::cout << "Benchmarking " << n_models
              << " models :: " << num_passes
              << " passes :: input data - " << midiStr 
              << " :: vocab size " << tokenizer->getVocabSize()
              << " :: max new tokens " << gen_cfg.max_new_tokens
              << "\n";

    // Benchmark loop
    print_progress(cum_pass, tot_passes);
    for (int pass = 0; pass < num_passes; ++pass) {
        for (auto& m : models) {
            cum_pass++;
            print_progress(cum_pass, tot_passes);

            mmm::IModel* model = (m.type == BenchmarkModel::Type::ONNX)
                        ? static_cast<mmm::IModel*>(m.onnx_model)
                        : static_cast<mmm::IModel*>(m.torch_model);

            if (model->is_cached()) {
                model->reset_cache();
            }

            engine.resetProfiler();

            try {
                LibTok::ScoreType gen_score = mmm::inference::generate(
                    model,
                    *tokenizer,
                    prompt_cfg,
                    engine,
                    score,
                    verbose
                );

                m.times.push_back(engine.totalTimeProfiler());

                std::filesystem::path output_folder = folder / m.name;
                std::filesystem::create_directories(output_folder);
                std::filesystem::path output_file = output_folder / (fname + "_mode_" + prompt_mode + "_gen_" + std::to_string(pass) + ".mid");
                LibTokUtils::saveMidiFromScore(gen_score, output_file);
            } catch (const std::exception& e) {
                std::cerr << "Unable to generate. Error: " << e.what() << ". Passing...\n";
            }

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