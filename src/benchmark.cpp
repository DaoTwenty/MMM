#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <string>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

// === Project Includes ===
#include "model.h"
#include "engine.h"
#include "sampler.h"
#include "logitsprocessor.h"
#include "logitswarper.h"
#include "profiler.h"
#include "generationconfig.h"
#include "inference.h"
#include "utils.h"
#include "logger.h"

// === LibTok ===
#include "mmm.h"
#include "utility_functions.h"

// ================================================================
// Helper: Compute stats
// ================================================================
struct Stats {
    double mean;
    double min;
    double max;
    double stddev;
};

static Stats compute_stats(const std::vector<double>& times) {
    if (times.empty()) return {0,0,0,0};

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    double min = *std::min_element(times.begin(), times.end());
    double max = *std::max_element(times.begin(), times.end());
    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / times.size() - mean * mean);

    return {mean, min, max, stddev};
}

// ================================================================
// Helper: Simple console progress bar
// ================================================================
static void print_progress(int current, int total, int bar_width = 40) {
    float progress = float(current) / float(total);
    int pos = int(bar_width * progress);

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i)
        std::cout << (i < pos ? "=" : (i == pos ? ">" : " "));
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

// ================================================================
// Model benchmark entry
// ================================================================
struct BenchmarkModel {
    std::string name;
    std::string path;
    bool cached = false;
#ifdef USE_ONNX
    bool coreml = false;
#endif 
    std::vector<double> times;
    std::unique_ptr<mmm::IModel> model;
};

// ================================================================
// Main
// ================================================================
int main(int argc, char** argv) {
    if (argc != 7) {
        std::cout << "Usage: " << argv[0]
                  << " <config.json> <tokenizer.json> <gen_config.json> <prompt.json> <file.mid> <verbose>\n";
        return 1;
    }

    // Parse verbosity flag
    std::string verbose_arg = argv[6];
    std::transform(verbose_arg.begin(), verbose_arg.end(), verbose_arg.begin(), ::tolower);
    bool verbose = (verbose_arg == "true" || verbose_arg == "1" || verbose_arg == "yes");

    // Load JSON config
    nlohmann::json j;
    {
        std::ifstream f(argv[1]);
        if (!f.is_open()) {
            std::cerr << "Error opening config file: " << argv[1] << "\n";
            return 1;
        }
        f >> j;
    }

    int num_passes = j.value("num_passes", 5);
    int vocab_size = j.value("vocab_size", 16000);

    // Load generation and prompt configs
    mmm::sampling::GenerationConfig gen_cfg;
    mmm::inference::PromptConfig prompt_cfg;
    try {
        mmm::utils::loadGenerationConfigFromJson(argv[3], gen_cfg);
        mmm::utils::loadPromptConfigFromJson(argv[4], prompt_cfg);
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << "\n";
        return 1;
    }

    std::string prompt_mode = prompt_cfg.bar_infilling() ? "infill" : "sample";

    // Paths
    std::filesystem::path midi_path(argv[5]);
    std::filesystem::path folder = midi_path.parent_path();
    std::filesystem::path tokenizer_path(argv[2]);
    std::string fname = midi_path.stem().string();

    // Load tokenizer
    std::unique_ptr<LibTok::MMM> tokenizer;
    try {
        tokenizer = std::make_unique<LibTok::MMM>(tokenizer_path, false);
    } catch (const std::exception& e) {
        std::cerr << "Error loading tokenizer: " << e.what() << "\n";
        return 1;
    }

    // Load MIDI
    LibTok::ScoreType score;
    try {
        score = LibTokUtils::loadScoreFromMidi(midi_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading MIDI: " << e.what() << "\n";
        return 1;
    }

    Logger logger = Logger(LogLevel::ERROR);
    if (verbose) {
        logger.setMaxLevel(LogLevel::DEBUG)
    }

    // Sampling engine
    mmm::sampling::SamplingEngine engine = mmm::inference::createEngine(gen_cfg, *tokenizer, logger, -1);

    // ================================================================
    // Load models
    // ================================================================
    std::vector<BenchmarkModel> models;
    for (auto& jm : j["models"]) {
        BenchmarkModel m;
        m.name   = jm.value("name", "unnamed");
        m.path   = jm.value("path", "");
        m.cached = jm.value("cache", false);
        m.coreml = jm.value("coreml", false);

    #ifdef USE_ONNX
        if (m.cached)
            m.model = std::make_unique<mmm::CausalLMCached>(m.path, vocab_size, m.coreml);
        else
            m.model = std::make_unique<mmm::CausalLM>(m.path, vocab_size, m.coreml);
    #elif defined(USE_TORCH)
        if (m.cached)
            m.model = std::make_unique<mmm::CausalLMCached>(m.path, vocab_size);
        else
            m.model = std::make_unique<mmm::CausalLM>(m.path, vocab_size);
    #else
        #error "Define either USE_ONNX or USE_TORCH at compile time."
    #endif

        models.push_back(std::move(m));
    }

    // ================================================================
    // Benchmark loop
    // ================================================================
    int tot_passes = static_cast<int>(models.size()) * num_passes;
    int cum_pass   = 0;

    std::cout << "Benchmarking " << models.size()
              << " models :: " << num_passes
              << " passes :: input " << midi_path
              << " :: vocab " << tokenizer->getVocabSize()
              << " :: max tokens " << gen_cfg.max_new_tokens
              << "\n";

    print_progress(cum_pass, tot_passes);

    for (int pass = 0; pass < num_passes; ++pass) {
        for (auto& m : models) {
            cum_pass++;
            print_progress(cum_pass, tot_passes);

            mmm::IModel* model = m.model.get();
            if (model->is_cached()) model->reset_cache();

            engine.resetProfiler();

            try {
                LibTok::ScoreType gen_score = mmm::inference::generate(
                    model, *tokenizer, prompt_cfg, engine, score, logger
                );
                m.times.push_back(engine.totalTimeProfiler());

                std::filesystem::path output_folder = folder / m.name;
                std::filesystem::create_directories(output_folder);
                std::filesystem::path output_file = output_folder /
                    (fname + "_mode_" + prompt_mode + "_gen_" + std::to_string(pass) + ".mid");
                LibTokUtils::saveMidiFromScore(gen_score, output_file);

            } catch (const std::exception& e) {
                std::cerr << "Generation error in " << m.name << ": " << e.what() << "\n";
            }
        }
    }

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "\n=== Benchmark Summary (" << num_passes << " passes) ===\n";
    for (auto& m : models) {
        Stats s = compute_stats(m.times);
        std::cout << m.name << " :: mean = " << s.mean
                  << " ms, min = " << s.min
                  << " ms, max = " << s.max
                  << " ms, stddev = " << s.stddev << "\n";
    }

    if (!models.empty()) {
        double ref_mean = compute_stats(models[0].times).mean;
        std::cout << "\nSpeedup relative to " << models[0].name << ":\n";
        for (auto& m : models) {
            double mean = compute_stats(m.times).mean;
            std::cout << m.name << " : " << ref_mean / mean << "x\n";
        }
    }

    return 0;
}
