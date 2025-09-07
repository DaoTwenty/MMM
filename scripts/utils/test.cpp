#include "model.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <string>

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
    mmm::CausalLM* onnx_model = nullptr;
    mmm::CausalLMTorch* torch_model = nullptr;
    std::vector<double> times;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_path1> [<model_path2> ...] [--torch <torchscript_path>]\n";
        return 1;
    }

    int num_passes = 100;
    std::vector<int64_t> input_ids = {1, 123, 456, 123, 432, 23, 4534, 4}; // example tokens

    std::vector<BenchmarkModel> models;

    // Parse arguments: any path is an ONNX model until optional --torch
    bool torch_next = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--torch") {
            torch_next = true;
            continue;
        }
        if (torch_next) {
            models.push_back({ "TORCHSCRIPT", BenchmarkModel::Type::TORCHSCRIPT, arg });
            torch_next = false;
        } else {
            models.push_back({ "ONNX", BenchmarkModel::Type::ONNX, arg });
        }
    }

    // Load models
    for (auto& m : models) {
        if (m.type == BenchmarkModel::Type::ONNX) {
            m.onnx_model = new mmm::CausalLM(m.path, 16000, 0);
        } else {
            m.torch_model = new mmm::CausalLMTorch(m.path, 16000, 0);
        }
    }

    // Benchmark loop
    for (int pass = 0; pass < num_passes; ++pass) {
        print_progress(pass + 1, num_passes);
        for (auto& m : models) {
            auto t_start = std::chrono::high_resolution_clock::now();

            if (m.type == BenchmarkModel::Type::ONNX) {
                auto logits = m.onnx_model->forward(input_ids);
            } else {
                auto logits = m.torch_model->forward(input_ids);
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            m.times.push_back(elapsed_ms);
        }
    }

    std::cout << "\n=== Benchmark Summary over " << num_passes << " passes ===\n";
    for (auto& m : models) {
        Stats s = compute_stats(m.times);
        std::cout << m.path << " :: mean = " << s.mean
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
}
