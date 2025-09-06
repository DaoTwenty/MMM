#include "model.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>

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

// In your main benchmarking code:
std::vector<double> default_times;
std::vector<double> optimized_times;
std::vector<double> torch_times;
int num_passes = 100;

int main() {
    mmm::CausalLM model("/Users/paultriana/creative_labs/models/MISTRAL_123000_ONNX/model.onnx", 16000, 0);
    mmm::CausalLM opt_model("/Users/paultriana/creative_labs/models/MISTRAL_123000_OPT_ONNX/model.onnx", 16000, 0);

    std::vector<int64_t> input_ids = {1, 123, 456, 123, 432, 23, 4534, 4}; // BOS and tokens

    for (int i = 0; i < num_passes; i++) {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto logits = model.forward(input_ids);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        default_times.push_back(elapsed_time_ms);

        auto opt_t_start = std::chrono::high_resolution_clock::now();
        auto opt_logits = opt_model.forward(input_ids);
        auto opt_t_end = std::chrono::high_resolution_clock::now();
        double opt_elapsed_time_ms = std::chrono::duration<double, std::milli>(opt_t_end - opt_t_start).count();
        optimized_times.push_back(opt_elapsed_time_ms);

        print_progress(i + 1, num_passes);
    }

    std::cout << std::endl;

    // Compute stats
    Stats default_stats = compute_stats(default_times);
    Stats optimized_stats = compute_stats(optimized_times);
    Stats torch_stats = compute_stats(torch_times);

    // Print summary
    std::cout << "\n=== Benchmark Summary over " << num_passes << " passes ===\n";
    std::cout << "DEFAULT   :: mean = " << default_stats.mean << " ms"
            << ", min = " << default_stats.min << " ms"
            << ", max = " << default_stats.max << " ms"
            << ", stddev = " << default_stats.stddev << " ms\n";

    std::cout << "OPTIMIZED :: mean = " << optimized_stats.mean << " ms"
            << ", min = " << optimized_stats.min << " ms"
            << ", max = " << optimized_stats.max << " ms"
            << ", stddev = " << optimized_stats.stddev << " ms\n";

    double speedup = default_stats.mean / optimized_stats.mean;
    std::cout << "SPEEDUP   :: " << speedup << "x faster (mean)\n";
}