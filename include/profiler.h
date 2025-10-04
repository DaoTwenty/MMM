#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <exception>

namespace mmm {
namespace utils {

class Profiler {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }

    void stop() {
        if (!running_) return; // prevent garbage
        auto end_time = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
        total_time_ += diff;
        running_ = false;
    }

    void reset() { total_time_ = 0.0; running_ = false; }
    double total_time() const { return total_time_; }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
    double total_time_ = 0.0;
    bool running_ = false;
};

} // namespace utils
} // namespace mmm
