#pragma once

#include <chrono>
#include <string>
#include <iostream>

namespace mmm {

namespace utils {

class Profiler {
public:
    void start(const std::string& label) {
        current_label_ = label;
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
        total_time_ += duration;
        //std::cout << "[Profiler] " << current_label_ << " took " 
        //          << duration << " ms\n";
    }

    void reset() { total_time_ = 0.0; }
    double total_time() const { return total_time_; }

private:
    std::string current_label_;
    std::chrono::high_resolution_clock::time_point start_time_;
    double total_time_ = 0.0;
};

}

}
