#pragma once

#include <optional>
#include <string>
#include <iostream>

namespace mmm {

namespace sampling {

struct GenerationConfig;

inline std::ostream& operator<<(std::ostream& os, const GenerationConfig& cfg);

struct GenerationConfig {
    bool do_sample = true;    
    int max_new_tokens = 256;
    int attempts = 4;
    int pad_token_id = 0;
    float repetition_penalty = 1.0f;
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 1.0f;

    void print() const {
        std::cout << *this;  // use operator<<
    }
};

// free function, not inside the struct
inline std::ostream& operator<<(std::ostream& os, const GenerationConfig& cfg) {
    os << "GenerationConfig:\n"
       << "  do_sample=" << cfg.do_sample << "\n"
       << "  max_new_tokens=" << cfg.max_new_tokens << "\n"
       << "  attempts=" << cfg.attempts << "\n"
       << "  pad_token_id=" << cfg.pad_token_id << "\n"
       << "  repetition_penalty=" << cfg.repetition_penalty << "\n"
       << "  temperature=" << cfg.temperature << "\n"
       << "  top_k=" << cfg.top_k << "\n"
       << "  top_p=" << cfg.top_p << "\n";
    return os;
}

}

}
