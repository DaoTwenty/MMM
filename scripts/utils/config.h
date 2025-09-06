#include <optional>
#include <string>
#include <iostream>

struct GenerationConfig {
    bool do_sample = true;
    std::optional<float> epsilon_cutoff = std::nullopt;
    std::optional<float> eta_cutoff = std::nullopt;     
    int max_new_tokens = 256;
    int pad_token_id = 0;
    float repetition_penalty = 1.0f;
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 1.0f;

    void print() const {
        std::cout << "GenerationConfig:\n"
                  << "  do_sample=" << do_sample << "\n"
                  << "  max_new_tokens=" << max_new_tokens << "\n"
                  << "  pad_token_id=" << pad_token_id << "\n"
                  << "  repetition_penalty=" << repetition_penalty << "\n"
                  << "  temperature=" << temperature << "\n"
                  << "  top_k=" << top_k << "\n"
                  << "  top_p=" << top_p << "\n"
    }
};
