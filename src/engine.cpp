#include "engine.h"

namespace mmm {

namespace sampling {

SamplingEngine::SamplingEngine(const GenerationConfig& config,
                int eos_token_id,
                int vocab_size,
                LogitsProcessorList processors,
                LogitsWarperList warpers,
                Sampler sampler,
                bool profiler)
    : config_(config),
        eos_token_id_(eos_token_id),
        vocab_size_(vocab_size),
        processors_(std::move(processors)),
        warpers_(std::move(warpers)),
        sampler_(sampler) {
            if (profiler) {
                profiler_ = new mmm::utils::Profiler();
            } else {
                profiler_ = nullptr;
            }
        }

void SamplingEngine::updateProcessors(const std::string& key, int value) {
    processors_.update(key, value);
}

std::vector<int64_t> SamplingEngine::generate(
    const std::vector<int64_t>& input_ids,
    mmm::IModel* model,
    bool verbose
) {

    if (verbose) std::cout << "[Generate] Entered generate method.\n";

    if (model->is_cached()) {
        model->reset_cache();
    }
    std::vector<int64_t> current_input;
    current_input = input_ids;
    std::optional<int64_t> next_token;

    if (verbose) std::cout << "[Generate] Generating max_new_tokens=" << config_.max_new_tokens << "\n";
    bool stop_generating = false;
    int step = 0;
    //for (int step = 0; step < config_.max_new_tokens; step++) {
    while (!stop_generating) {

        //if (verbose) std::cout << "[Generate] Step=" << step << "\n";
        
        if (profiler_) profiler_->start();
        std::vector<float> logits;
        if (model->is_cached() && next_token.has_value()) {
            logits = model->forward({*next_token});
        } else {
            logits = model->forward(current_input);
        }

        processors_.process(logits, current_input);
        warpers_.warp(logits);

        next_token = config_.do_sample
            ? sampler_.sample(logits)
            : sampler_.argmax(logits);
        //if (verbose) std::cout << "[Generate] Sampled next token :: " << std::to_string(std::min(*next_token, static_cast<int64_t>(vocab_size_ - 1))) << "\n";

        if (*next_token >= vocab_size_) {
            *next_token = vocab_size_ - 1;
            if (verbose) std::cerr << "[Generate] Sampled over vocab size (sampled " << *next_token << ")\n";
            //throw std::runtime_error("[Generate] Sampled over vocab size (sampled " + std::to_string(*next_token) + ")");
        }

        if (next_token.has_value() && static_cast<int>(*next_token) == eos_token_id_) {
            if (verbose) std::cout << "[Generate] Sampled EOS token, stopping generation.\n";
            stop_generating = true;
        }
        current_input.push_back(*next_token);

        if (profiler_) profiler_->stop();
        step++;
        if (!stop_generating && step >= config_.max_new_tokens) {
            if (verbose) std::cout << "[Generate] Generated max number of tokens (" << config_.max_new_tokens << " tokens), stopping generation.\n"; 
            stop_generating = true;
        }
    }

    return current_input;
}

}

}
