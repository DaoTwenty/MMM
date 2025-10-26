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
    mmm::utils::Logger& logger
) {
    logger.log(mmm::utils::LogLevel::DEBUG, "[Generate] Entered generate method.");

    if (model->is_cached()) {
        logger.log(mmm::utils::LogLevel::DEBUG, "[Generate] Resetting model cache.");
        model->reset_cache();
    }

    std::vector<int64_t> current_input = input_ids;
    std::optional<int64_t> next_token;

    logger.log(mmm::utils::LogLevel::DEBUG,
               "[Generate] Generating max_new_tokens=" + std::to_string(config_.max_new_tokens));

    bool stop_generating = false;
    int step = 0;

    while (!stop_generating) {
        if (profiler_) profiler_->start();

        std::vector<float> logits;
        if (model->is_cached() && next_token.has_value()) {
            logits = model->forward({*next_token});
        } else {
            logits = model->forward(current_input);
        }

        processors_.process(logits, current_input, logger);
        warpers_.warp(logits);

        next_token = config_.do_sample
            ? sampler_.sample(logits)
            : sampler_.argmax(logits);

        if (*next_token >= vocab_size_) {
            logger.log(mmm::utils::LogLevel::WARN,
                       "[Generate] Sampled over vocab size (" + std::to_string(*next_token) + ")");
            *next_token = vocab_size_ - 1;
        }

        logger.log(mmm::utils::LogLevel::TRACE,
                   "[Generate] Step " + std::to_string(step) +
                   " sampled token=" + std::to_string(*next_token));

        if (next_token.has_value() && static_cast<int>(*next_token) == eos_token_id_) {
            logger.log(mmm::utils::LogLevel::DEBUG, "[Generate] Sampled EOS token, stopping generation.");
            stop_generating = true;
        }

        current_input.push_back(*next_token);

        if (profiler_) profiler_->stop();
        step++;

        if (!stop_generating && step >= config_.max_new_tokens) {
            logger.log(mmm::utils::LogLevel::DEBUG,
                       "[Generate] Generated max number of tokens (" +
                       std::to_string(config_.max_new_tokens) + "), stopping generation.");
            stop_generating = true;
        }
    }

    logger.log(mmm::utils::LogLevel::DEBUG,
               "[Generate] Finished generation. Total tokens: " + std::to_string(current_input.size()));

    return current_input;
}


}

}
