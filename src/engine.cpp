#include "engine.h"

namespace mmm {

namespace sampling {

SamplingEngine::SamplingEngine(const GenerationConfig& config,
                LogitsProcessorList processors,
                LogitsWarperList warpers,
                Sampler sampler,
                mmm::utils::Profiler* profiler)
    : config_(config),
        processors_(std::move(processors)),
        warpers_(std::move(warpers)),
        sampler_(sampler),
        profiler_(profiler) {}


std::vector<int64_t> SamplingEngine::generate(
    const std::vector<int64_t>& input_ids,
    mmm::IModel* model
) {

    if (model->is_cached()) {
        model->reset_cache();
    }
    std::vector<int64_t> current_input;
    current_input = input_ids;
    std::optional<int64_t> next_token;
    for (int step = 0; step < config_.max_new_tokens; ++step) {
        if (profiler_) profiler_->start("model.forward");
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

        current_input.push_back(*next_token);

        if (profiler_) profiler_->stop();
    }

    return current_input;
}

}

}
