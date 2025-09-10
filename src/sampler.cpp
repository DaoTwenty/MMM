#include "sampler.h"

namespace mmm {

namespace sampling {

Sampler::Sampler(std::mt19937::result_type seed)
        : rng_(seed) {}

int64_t Sampler::sample(const std::vector<float>& logits) {
    // Convert to probabilities (softmax)
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
    return dist(rng_);
}

int64_t Sampler::argmax(const std::vector<float>& logits) {
    return std::distance(logits.begin(),
                            std::max_element(logits.begin(), logits.end()));
}

}

}
