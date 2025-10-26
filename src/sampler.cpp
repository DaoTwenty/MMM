#include "sampler.h"

namespace mmm {

namespace sampling {

Sampler::Sampler(int seed)
        : seed_(seed), rng_(static_cast<std::mt19937::result_type>(seed)) {}

int64_t Sampler::sample(const std::vector<float>& logits) {

    if (logits.empty()) {
        throw std::runtime_error("[Sampler] Logits vector is empty!");
    }

    // Convert to probabilities (softmax)
    float max_logit = *std::max_element(logits.begin(), logits.end());

    std::vector<float> probs(logits.size());
    float sum = 0.0f;

    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        if (!std::isfinite(probs[i])) {
            std::cout << "  [Warn] Non-finite prob at index " << i 
                      << " (logit=" << logits[i] << ")\n";
        }
        sum += probs[i];
    }


    if (sum == 0.0f || !std::isfinite(sum)) {
        throw std::runtime_error("[Sampler] Invalid softmax sum (zero or non-finite)");
    }

    for (auto& p : probs) p /= sum;

    // Build distribution
    std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
    int64_t sampled = dist(rng_);


    // Sanity check
    if (sampled < 0 || sampled >= static_cast<int64_t>(logits.size())) {
        std::ostringstream oss;
        oss << "[Sampler] Out-of-range sample: " << sampled 
            << " (logits size=" << logits.size() << ")";
        throw std::runtime_error(oss.str());
    }

    return sampled;
}


int64_t Sampler::argmax(const std::vector<float>& logits) {
    return std::distance(logits.begin(),
                            std::max_element(logits.begin(), logits.end()));
}

}

}
