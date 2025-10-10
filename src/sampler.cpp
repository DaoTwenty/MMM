#include "sampler.h"

namespace mmm {

namespace sampling {

Sampler::Sampler(int seed)
        : seed_(seed), rng_(static_cast<std::mt19937::result_type>(seed)) {}

int64_t Sampler::sample(const std::vector<float>& logits) {
    bool verbose_ = false;

    if (verbose_) {  // assuming your sampler has a verbose flag; otherwise remove this guard
        std::cout << "[Sampler] Sampling from logits\n";
        std::cout << "  Logits size = " << logits.size() << "\n";
    }

    if (logits.empty()) {
        throw std::runtime_error("[Sampler] Logits vector is empty!");
    }

    // Convert to probabilities (softmax)
    float max_logit = *std::max_element(logits.begin(), logits.end());
    if (verbose_) {
        float max_val = *std::max_element(logits.begin(), logits.end());
        float min_val = *std::min_element(logits.begin(), logits.end());
        double sum = 0.0;
        int nan_count = 0, inf_count = 0;
        for (float v : logits) {
            sum += v;
            if (!std::isfinite(v)) {
                if (std::isnan(v)) nan_count++;
                else inf_count++;
            }
        }
        float mean = static_cast<float>(sum / logits.size());

        std::cout << "[LogitsAnalysis] size=" << logits.size()
                << ", max=" << max_val
                << ", min=" << min_val
                << ", mean=" << mean
                << ", nans=" << nan_count
                << ", infs=" << inf_count
                << "\n";
    }

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

    if (verbose_) std::cout << "  Sum before normalization = " << sum << "\n";

    if (sum == 0.0f || !std::isfinite(sum)) {
        throw std::runtime_error("[Sampler] Invalid softmax sum (zero or non-finite)");
    }

    for (auto& p : probs) p /= sum;

    // Optional: print a few top probabilities
    if (verbose_ && logits.size() <= 20) {
        std::cout << "  Normalized probs:\n";
        for (size_t i = 0; i < probs.size(); ++i) {
            std::cout << "    [" << i << "] = " << probs[i] << "\n";
        }
    }

    // Build distribution
    std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
    int64_t sampled = dist(rng_);

    if (verbose_) std::cout << "  Sampled token = " << sampled << "\n";

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
