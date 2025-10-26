#pragma once

#include <numeric>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

#include "generationconfig.h"

namespace mmm {

namespace sampling {

class LogitsWarper {
public:
    virtual ~LogitsWarper() = default;
    virtual void warp(std::vector<float>& logits) = 0;
};

class LogitsWarperList {
public:
    void addWarper(std::shared_ptr<LogitsWarper> warper) {
        warpers_.push_back(std::move(warper));
    }

    void warp(std::vector<float>& logits) const {
        for (const auto& w : warpers_) {
            w->warp(logits);
        }
    }

private:
    std::vector<std::shared_ptr<LogitsWarper>> warpers_;
};

// ----------------------------------------
// Temperature scaling
// ----------------------------------------
class TemperatureLogitsWarper : public LogitsWarper {
public:
    explicit TemperatureLogitsWarper(float temp) : temp_(temp) {}

    void warp(std::vector<float>& logits) override {
        if (temp_ == 1.0f) return;
        for (auto& logit : logits) logit /= temp_;
    }

private:
    float temp_;
};

// ----------------------------------------
// Top-K filtering
// ----------------------------------------
class TopKLogitsWarper : public LogitsWarper {
public:
    explicit TopKLogitsWarper(size_t k) : k_(k) {}

    void warp(std::vector<float>& logits) override {
        if (k_ == 0 || k_ >= logits.size()) return;

        // Find kth largest logit (partial sort)
        std::vector<float> temp = logits;
        std::nth_element(temp.begin(), temp.begin() + k_, temp.end(), std::greater<float>());
        float kth_value = temp[k_ - 1];

        for (auto& logit : logits) {
            if (logit < kth_value) logit = -1e10f;
        }
    }

private:
    size_t k_;
};

// ----------------------------------------
// Top-P (nucleus) filtering
// ----------------------------------------
class TopPLogitsWarper : public LogitsWarper {
public:
    explicit TopPLogitsWarper(float p) : p_(p) {}

    void warp(std::vector<float>& logits) override {
        if (p_ >= 1.0f) return;

        // Compute softmax probabilities
        std::vector<float> probs(logits.size());
        float max_logit = *std::max_element(logits.begin(), logits.end());

        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
        }
        for (float& p : probs) p /= sum;

        // Sort by probability
        std::vector<size_t> idx(logits.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return probs[a] > probs[b]; });

        // Cumulative sum & mask beyond p_
        float cumulative = 0.0f;
        for (size_t i = 0; i < idx.size(); ++i) {
            cumulative += probs[idx[i]];
            if (cumulative >= p_) {
                for (size_t j = i + 1; j < idx.size(); ++j)
                    logits[idx[j]] = -1e10f;
                break;
            }
        }
    }

private:
    float p_;
};

}

}