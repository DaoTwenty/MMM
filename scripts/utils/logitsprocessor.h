#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace mmm {

namespace sampling {

// ----------------------------------------
// Base class for all processors
// ----------------------------------------
class LogitsProcessor {
public:
    virtual ~LogitsProcessor() = default;

    // Process logits in-place, can use current generated sequence if needed
    virtual void process(std::vector<float>& logits,
                         const std::vector<int64_t>& current_tokens) = 0;
};

// ----------------------------------------
// List of processors
// ----------------------------------------
class LogitsProcessorList {
public:
    void addProcessor(std::shared_ptr<LogitsProcessor> processor) {
        processors_.push_back(std::move(processor));
    }

    void process(std::vector<float>& logits,
                 const std::vector<int64_t>& current_tokens) const {
        for (const auto& processor : processors_) {
            processor->process(logits, current_tokens);
        }
    }

private:
    std::vector<std::shared_ptr<LogitsProcessor>> processors_;
};

// ----------------------------------------
// Repetition Penalty Processor
// ----------------------------------------
class RepetitionPenaltyLogitsProcessor : public LogitsProcessor {
public:
    explicit RepetitionPenaltyLogitsProcessor(float penalty) : penalty_(penalty) {}

    void process(std::vector<float>& logits,
                 const std::vector<int64_t>& current_tokens) override {
        if (penalty_ == 1.0f) return; // no-op

        for (int64_t token : current_tokens) {
            if (token >= 0 && token < (int64_t)logits.size()) {
                if (logits[token] < 0)
                    logits[token] *= penalty_;
                else
                    logits[token] /= penalty_;
            }
        }
    }

private:
    float penalty_;
};

// ----------------------------------------
// MinLength Processor: forbid EOS until min length is reached
// ----------------------------------------
class MinLengthLogitsProcessor : public LogitsProcessor {
public:
    MinLengthLogitsProcessor(size_t min_length, int eos_token_id)
        : min_length_(min_length), eos_token_id_(eos_token_id) {}

    void process(std::vector<float>& logits,
                 const std::vector<int64_t>& current_tokens) override {
        if (current_tokens.size() < min_length_ &&
            eos_token_id_ >= 0 &&
            eos_token_id_ < (int)logits.size()) {
            logits[eos_token_id_] = -1e10f; // mask EOS
        }
    }

private:
    size_t min_length_;
    int eos_token_id_;
};

// ----------------------------------------
// Temperature Processor (scales logits)
// ----------------------------------------
class TemperatureLogitsProcessor : public LogitsProcessor {
public:
    explicit TemperatureLogitsProcessor(float temperature) : temp_(temperature) {}

    void process(std::vector<float>& logits,
                 const std::vector<int64_t>& current_tokens) override {
        if (temp_ == 1.0f) return;
        for (auto& logit : logits) {
            logit /= temp_;
        }
    }

private:
    float temp_;
};

/*
// ----------------------------------------
// Bar/Track Stop Processor : Stop generation when enough content is generated.
// ----------------------------------------
class StopLogitsProcessor : public LogitsProcessor {
public:
    StopLogitsProcessor(
        int bar_start_token_id,
        int eos_token_id,
        libtok::Tokenizer* tokenizer)
        : bar_start_token_id_(bar_start_token_id),
          eos_token_id_(eos_token_id),
          tokenizer_(tokenizer) {}

    void set_n_bars_to_infill(int n) { n_bars_to_infill_ = n; }
    void set_n_attribute_controls(int n) { n_attribute_controls_ = n; }

    void operator()(std::vector<int64_t>& input_ids,
                    std::vector<float>& scores) override {
        auto start = std::chrono::high_resolution_clock::now();

        // 1. Find FillBar_Start position
        auto it = std::find(input_ids.begin(), input_ids.end(), bar_start_token_id_);
        if (it == input_ids.end()) {
            // No FillBar_Start token found, do nothing
            return;
        }

        size_t fill_start_idx = std::distance(input_ids.begin(), it);

        // 2. Extract tokens after attribute controls
        size_t decode_start = fill_start_idx + n_attribute_controls_ + 1;
        std::vector<int64_t> generated_tokens;
        if (decode_start < input_ids.size()) {
            generated_tokens.assign(input_ids.begin() + decode_start, input_ids.end());
            tokenizer_->decode(generated_tokens);  // Assume decode fills some internal state
        }

        // 3. Count Bar_None occurrences
        int n_bar_none = 0;
        for (int64_t id : generated_tokens) {
            if (id == tokenizer_->vocab_id("Bar_None")) {
                ++n_bar_none;
            }
        }

        // 4. Stop condition: force EOS if enough bars generated
        if (n_bar_none > n_bars_to_infill_) {
            std::fill(scores.begin(), scores.end(), -1e9f); // mask all
            scores[eos_token_id_] = 1e9f;                  // force EOS
        } else {
            // Prevent EOS until enough bars are generated
            scores[eos_token_id_] = -1e9f;
        }

        // 5. Custom rule: don't stop after Duration_5.0.1
        if (!input_ids.empty() &&
            input_ids.back() == tokenizer_->vocab_id("Duration_5.0.1")) {
            scores[eos_token_id_] = -1e9f;
        }

        auto end = std::chrono::high_resolution_clock::now();
        total_time_ += std::chrono::duration<double>(end - start).count();
    }

    double total_time() const { return total_time_; }

private:
    int bar_start_token_id_;
    int eos_token_id_;
    libtok::Tokenizer* tokenizer_;
    int n_bars_to_infill_ = 0;
    int n_attribute_controls_ = 0;
    double total_time_ = 0.0;
};
*/

}

}
