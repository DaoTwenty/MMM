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

    virtual const char* name() const = 0;

    virtual void updateConfig(const std::string& key, int value) {}
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

    void update(const std::string& key, int value) { 
        for (auto& processor : processors_) { 
            processor->updateConfig(key, value); 
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
// MaxLength Processor: force EOS when max length is reached
// ----------------------------------------
class MaxLengthLogitsProcessor : public LogitsProcessor {
public:
    MaxLengthLogitsProcessor(size_t max_length, int eos_token_id)
        : max_length_(max_length), eos_token_id_(eos_token_id) {}

    void process(std::vector<float>& logits,
                 const std::vector<int64_t>& current_tokens) override {
        if (current_tokens.size() < min_length_ &&
            eos_token_id_ >= 0 &&
            eos_token_id_ < (int)logits.size()) {
            std::fill(logits.begin(), logits.end(), -1e9f); // mask all
            scores[eos_token_id_] = 1e9f;
        }
    }

private:
    size_t max_length_;
    int eos_token_id_;
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
// Bar Infill Stop Processor : Stop generation when enough content is generated.
// ----------------------------------------
class InfillStopLogitsProcessor : public LogitsProcessor {
public:
    InfillStopLogitsProcessor(
        int infill_token_id,
        int bar_token_id,
        int eos_token_id)
        : infill_token_id_(infill_token_id),
          bar_token_id_(bar_token_id),
          eos_token_id_(eos_token_id) {}

    const char* name() const override { 
        return "InfillStopLogitsProcessor"; 
    }

    void updateConfig(const std::string& key, int value) override { 
        // expect keys like "InfillStopLogitsProcessor.n_bars_to_infill" 
        std::string prefix = std::string(name()) + "."; 
        if (key.rfind(prefix, 0) != 0) return; 
        // not our prefix 
        std::string local_key = key.substr(prefix.size()); 
        if (local_key == "n_bars_to_infill") { 
            set_n_bars_to_infill(value); 
        } else if (local_key == "n_attribute_controls") { 
            set_n_attribute_controls(value); 
        } else if (local_key == "reset") {
            reset();
        }
    }

    void set_n_bars_to_infill(int n) { n_bars_to_infill_ = n; }
    void set_n_attribute_controls(int n) { n_attribute_controls_ = n; }

    void reset() {
        bar_start_found_ = false;
        fill_start_idx_ = 0;
        n_bar_none_ = 0;
    }

    void operator()(std::vector<int64_t>& input_ids,
                    std::vector<float>& scores) override {
        auto start = std::chrono::high_resolution_clock::now();

        if (!bar_start_found_) {
            // 1. Look for FillBar_Start for the first time
            auto it = std::find(input_ids.begin(), input_ids.end(), infill_token_id_);
            if (it == input_ids.end()) {
                // Not found yet → do nothing
                return;
            }
            fill_start_idx_ = std::distance(input_ids.begin(), it);
            bar_start_found_ = true;
            n_bar_none_ = 0; // reset counter
            return;          // skip stopping logic this step
        }

        // 2. If bar_start already found, only check the *last appended token*
        if (!input_ids.empty()) {
            int64_t last_id = input_ids.back();

            if (last_id == bar_token_id_) {
                ++n_bar_none_;
            }

            // Stop condition: if we exceeded the planned number of bars
            if (n_bar_none_ > n_bars_to_infill_) {
                std::fill(scores.begin(), scores.end(), -1e9f); // mask all
                scores[eos_token_id_] = 1e9f;                   // force EOS
            } else {
                // Prevent EOS until enough bars are generated
                scores[eos_token_id_] = -1e9f;
            }
        }
    }

private:
    int infill_token_id_;
    int bar_token_id_;
    int eos_token_id_;

    int n_bars_to_infill_ = 0;
    int n_attribute_controls_ = 0;

    // State tracking
    bool bar_start_found_ = false;
    size_t fill_start_idx_ = 0;
    int n_bar_none_ = 0; // running count
};

LogitsProcessorList createProcessorListFromConfig(const GenerationConfig& config, int eos_token_id, int infill_token_id, int bar_token_id) { 
    LogitsProcessorList processors; 
    // Repetition penalty 
    if (config.repetition_penalty != 1.0f) { 
        processors.addProcessor( std::make_shared<RepetitionPenaltyLogitsProcessor>(config.repetition_penalty)); 
    } 
    // Min length processor (forbid EOS until at least min length) 
    if (config.max_new_tokens > 0) { 
        processors.addProcessor( std::make_shared<MaxLengthLogitsProcessor>(config.max_new_tokens, eos_token_id)); 
    } 

    if (config.min_new_tokens > 0) { 
        processors.addProcessor( std::make_shared<MinLengthLogitsProcessor>(config.min_new_tokens, eos_token_id)); 
    }
    // Always include InfillStopLogitsProcessor  
    auto infill_stop = std::make_shared<InfillStopLogitsProcessor>( infill_token_id, bar_token_id, eos_token_id); 
    processors.addProcessor(infill_stop); 
    return processors;
}

}
}
