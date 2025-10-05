#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <cmath>
#include <algorithm>

#include "generationconfig.h"

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

    const char* name() const override { return "RepetitionPenaltyLogitsProcessor"; }

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

    const char* name() const override { return "MaxLengthLogitsProcessor"; }

    void process(std::vector<float>& logits,
                 const std::vector<int64_t>& current_tokens) override {
        if (current_tokens.size() >= max_length_ &&
            eos_token_id_ >= 0 &&
            eos_token_id_ < (int)logits.size()) {
            std::fill(logits.begin(), logits.end(), -1e9f); // mask all
            logits[eos_token_id_] = 1e9f;
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

    const char* name() const override { return "MinLengthLogitsProcessor"; }

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
// BaseStopLogitsProcessor : Common logic for active state and config updates.
// ----------------------------------------
class BaseStopLogitsProcessor : public LogitsProcessor {
public:
    BaseStopLogitsProcessor(const std::string& name) : name_(name) {}

    const char* name() const override { return name_.c_str(); }

    // Generic updateConfig that handles "reset" and "active"
    void updateConfig(const std::string& key, int value) override {
        std::string prefix = name_ + ".";
        if (key.rfind(prefix, 0) != 0) return; // not our prefix

        std::string local_key = key.substr(prefix.size());
        if (local_key == "reset") {
            reset();
        } else if (local_key == "active") {
            active_ = (value != 0);
        } else {
            handleUpdate(local_key, value); // subclass-specific
        }
    }

    void reset() {
        handleReset();
    }

    bool isActive() const { return active_; }

protected:
    // Hooks for subclasses
    virtual void handleUpdate(const std::string& key, int value) {}
    virtual void handleReset() {}

    bool active_ = false;

private:
    std::string name_;
};


// ----------------------------------------
// Bar Infill Stop Processor : Stop generation when enough content is generated.
// ----------------------------------------
class BarInfillStopLogitsProcessor : public BaseStopLogitsProcessor {
public:
    BarInfillStopLogitsProcessor(int infill_token_id, int bar_token_id, int eos_token_id)
        : BaseStopLogitsProcessor("BarInfillStopLogitsProcessor"),
          infill_token_id_(infill_token_id),
          bar_token_id_(bar_token_id),
          eos_token_id_(eos_token_id) {}

    void process(std::vector<float>& logits,
                 const std::vector<int64_t>& current_tokens) override {
        if (!isActive()) return;

        logits[track_start_token_id_] = -1e9f;

        if (!bar_start_found_) {
            auto it = std::find(current_tokens.begin(), current_tokens.end(), infill_token_id_);
            if (it == current_tokens.end()) return;

            fill_start_idx_ = std::distance(current_tokens.begin(), it);
            bar_start_found_ = true;
            n_bar_none_ = 0;
            return;
        }

        if (!current_tokens.empty()) {
            int64_t last_id = current_tokens.back();
            if (last_id == bar_token_id_) ++n_bar_none_;

            if (n_bar_none_ > n_bars_to_infill_) {
                std::fill(logits.begin(), logits.end(), -1e9f);
                logits[eos_token_id_] = 1e9f;
            } else {
                logits[eos_token_id_] = -1e9f;
            }
        }
    }

protected:
    void handleUpdate(const std::string& key, int value) override {
        if (key == "n_bars_to_infill") n_bars_to_infill_ = value;
        else if (key == "n_attribute_controls") n_attribute_controls_ = value;
    }

    void handleReset() override {
        bar_start_found_ = false;
        fill_start_idx_ = 0;
        n_bar_none_ = 0;
    }

private:
    int infill_token_id_;
    int bar_token_id_;
    int eos_token_id_;

    int n_bars_to_infill_ = 0;
    int n_attribute_controls_ = 0;

    bool bar_start_found_ = false;
    size_t fill_start_idx_ = 0;
    int n_bar_none_ = 0;
};

// ----------------------------------------
// Track Sample Stop Processor : Stop generation when enough content is generated.
// ----------------------------------------
class TrackSampleStopLogitsProcessor : public BaseStopLogitsProcessor {
public:
    TrackSampleStopLogitsProcessor(int bar_token_id, int track_start_token_id, int track_end_token_id, int eos_token_id)
        : BaseStopLogitsProcessor("TrackSampleStopLogitsProcessor"),
          bar_token_id_(bar_token_id),
          track_end_token_id_(track_end_token_id),
          track_start_token_id_(track_start_token_id),
          eos_token_id_(eos_token_id) {}

    void process(std::vector<float>& logits,
                 const std::vector<int64_t>& current_tokens) override {
        if (!isActive() || current_tokens.empty()) return;

        logits[track_start_token_id_] = -1e9f;

        int64_t last_id = current_tokens.back();

        if (!finished_bars_ && last_id == bar_token_id_) {
            ++n_bars_generated_;
            if (n_bars_generated_ >= n_bars_to_generate_) finished_bars_ = true;
        }

        if (!finished_bars_) {
            logits[track_end_token_id_] = -1e9f;
            logits[eos_token_id_] = -1e9f;
        } else {
            logits[bar_token_id_] = -1e9f;
            if (last_id == track_end_token_id_) {
                std::fill(logits.begin(), logits.end(), -1e9f);
                logits[eos_token_id_] = 1e9f;
            }
        }
    }

protected:
    void handleUpdate(const std::string& key, int value) override {
        if (key == "n_bars_to_generate") n_bars_to_generate_ = value;
    }

    void handleReset() override {
        n_bars_generated_ = 0;
        finished_bars_ = false;
    }

private:
    int bar_token_id_;
    int track_start_token_id_;
    int track_end_token_id_;
    int eos_token_id_;

    int n_bars_to_generate_ = 0;
    int n_bars_generated_ = 0;
    bool finished_bars_ = false;
};

}
}
