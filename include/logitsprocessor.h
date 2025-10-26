#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <string> 

#include "generationconfig.h"
#include "logger.h"

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
                        const std::vector<int64_t>& current_tokens,
                        mmm::utils::Logger& logger
                    ) = 0;

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
                const std::vector<int64_t>& current_tokens,
                mmm::utils::Logger& logger
            ) const {
        for (const auto& processor : processors_) {
            processor->process(logits, current_tokens, logger);
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
                 const std::vector<int64_t>& current_tokens,
                mmm::utils::Logger& logger) override {

        if (penalty_ == 1.0f) return; // no-op

        for (int64_t token : current_tokens) {
            if (token >= 0 && token < (int64_t)logits.size()) {
                float old_val = logits[token];
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
class BarInfillStopLogitsProcessor : public BaseStopLogitsProcessor {
public:
    BarInfillStopLogitsProcessor(
        int infill_start_token_id,
        int infill_end_token_id,
        int bar_token_id,
        int eos_token_id,
        const std::unordered_map<int, bool>& contains_bar)
        : BaseStopLogitsProcessor("BarInfillStopLogitsProcessor"),
          infill_start_token_id_(infill_start_token_id),
          infill_end_token_id_(infill_end_token_id),
          bar_token_id_(bar_token_id),
          eos_token_id_(eos_token_id),
          contains_bar_(contains_bar)
    {
        // Precompute mask list (for faster iteration later)
        for (const auto& [token_id, has_bar] : contains_bar_) {
            if (has_bar) {
                tokens_containing_bar_.push_back(token_id);
            }
        }
    }

    void process(std::vector<float>& logits, const std::vector<int64_t>& current_tokens, mmm::utils::Logger& logger) override {
        if (!isActive() || current_tokens.empty()) return;

        // Mask EOS by default
        logits[eos_token_id_] = -1e10f;

        int64_t last_token = current_tokens.back();
        logger.log(mmm::utils::LogLevel::TRACE, "[BarInfillStopLogitsProcessor] last_token=" + std::to_string(last_token)); 

        // If the fill has ended, force EOS
        if (last_token == infill_end_token_id_) {
            logger.log(mmm::utils::LogLevel::TRACE, "[BarInfillStopLogitsProcessor] Last token is INFILL_END");
            std::fill(logits.begin(), logits.end(), -1e10f);
            logits[eos_token_id_] = 1e10f;
            return;
        }
        // Count bars
        else if (last_token == bar_token_id_ || contains_bar_[last_token]) {
            logger.log(mmm::utils::LogLevel::TRACE, "[BarInfillStopLogitsProcessor] Last token contains BAR_NONE");
            ++num_bars_generated_;
        }

        // If enough bars generated, stop allowing more bar tokens
        if (num_bars_generated_ >= n_bars_to_infill_) {
            logger.log(mmm::utils::LogLevel::TRACE, "[BarInfillStopLogitsProcessor] Enough bars generated");

            // Mask all tokens containing "bar"
            for (int token_id : tokens_containing_bar_) {
                logits[token_id] = -1e10f;
            }

            // Also mask the main bar token
            logits[bar_token_id_] = -1e10f;

            // Allow FillBar_End
            //logits[infill_end_token_id_] = 1e10f;
            return;
        }
    }

protected:
    void handleUpdate(const std::string& key, int value) override {
        if (key == "n_bars_to_infill") n_bars_to_infill_ = value;
    }

    void handleReset() override {
        bar_start_found_ = false;
        num_bars_generated_ = 0;
    }

private:
    int infill_start_token_id_;
    int infill_end_token_id_;
    int bar_token_id_;
    int eos_token_id_;
    std::unordered_map<int, bool> contains_bar_;

    // Precomputed mask list
    std::vector<int> tokens_containing_bar_;

    int n_bars_to_infill_ = 0;
    int num_bars_generated_ = 0;
    bool bar_start_found_ = false;
};


// ----------------------------------------
// Track Sample Stop Processor : Stop generation when enough content is generated.
// ----------------------------------------
class TrackSampleStopLogitsProcessor : public BaseStopLogitsProcessor {
public:
    TrackSampleStopLogitsProcessor(int bar_token_id, int track_start_token_id, int track_end_token_id, int eos_token_id, std::unordered_map<int, bool> contains_bar, std::unordered_map<int, bool> contains_track_end)
        : BaseStopLogitsProcessor("TrackSampleStopLogitsProcessor"),
          bar_token_id_(bar_token_id),
          track_end_token_id_(track_end_token_id),
          track_start_token_id_(track_start_token_id),
          eos_token_id_(eos_token_id), 
          contains_track_end_(contains_track_end),
          contains_bar_(contains_bar) {}

    void process(std::vector<float>& logits,
                 const std::vector<int64_t>& current_tokens,
                mmm::utils::Logger& logger) override {
        if (!isActive() || current_tokens.empty()) {
            return;
        }

        int64_t last_id = current_tokens.back();

        if (last_id == track_end_token_id_ || contains_track_end_[last_id]) {
            std::fill(logits.begin(), logits.end(), -1e10f);
            logits[eos_token_id_] = 1e10f;
            return;
        }
    
        logits[track_start_token_id_] = -1e10f;

        if (!finished_bars_ && (last_id == bar_token_id_ || contains_bar_[last_id])) {
            ++n_bars_generated_;
            if (n_bars_generated_ >= n_bars_to_generate_) {
                finished_bars_ = true;
            }
        }

        if (!finished_bars_) {
            // Removing because bar counting doesn't work due to BPE bar token encoding
            //logits[track_end_token_id_] = -1e10f;
            logits[eos_token_id_] = -1e10f;
        } else {
            logits[bar_token_id_] = -1e10f;
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
    std::unordered_map<int, bool> contains_bar_;
    std::unordered_map<int, bool> contains_track_end_;

    int n_bars_to_generate_ = 0;
    int n_bars_generated_ = 0;
    bool finished_bars_ = false;
};

}
}
