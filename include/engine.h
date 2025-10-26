#pragma once

#include <algorithm>

#include "sampler.h"
#include "logitsprocessor.h"
#include "logitswarper.h"
#include "model.h"
#include "profiler.h"
#include "generationconfig.h"
#include "logger.h"

namespace mmm {

namespace sampling {

class SamplingEngine {
public:
    SamplingEngine(const GenerationConfig& config,
                    int eos_token_id,
                    int vocab_size,
                    LogitsProcessorList processors = {},
                    LogitsWarperList warpers = {},
                    Sampler sampler = {},
                    bool profiler = false);

    std::vector<int64_t> generate(
        const std::vector<int64_t>& input_ids,
        mmm::IModel* model,
        mmm::utils::Logger& logger
    );

    void updateProcessors(const std::string& key, int value);

    void resetProfiler() {
        if (profiler_) profiler_->reset();
    }

    double totalTimeProfiler() {
        if (profiler_) return profiler_->total_time();
        return 0.0;
    }

    const GenerationConfig& getConfig() const { return config_; }
    void setConfig(const GenerationConfig& cfg) { config_ = cfg; }

    int getVocabSize() const { return vocab_size_; }
    int getEosTokenId() const { return eos_token_id_; }

    // Store the seed (optional — if not already)
    int getSeed() const { return sampler_.getSeed(); }
    void setSeed(int new_seed) { 
        sampler_.setSeed(new_seed); }

private:
    GenerationConfig config_;
    LogitsProcessorList processors_;
    LogitsWarperList warpers_;
    Sampler sampler_;
    mmm::utils::Profiler* profiler_;
    int eos_token_id_;
    int vocab_size_;
};

}
}
