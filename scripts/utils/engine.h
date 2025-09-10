#pragma once

#include "sampler.h"
#include "logitsprocessor.h"
#include "logitswarper.h"
#include "model.h"
#include "profiler.h"
#include "config.h"

namespace mmm {

namespace sampling {

class SamplingEngine {
public:
    SamplingEngine(const GenerationConfig& config,
                   LogitsProcessorList processors = {},
                   LogitsWarperList warpers = {},
                   Sampler sampler = {},
                   mmm::utils::Profiler* profiler = nullptr);

    std::vector<int64_t> generate(
        const std::vector<int64_t>& input_ids,
        mmm::IModel* model
    );

private:
    GenerationConfig config_;
    LogitsProcessorList processors_;
    LogitsWarperList warpers_;
    Sampler sampler_;
    mmm::utils::Profiler* profiler_;
};

}
}
