#pragma once

#include "model.h"
#include "engine.h"
#include "promptconfig.h"
#include "mmm.h"
#include "tok_sequence.h"

namespace mmm {
namespace inference {

using Controls = std::vector<std::string>;
using BarSubset = std::tuple<int, int, Controls>;

mmm::sampling::SamplingEngine createEngine(
    mmm::sampling::GenerationConfig& config,
    LibTok::MMM &tokenizer,
    int seed = -1
);

LibTok::ScoreType generate(
    mmm::IModel* model, 
    PromptConfig &prompt_config,
    mmm::sampling::SamplingEngine &engine,
    LibTok::ScoreType &score
);

void infill_bars(
    std::vector<LibTok::TokSequence> &token_seq,
    mmm::inference::BarInfilling &infill_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine
);

std::pair<int, int> _adapt_prompt_for_infilling(
    int track_idx, 
    BarSubset &subset,
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::TokSequence &input_tokens
);

}
}