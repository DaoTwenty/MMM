#pragma once

#include "promptconfig.h"
#include "model.h"
#include "engine.h"
#include "mmm.h"
#include "tok_sequence.h"
#include "utils.h"
#include "token_utils.h"
#include "logger.h"

namespace mmm {
namespace inference {

mmm::sampling::SamplingEngine createEngine(
    mmm::sampling::GenerationConfig& config,
    LibTok::MMM &tokenizer,
    mmm::utils::Logger& logger,
    int seed = -1
);

LibTok::ScoreType generate(
    mmm::IModel* model, 
    LibTok::MMM &tokenizer,
    PromptConfig &cfg,
    mmm::sampling::SamplingEngine &engine,
    LibTok::ScoreType &score,
    mmm::utils::Logger& logger
);

void _preprocess_token_sequence_vector(
    std::vector<LibTok::TokSequence> &token_seq,
    mmm::utils::Logger& logger,
    bool pad = true
);

void sample_tracks(
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::MMM &tokenizer,           
    const mmm::inference::TrackSampling &sample_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine,
    int context_length,
    mmm::utils::Logger& logger
);

void infill_bars(
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::MMM &tokenizer,                               
    const mmm::inference::BarInfilling &infill_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine,
    int context_length,
    mmm::utils::Logger& logger
);

std::pair<int, int> _adapt_prompt_for_infilling(
    LibTok::MMM &tokenizer,
    int track_idx, 
    const BarSubset &subset,
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::TokSequence &input_tokens,
    int context_length,
    mmm::utils::Logger& logger
);

std::vector<std::string> extractInfilledContent(
    const std::vector<std::string>& tokens,
    size_t fillbar_start_idx,
    int num_bars_to_infill,
    mmm::utils::Logger& logger
);

int _adapt_prompt_for_sampling(
    LibTok::MMM &tokenizer,
    int program,
    const Controls &controls,
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::TokSequence &input_tokens,
    int context_length,
    mmm::utils::Logger& logger
);

std::vector<std::string> extractSampledContent(
    std::vector<std::string>& tokens,
    size_t last_track_start_idx,
    int num_bars_to_gen,
    int num_tracks_before,
    mmm::utils::Logger& logger
);

}
}