#pragma once

#include "model.h"
#include "engine.h"
#include "promptconfig.h"
#include "mmm.h"
#include "tok_sequence.h"
#include "utils.h"

namespace mmm {
namespace inference {

mmm::sampling::SamplingEngine createEngine(
    mmm::sampling::GenerationConfig& config,
    LibTok::MMM &tokenizer,
    int seed = -1,
    bool verbose = false
);

LibTok::ScoreType generate(
    mmm::IModel* model, 
    LibTok::MMM &tokenizer,
    PromptConfig &cfg,
    mmm::sampling::SamplingEngine &engine,
    LibTok::ScoreType &score,
    bool verbose = false
);

void _preprocess_token_sequence_vector(
    std::vector<LibTok::TokSequence> &token_seq,
    bool pad = true,
    bool verbose = false
);

void sample_tracks(
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::MMM &tokenizer,           
    mmm::inference::TrackSampling &sample_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine,
    int context_length,
    bool verbose = false
);

void infill_bars(
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::MMM &tokenizer,                               
    mmm::inference::BarInfilling &infill_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine,
    int context_length,
    bool verbose = false
);

std::pair<int, int> _adapt_prompt_for_infilling(
    LibTok::MMM &tokenizer,
    int track_idx, 
    BarSubset &subset,
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::TokSequence &input_tokens,
    int context_length,
    bool verbose = false
);

std::vector<std::string> extractInfilledContent(
    const std::vector<std::string>& tokens,
    size_t fillbar_start_idx,
    int num_bars_to_infill,
    bool verbose = false
);

int _adapt_prompt_for_sampling(
    LibTok::MMM &tokenizer,
    int program,
    Controls &controls,
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::TokSequence &input_tokens,
    int context_length,
    bool verbose = false
);

std::vector<std::string> extractSampledContent(
    const std::vector<std::string>& tokens,
    size_t last_track_start_idx,
    int num_bars_to_gen,
    int num_tracks_before,
    bool verbose
);

}
}