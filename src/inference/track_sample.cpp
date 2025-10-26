#include "inference.h"

#include <algorithm>

namespace mmm {

namespace inference {

void sample_tracks(
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::MMM &tokenizer,           
    const mmm::inference::TrackSampling& sample_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine,
    int context_length,
    mmm::utils::Logger& logger
) {
    using Level = mmm::utils::LogLevel;

    logger.log(Level::DEBUG, "[SampleTracks] Beginning sampling");

    for (auto &[program, controls] : sample_config.tracks) {
        logger.log(Level::DEBUG, "[SampleTracks] New track, program " + std::to_string(program));

        for (auto &control : controls) {
            logger.log(Level::DEBUG, "[SampleTracks] Using control: " + control);
        }

        int num_bars_to_gen = token_seq[0].splitPerBars().size();
        int num_tracks_before = token_seq.size();

        engine.updateProcessors("TrackSampleStopLogitsProcessor.active", 1);
        logger.log(Level::DEBUG, "[SampleTracks] Activated track sample LogitsProcessor");

        engine.updateProcessors("TrackSampleStopLogitsProcessor.reset", 0);
        logger.log(Level::DEBUG, "[SampleTracks] Reset LogitsProcessor");

        engine.updateProcessors("TrackSampleStopLogitsProcessor.n_bars_to_generate", num_bars_to_gen);
        logger.log(Level::DEBUG, "[SampleTracks] Set LogitsProcessor bar count");

        if (model->is_cached()) {
            model->reset_cache();
            logger.log(Level::DEBUG, "[SampleTracks] Reset model cache");
        }

        LibTok::TokSequence input_tokens({}, {}, {}, {}, true);
        int last_track_start_idx = _adapt_prompt_for_sampling(
            tokenizer,
            program,
            controls,
            token_seq,
            input_tokens,
            context_length,
            logger
        );

        std::vector<int64_t> input_ids;
        input_ids.reserve(input_tokens.size());
        for (int id : input_tokens.ids) {
            input_ids.push_back(static_cast<int64_t>(id));
        }

        logger.log(Level::DEBUG, "[SampleTracks] Generating " + std::to_string(input_ids.size()) + " input ids");

        std::vector<int64_t> generated = engine.generate(input_ids, model, logger);
        logger.log(Level::DEBUG, "[SampleTracks] Generation complete. Tokens = " + std::to_string(generated.size()));

        std::vector<int> gen_ids;
        gen_ids.reserve(generated.size());
        for (auto v : generated) gen_ids.push_back(static_cast<int>(v));

        LibTok::TokSequence generated_seq({}, gen_ids, {}, {}, true);

        if (tokenizer.isTrained()) {
            tokenizer.decodeTokenIds(generated_seq);
        }

        std::vector<std::string> tokens = generated_seq.tokens;
        logger.log(Level::DEBUG, "[SampleTracks] Generated sequence decoded. Size = " + std::to_string(tokens.size()));

        auto sampled_track_tokens = extractSampledContent(tokens, last_track_start_idx, num_bars_to_gen, num_tracks_before, logger);
        LibTok::TokSequence sampled_track(sampled_track_tokens);
        tokenizer.complete_sequence(sampled_track);

        logger.log(Level::DEBUG, "[SampleTracks] Sampled track complete. Length = " + std::to_string(sampled_track.size()));
        token_seq.push_back(sampled_track);

        logger.log(Level::DEBUG, "[SampleTracks] Track " + std::to_string(token_seq.size()) + " sampled.");
    }

    engine.updateProcessors("TrackSampleStopLogitsProcessor.active", 0);
    logger.log(Level::DEBUG, "[SampleTracks] Deactivated track sample LogitsProcessor");
    logger.log(Level::DEBUG, "[SampleTracks] Sampling finished");
}



int _adapt_prompt_for_sampling(
    LibTok::MMM &tokenizer,
    int program,
    const Controls &controls,
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::TokSequence &input_tokens,
    int context_length,
    mmm::utils::Logger& logger
) {
    using Level = mmm::utils::LogLevel;

    logger.log(Level::DEBUG, "[AdaptPromptForTrackSample] Context Length = " + std::to_string(context_length));
    logger.log(Level::DEBUG, "[AdaptPromptForTrackSample] Track Program = " + std::to_string(program));

    std::string controls_str;
    for (const auto &control : controls) {
        controls_str += control + " ";
    }
    logger.log(Level::DEBUG, "[AdaptPromptForTrackSample] Controls = [" + controls_str + "]");

    // ------------------------------------------------------------
    // Build the context from previous token sequences
    // ------------------------------------------------------------
    for (int i = 0; i < static_cast<int>(token_seq.size()); i++) {
        auto bar_subseqs = token_seq[i].splitPerBars();
        int num_bars = static_cast<int>(bar_subseqs.size());
        int tokseq_len = static_cast<int>(token_seq[i].size());

        int context_start_idx = 0;
        int context_end_idx = std::min(context_length, num_bars);

        logger.log(Level::TRACE, "[AdaptPromptForTrackSample] Track " + std::to_string(i) +
                                 " has " + std::to_string(num_bars) +
                                 " bars, using bars [" + std::to_string(context_start_idx) +
                                 ", " + std::to_string(context_end_idx) + ")");

        // Optionally add small intro tokens if not at the start
        if (context_start_idx != 0) {
            input_tokens += token_seq[i].slice(0, 2);
        }

        for (int bar_idx = context_start_idx; bar_idx < context_end_idx; bar_idx++) {
            input_tokens += bar_subseqs[bar_idx];
        }
    }

    int last_track_start_idx = static_cast<int>(input_tokens.size());
    tokenizer.encodeTokenIds(input_tokens);

    // ------------------------------------------------------------
    // Build "new track" start tokens
    // ------------------------------------------------------------
    std::vector<std::string> new_track_tokens = {
        "Track_Start",
        "Program_" + std::to_string(program)
    };

    for (const auto &control : controls) {
        new_track_tokens.push_back(control);
    }

    LibTok::TokSequence new_track_start_seq(new_track_tokens);
    tokenizer.complete_sequence(new_track_start_seq);
    input_tokens += new_track_start_seq;

    logger.log(Level::DEBUG, "[AdaptPromptForTrackSample] Final input_tokens size = " + std::to_string(input_tokens.size()));

    return last_track_start_idx;
}


std::vector<std::string> extractSampledContent(
    std::vector<std::string>& tokens,
    size_t last_track_start_idx,
    int num_bars_to_gen,
    int num_tracks_before,
    mmm::utils::Logger& logger
) {
    using Level = mmm::utils::LogLevel;
    std::vector<std::string> track_tokens;

    logger.log(Level::DEBUG, "[extractSampledContent] Starting extraction. "
                             "last_track_start_idx=" + std::to_string(last_track_start_idx) +
                             ", num_bars_to_gen=" + std::to_string(num_bars_to_gen) +
                             ", num_tracks_before=" + std::to_string(num_tracks_before));

    // --- 1. Remove EOS if last
    if (!tokens.empty() && tokens.back() == "EOS_None") {
        logger.log(Level::DEBUG, "[extractSampledContent] Removing EOS_None token at end");
        tokens.pop_back();
    }

    // --- 2. Find all Track_Start indices
    std::vector<size_t> track_starts;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == "Track_Start") {
            track_starts.push_back(i);
        }
    }

    if (track_starts.empty()) {
        logger.log(Level::ERROR, "[extractSampledContent] No Track_Start found in tokens.");
        throw std::runtime_error("[extractSampledContent] No Track_Start found in tokens.");
    }

    // --- 3. Determine which track to extract
    size_t expected_track_idx = static_cast<size_t>(num_tracks_before); // 0-based
    if (expected_track_idx >= track_starts.size()) {
        std::string msg = "[extractSampledContent] Not enough tracks generated. Expected at least " +
                          std::to_string(num_tracks_before + 1) + ", found " + std::to_string(track_starts.size());
        logger.log(Level::ERROR, msg);
        throw std::runtime_error(msg);
    }

    size_t correct_start_idx = track_starts[expected_track_idx];
    if (last_track_start_idx != correct_start_idx) {
        logger.log(Level::WARN, "[extractSampledContent] last_track_start_idx=" +
                                std::to_string(last_track_start_idx) +
                                " does not match expected track_start=" +
                                std::to_string(correct_start_idx) +
                                ". Using expected track start instead.");
    }

    last_track_start_idx = correct_start_idx;

    // --- 4. Extract tokens for this track
    size_t start = last_track_start_idx;

    auto it_next_start = std::find(tokens.begin() + start + 1, tokens.end(), "Track_Start");
    auto it_end = std::find(tokens.begin() + start,
                            (it_next_start != tokens.end() ? it_next_start : tokens.end()),
                            "Track_End");

    if (it_end != tokens.end()) {
        // Found a proper Track_End
        track_tokens.assign(tokens.begin() + start, it_end + 1);
        logger.log(Level::DEBUG, "[extractSampledContent] Extracted track section up to Track_End");
    } 
    else if (it_next_start != tokens.end()) {
        // No Track_End, but another Track_Start came → cut there
        track_tokens.assign(tokens.begin() + start, it_next_start);
        track_tokens.push_back("Track_End");
        logger.log(Level::WARN, "[extractSampledContent] No Track_End before next Track_Start, appended one manually.");
    } 
    else {
        // Last track, no further Track_Start
        track_tokens.assign(tokens.begin() + start, tokens.end());
        if (track_tokens.empty() || track_tokens.back() != "Track_End") {
            track_tokens.push_back("Track_End");
            logger.log(Level::WARN, "[extractSampledContent] No Track_End found at end of tokens, appended one manually.");
        }
    }

    if (track_tokens.empty() || track_tokens.back() != "Track_End") {
        logger.log(Level::WARN, "[extractSampledContent] Track_End token absent even after extraction, forcing append.");
        track_tokens.push_back("Track_End");
    }

    logger.log(Level::DEBUG, "[extractSampledContent] Extraction complete. Extracted " +
                            std::to_string(track_tokens.size()) + " tokens.");

    return track_tokens;
}

}
}