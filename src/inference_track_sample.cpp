#include "inference.h"

#include <algorithm>

namespace mmm {

namespace inference {

void sample_tracks(
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::MMM &tokenizer,           
    mmm::inference::TrackSampling &sample_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine,
    bool verbose
) {
    if (verbose) std::cout << "[SampleTracks] Beginning infill\n";

    for (auto &[program, controls] : sample_config.tracks) {
        if (verbose) std::cout << "[SampleTracks] New track, program " << program << "\n";
        for (auto &control : controls) {
            if (verbose) std::cout << "[SampleTracks] Using control : " << control << "\n";
        }

        // Preprocessed so every track has same number of bars
        int num_bars_to_gen = token_seq[0].splitPerBars().size();
        int num_tracks_before = token_seq.size();

        engine.updateProcessors("TrackSampleStopLogitsProcessor.active", 1);
        if (verbose) std::cout << "[SampleTracks] Activated track sample LogitsProcessor\n";
        engine.updateProcessors("TrackSampleStopLogitsProcessor.reset", 0);
        if (verbose) std::cout << "[SampleTracks] Reset LogitsProcessor\n";
        engine.updateProcessors("TrackSampleStopLogitsProcessor.n_bars_to_generate", num_bars_to_gen);
        if (verbose) std::cout << "[SampleTracks] Reset LogitsProcessor bar count\n";

        if (model->is_cached()) {
            if (verbose) std::cout << "[SampleTracks] Resetting model cache\n";
            model->reset_cache();
        }

        LibTok::TokSequence input_tokens({}, {}, {}, {}, true);
        int last_track_start_idx = _adapt_prompt_for_sampling(
            tokenizer,
            program,
            controls,
            token_seq,
            input_tokens,
            verbose
        );

        if (verbose) { 
            std::cout << "[SampleTracks] Input sequence. Size=" 
            << input_tokens.tokens.size() << ".\n";
        }

        std::vector<int64_t> input_ids;
        input_ids.reserve(input_tokens.size());

        for (int id : input_tokens.ids) {
            input_ids.push_back(static_cast<int64_t>(id));
        }

        if (verbose) std::cout << "[SampleTracks] Generating " << input_ids.size() << " input ids\n";
        std::vector<int64_t> generated = engine.generate(input_ids, model, verbose);
        if (verbose) std::cout << "[SampleTracks] Generation complete. Tokens=" << generated.size() << "\n";

        std::vector<int> gen_ids;
        gen_ids.reserve(generated.size());
        for (auto v : generated) gen_ids.push_back(static_cast<int>(v));

        LibTok::TokSequence generated_seq({}, gen_ids, {}, {}, true);

        tokenizer.complete_sequence(generated_seq);
        if (tokenizer.isTrained()) {
            tokenizer.decodeTokenIds(generated_seq);
        }

        std::vector<std::string> tokens = generated_seq.tokens;

        if (verbose) { 
            std::cout << "[SampleTracks] Generated sequence decoded. Size=" 
            << tokens.size() << ".\n";
        }

        auto sampled_track_tokens = extractSampledContent(tokens, last_track_start_idx, num_bars_to_gen, num_tracks_before, verbose);

        LibTok::TokSequence sampled_track(sampled_track_tokens);
        tokenizer.complete_sequence(sampled_track);

        if (verbose) { 
            std::cout << "[SampleTracks] Sampled track sequence complete. Length=" << sampled_track.size() << "\n"; 
        }

        token_seq.push_back(sampled_track);

        if (verbose) { 
            std::cout << "[SampleTracks] Track " << token_seq.size() << " sampled.\n"; 
        }

    }

    engine.updateProcessors("TrackSampleStopLogitsProcessor.active", 0);
    if (verbose) std::cout << "[SampleTracks] Deactivated track sample LogitsProcessor\n";

    if (verbose) std::cout << "[SampleTracks] Sampling finished\n";

}  


int _adapt_prompt_for_sampling(
    LibTok::MMM &tokenizer,
    int program,
    Controls &controls,
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::TokSequence &input_tokens,
    bool verbose
) {

    int num_context_bars = 8;

    if (verbose) {
        std::cout << "[AdaptPromptForTrackSample] Track Program=" << program
        << " Controls=[";
        for (auto &control : controls) {
            std::cout << control;
        }
        std::cout << "].\n";
    }

    for (int i = 0; i < token_seq.size(); i++) {

        auto bar_subseqs = token_seq[i].splitPerBars();
        int num_bars = bar_subseqs.size();
        int tokseq_len = token_seq[i].size();

        int context_start_idx = std::max(0, num_bars - num_context_bars);
        int context_end_idx = num_bars;
        
        if (context_start_idx != 0) {
            input_tokens += token_seq[i].slice(0, 2);
        }

        for (int bar_idx = context_start_idx; bar_idx < context_end_idx; bar_idx++) {
            input_tokens += bar_subseqs[bar_idx];
        }

    } 

    int last_track_start_idx = input_tokens.size();

    std::vector<std::string> new_track_tokens = {"Track_Start"};
    for (auto &control : controls) {
        new_track_tokens.push_back(control);
    }

    LibTok::TokSequence new_track_start_seq(new_track_tokens);
    tokenizer.complete_sequence(new_track_start_seq);

    input_tokens += new_track_start_seq;

    if (verbose) {
        std::cout << "[AdaptPromptForTrackSample] Final input_tokens size=" 
        << input_tokens.size() << "\n";
    } 

    return last_track_start_idx;

}

std::vector<std::string> extractSampledContent(
    const std::vector<std::string>& tokens,
    size_t last_track_start_idx,
    int num_bars_to_gen,
    int num_tracks_before,
    bool verbose
) {
    std::vector<std::string> track_tokens;

    // --- 1. Find all Track_Starts
    std::vector<size_t> track_starts;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == "Track_Start") {
            track_starts.push_back(i);
        }
    }

    if (track_starts.empty()) {
        throw std::runtime_error("[extractSampledContent] No Track_Start found in tokens.");
    }

    // --- 2. Verify track index
    size_t expected_track_idx = num_tracks_before; // 0-based
    if (expected_track_idx >= track_starts.size()) {
        throw std::runtime_error("[extractSampledContent] Not enough tracks generated. "
                                 "Expected at least " + std::to_string(num_tracks_before + 1));
    }

    size_t correct_start_idx = track_starts[expected_track_idx];
    if (last_track_start_idx != correct_start_idx) {
        if (verbose) {
            std::cerr << "[extractSampledContent] Warning: last_track_start_idx=" 
                      << last_track_start_idx 
                      << " does not match expected track_start at index="
                      << correct_start_idx
                      << ". Will take expected track.\n";
        }
    }
    last_track_start_idx = correct_start_idx;

    // --- 3. Extract from Track_Start to Track_End or next Track_Start
    size_t start = last_track_start_idx;

    // Find next Track_Start (after the current one)
    auto it_next_start = std::find(tokens.begin() + start + 1, tokens.end(), "Track_Start");

    // Find Track_End for this track
    auto it_end = std::find(tokens.begin() + start, 
                            (it_next_start != tokens.end() ? it_next_start : tokens.end()), 
                            "Track_End");

    if (it_end != tokens.end()) {
        // Found a Track_End before next Track_Start
        track_tokens.assign(tokens.begin() + start, it_end + 1);
    } else if (it_next_start != tokens.end()) {
        // No Track_End, but another Track_Start came → cut there, append Track_End manually
        track_tokens.assign(tokens.begin() + start, it_next_start);
        track_tokens.push_back("Track_End");
        if (verbose) {
            std::cerr << "[extractSampledContent] Warning: No Track_End found before next Track_Start, added one.\n";
        }
    } else {
        // No more Track_Start → take everything until end, ensure Track_End
        track_tokens.assign(tokens.begin() + start, tokens.end());
        if (track_tokens.empty() || track_tokens.back() != "Track_End") {
            track_tokens.push_back("Track_End");
            if (verbose) {
                std::cerr << "[extractSampledContent] Warning: No Track_End found, appended one.\n";
            }
        }
    }


    // --- 4. Verify bar count
    int bar_count = std::count(track_tokens.begin(), track_tokens.end(), "Bar_None");
    if (bar_count < num_bars_to_gen) {
        if (verbose) {
            std::cerr << "[extractSampledContent] Warning: Only " << bar_count
                      << " bars generated, expected " << num_bars_to_gen
                      << ". Filling with empty bars.\n";
        }
        for (int i = bar_count; i < num_bars_to_gen; ++i) {
            track_tokens.insert(track_tokens.end() - 1, "Bar_None");
        }
    } else if (bar_count > num_bars_to_gen) {
        if (verbose) {
            std::cerr << "[extractSampledContent] Warning: Generated " << bar_count
                      << " bars, expected " << num_bars_to_gen
                      << ". Trimming extras.\n";
        }
        int excess = bar_count - num_bars_to_gen;
        for (auto it = track_tokens.end(); it != track_tokens.begin() && excess > 0;) {
            --it;
            if (*it == "Bar_None") {
                it = track_tokens.erase(it);
                excess--;
            }
        }
    }

    // --- 5. Ensure ends with Track_End
    if (track_tokens.empty() || track_tokens.back() != "Track_End") {
        track_tokens.push_back("Track_End");
    }

    return track_tokens;
}



}
}