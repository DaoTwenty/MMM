#include "inference.h"

#include <algorithm>

namespace mmm {

namespace inference {

mmm::sampling::SamplingEngine createEngine( 
    mmm::sampling::GenerationConfig& config, 
    LibTok::MMM &tokenizer, 
    int seed, 
    bool verbose 
) {

    if (verbose) { 
        std::cout << "[Engine] Creating engine with seed=" << seed << "\n"; 
    } 

    mmm::sampling::Sampler sampler; 
    
    if (seed > 0) { 
        if (verbose) std::cout << "[Engine] Using fixed seed\n"; 
        sampler = mmm::sampling::Sampler(static_cast<std::mt19937::result_type>(seed)); 
    } 
    
    auto vocab_var = tokenizer.getVocab(); 
    auto* vocab_ptr = std::get_if<std::unordered_map<std::string,int>>(&vocab_var);

    if (!vocab_ptr) { 
        throw std::runtime_error("Tokenizer vocab is not an unordered_map<string,int>"); 
    } 
    auto& vocab = *vocab_ptr;

    if (verbose) std::cout << "[Engine] Vocab loaded. Size=" << vocab.size() << "\n"; 
    
    mmm::sampling::LogitsProcessorList processors = mmm::utils::createProcessorListFromConfig( config, vocab["EOS_None"], vocab["FillBar_Start"], vocab["Bar_None"], vocab["Track_Start"], vocab["Track_End"] ); 

    if (verbose) std::cout << "[Engine] Processors created\n"; 
    
    mmm::sampling::LogitsWarperList warpers = mmm::utils::createWarperListFromConfig(config); 
    
    if (verbose) std::cout << "[Engine] Warpers created\n";

    bool profiler = true; 
    
    if (verbose) std::cout << "[Engine] Profiler attached\n"; 

    mmm::sampling::SamplingEngine engine( config, vocab["EOS_None"], tokenizer.getVocabSize(), processors, warpers, sampler, profiler );
    
    if (verbose) std::cout << "[Engine] Engine created successfully\n"; 
    
    return engine; }

LibTok::ScoreType generate(
    mmm::IModel* model, 
    LibTok::MMM &tokenizer,
    PromptConfig &cfg,
    mmm::sampling::SamplingEngine &engine,
    LibTok::ScoreType &score,
    bool verbose
) {

    if (verbose) std::cout << "[Generate] Starting generation\n";

    auto tokens = tokenizer.encode(score, true, false, {}, false);
    std::vector<LibTok::TokSequence> token_seq = std::get<std::vector<LibTok::TokSequence>>(tokens);

    _preprocess_token_sequence_vector(token_seq, true, verbose);

    if (cfg.bar_infilling()) {
        if (cfg.empty()) {
            throw std::invalid_argument("Bar Infilling mode requested but no bar subset provided.");
        }
        if (verbose) std::cout << "[Generate] Running bar infilling\n";
        if (auto barCfg = std::get_if<mmm::inference::BarInfilling>(&cfg.mode)) {
            infill_bars(token_seq, tokenizer, *barCfg, model, engine, verbose);
        } else {
            throw std::runtime_error("[Generate] Given config does not hold BarInfilling variant");
        }

    } else if (cfg.track_infilling()) {
        if (cfg.empty()) {
            throw std::invalid_argument("Track infilling mode requested but no track provided.");
        }
        throw std::invalid_argument("Track infilling is not yet supported.");
    } else if (cfg.track_sampling()) {
        if (cfg.empty()) {
            throw std::invalid_argument("Track sampling mode requested but no track provided.");
        }
        if (verbose) std::cout << "[Generate] Running bar infilling\n";
        if (auto barCfg = std::get_if<mmm::inference::TrackSampling>(&cfg.mode)) {
            sample_tracks(token_seq, tokenizer, *barCfg, model, engine, verbose);
        } else {
            throw std::runtime_error("[Generate] Given config does not hold TrackSampling variant");
        }
    } else {

        if (verbose) std::cout << "[Generate] No mode matched, returning input score\n";
        return score;
    }

    LibTok::ScoreType result;

    std::vector<int> res;
    size_t total_size = 0;
    for (const auto& seq : token_seq) total_size += seq.ids.size();
    res.reserve(total_size);

    for (const auto& seq : token_seq) {
        res.insert(res.end(), seq.ids.begin(), seq.ids.end());
    }
    try {
        result = tokenizer.decode(res);
    } catch (const std::exception& e) {
        std::cerr << "[Generate] Error in decode: " << e.what() << "\n";
        throw;
    }
    if (verbose) std::cout << "[Generate] Generation complete\n";
    return result;
}

void _preprocess_token_sequence_vector(
    std::vector<LibTok::TokSequence> &token_seq,  // pass by ref!
    bool pad,
    bool verbose
) {
    if (verbose) {
        std::cout << "[Preprocess] Preprocessing token sequence\n";
    }

    // --- 1. Split per bars for each track
    std::vector<std::vector<LibTok::TokSequence>> track_bars(token_seq.size());
    for (size_t track = 0; track < token_seq.size(); ++track) {
        track_bars[track] = token_seq[track].splitPerBars();
    }

    // --- 2. Find min and max number of bars
    size_t min_bars = SIZE_MAX;
    size_t max_bars = 0;
    for (const auto &bars : track_bars) {
        min_bars = std::min(min_bars, bars.size());
        max_bars = std::max(max_bars, bars.size());
    }

    if (verbose) {
        std::cout << "[Preprocess]  Found min_bars=" << min_bars
                  << ", max_bars=" << max_bars << "\n";
    }

    if (min_bars == max_bars) return;

    // --- 3. Normalize each track
    for (size_t track = 0; track < track_bars.size(); ++track) {
        auto &bars = track_bars[track];

        if (pad) {
            // Pad to max_bars
            while (bars.size() < max_bars) {
                LibTok::TokSequence empty_bar_seq({"Bar_None"});
                bars.push_back(empty_bar_seq);
            }
        } else {
            // Truncate to min_bars
            if (bars.size() > min_bars) {
                bars.resize(min_bars);
            }
        }
    }

    // --- 4. Reconstruct the token_seq tracks from bars
    for (size_t track = 0; track < token_seq.size(); ++track) {
        LibTok::TokSequence rebuilt;
        for (const auto &bar : track_bars[track]) {
            rebuilt += bar;
        }
        token_seq[track] = std::move(rebuilt);
    }

    if (verbose) {
        std::cout << "[Preprocess]  Normalized all tracks to "
                  << (pad ? max_bars : min_bars) << " bars\n";
    }
}

}
}