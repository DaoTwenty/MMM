#include "inference.h"

#include <algorithm>

namespace mmm {
namespace inference {

mmm::sampling::SamplingEngine createEngine( 
    mmm::sampling::GenerationConfig& config, 
    LibTok::MMM &tokenizer, 
    mmm::utils::Logger& logger,
    int seed
) {
    logger.log(mmm::utils::LogLevel::INFO, "[Engine] Creating engine with seed=" + std::to_string(seed));

    mmm::sampling::Sampler sampler; 
    
    if (seed > 0) { 
        logger.log(mmm::utils::LogLevel::DEBUG, "[Engine] Using fixed seed");
        sampler = mmm::sampling::Sampler(seed); 
    } 
    
    auto vocab_var = tokenizer.getVocab(); 
    auto* vocab_ptr = std::get_if<std::unordered_map<std::string,int>>(&vocab_var);

    if (!vocab_ptr) { 
        logger.log(mmm::utils::LogLevel::FATAL, "Tokenizer vocab is not an unordered_map<string,int>");
        throw std::runtime_error("Tokenizer vocab is not an unordered_map<string,int>"); 
    } 
    auto& vocab = *vocab_ptr;

    logger.log(mmm::utils::LogLevel::INFO, "[Engine] Vocab loaded. Size=" + std::to_string(vocab.size()));

    mmm::utils::SpecialTokens special_tokens(vocab);

    std::unordered_map<int, bool> contains_bar;
    std::unordered_map<int, bool> contains_track_end;

    mmm::utils::buildContainMaps(
        tokenizer,
        vocab.size(),
        special_tokens,
        logger,
        contains_bar,
        contains_track_end
    );

    
    mmm::sampling::LogitsProcessorList processors = 
        mmm::utils::createProcessorListFromConfig(config, special_tokens, contains_bar, contains_track_end); 

    logger.log(mmm::utils::LogLevel::DEBUG, "[Engine] Processors created");
    
    mmm::sampling::LogitsWarperList warpers = 
        mmm::utils::createWarperListFromConfig(config); 
    
    logger.log(mmm::utils::LogLevel::DEBUG, "[Engine] Warpers created");

    bool profiler = true; 
    
    logger.log(mmm::utils::LogLevel::DEBUG, "[Engine] Profiler attached");

    mmm::sampling::SamplingEngine engine(
        config, 
        special_tokens.eos_none, 
        tokenizer.getVocabSize(), 
        processors, 
        warpers, 
        sampler, 
        profiler
    );
    
    logger.log(mmm::utils::LogLevel::INFO, "[Engine] Engine created successfully");
    
    return engine;
}

LibTok::ScoreType generate(
    mmm::IModel* model, 
    LibTok::MMM &tokenizer,
    PromptConfig &cfg,
    mmm::sampling::SamplingEngine &engine,
    LibTok::ScoreType &score,
    mmm::utils::Logger& logger
) {
    logger.log(mmm::utils::LogLevel::INFO, "[Generate] Starting generation.");

    int attempts = engine.getConfig().attempts;
    if (attempts <= 0) attempts = 1;
    logger.log(mmm::utils::LogLevel::INFO, "[Generate] Number of attempts = " + std::to_string(attempts));

    for (int attempt = 1; attempt <= attempts; ++attempt) {
        logger.log(mmm::utils::LogLevel::DEBUG, "[Generate] Attempt " + std::to_string(attempt) + " / " + std::to_string(attempts));

        try {
            // Initial encoding from the input score
            auto tokens = tokenizer.encodeWithoutConcatenation(score, false, false, {});
            logger.log(mmm::utils::LogLevel::DEBUG, "[Generate] Encoding MIDI file.");

            std::vector<LibTok::TokSequence> token_seq =
                std::get<std::vector<LibTok::TokSequence>>(tokens);

            _preprocess_token_sequence_vector(token_seq, logger, true);

            // --- Perform bar infilling or track sampling ---
            if (cfg.bar_infilling()) {
                if (cfg.empty()) {
                    throw std::invalid_argument("Bar Infilling mode requested but no bar subset provided.");
                }
                logger.log(mmm::utils::LogLevel::INFO, "[Generate] Running bar infilling");

                if (auto barCfg = std::get_if<mmm::inference::BarInfilling>(&cfg.mode)) {
                    infill_bars(token_seq, tokenizer, *barCfg, model, engine, cfg.context_length, logger);

                    std::vector<int> res;
                    for (const auto& seq : token_seq)
                        res.insert(res.end(), seq.ids.begin(), seq.ids.end());
                    score = tokenizer.decode(res);

                    tokens = tokenizer.encodeWithoutConcatenation(score, false, false, {});
                    token_seq = std::get<std::vector<LibTok::TokSequence>>(tokens);
                    _preprocess_token_sequence_vector(token_seq, logger, true);
                } else {
                    throw std::runtime_error("[Generate] Given config does not hold BarInfilling variant");
                }

            } else if (cfg.track_sampling()) {
                if (cfg.empty()) {
                    throw std::invalid_argument("Track sampling mode requested but no track provided.");
                }
                logger.log(mmm::utils::LogLevel::INFO, "[Generate] Running track sampling");

                if (auto trackCfg = std::get_if<mmm::inference::TrackSampling>(&cfg.mode)) {
                    sample_tracks(token_seq, tokenizer, *trackCfg, model, engine, cfg.context_length, logger);

                    std::vector<int> res;
                    for (const auto& seq : token_seq)
                        res.insert(res.end(), seq.ids.begin(), seq.ids.end());
                    score = tokenizer.decode(res);

                    tokens = tokenizer.encodeWithoutConcatenation(score, false, false, {});
                    token_seq = std::get<std::vector<LibTok::TokSequence>>(tokens);
                    _preprocess_token_sequence_vector(token_seq, logger, true);
                } else {
                    throw std::runtime_error("[Generate] Given config does not hold TrackSampling variant");
                }

            } else {
                logger.log(mmm::utils::LogLevel::WARN, "[Generate] No mode matched, returning input score");
                return score;
            }

            // --- Final flatten & decode ---
            std::vector<int> final_res;
            for (const auto& seq : token_seq)
                final_res.insert(final_res.end(), seq.ids.begin(), seq.ids.end());

            logger.log(mmm::utils::LogLevel::TRACE, "[Generate] Total tokens to decode = " + std::to_string(final_res.size()));

            LibTok::ScoreType result = tokenizer.decode(final_res);

            logger.log(mmm::utils::LogLevel::INFO, "[Generate] Generation complete on attempt " + std::to_string(attempt));
            return result;

        } catch (const std::exception& e) {
            logger.log(mmm::utils::LogLevel::ERROR, "[Generate] Attempt " + std::to_string(attempt) + " failed: " + e.what());
            if (attempt < attempts)
                logger.log(mmm::utils::LogLevel::WARN, "[Generate] Retrying...");
            else {
                logger.log(mmm::utils::LogLevel::FATAL, "[Generate] All attempts failed.");
                throw;
            }
        }
    }

    logger.log(mmm::utils::LogLevel::FATAL, "[Generate] Reached unreachable state (no successful attempts)");
    throw std::runtime_error("[Generate] Reached unreachable state (no successful attempts)");
}



void _preprocess_token_sequence_vector(
    std::vector<LibTok::TokSequence> &token_seq,
    mmm::utils::Logger& logger,
    bool pad
) {
    logger.log(mmm::utils::LogLevel::DEBUG, "[Preprocess] Preprocessing token sequence");

    std::vector<std::vector<LibTok::TokSequence>> track_bars(token_seq.size());
    for (size_t track = 0; track < token_seq.size(); ++track) {
        logger.log(mmm::utils::LogLevel::TRACE, "  [Track " + std::to_string(track) + "] size=" + std::to_string(token_seq[track].size()) +
                                    " tokens before split (tokens=" + std::to_string(token_seq[track].tokens.size()) +
                                    ", ids=" + std::to_string(token_seq[track].ids.size()) + ").");

        auto bars = token_seq[track].splitPerBars();
        track_bars[track] = std::move(bars);
    }

    size_t min_bars = SIZE_MAX;
    size_t max_bars = 0;
    for (const auto &bars : track_bars) {
        min_bars = std::min(min_bars, bars.size());
        max_bars = std::max(max_bars, bars.size());
    }

    logger.log(mmm::utils::LogLevel::DEBUG, "[Preprocess] Found min_bars=" + std::to_string(min_bars) +
                                ", max_bars=" + std::to_string(max_bars));

    if (min_bars == max_bars) return;

    for (size_t track = 0; track < track_bars.size(); ++track) {
        auto &bars = track_bars[track];
        if (pad) {
            while (bars.size() < max_bars)
                bars.emplace_back(LibTok::TokSequence({"Bar_None"}));
        } else if (bars.size() > min_bars) {
            bars.resize(min_bars);
        }
    }

    for (size_t track = 0; track < token_seq.size(); ++track) {
        LibTok::TokSequence rebuilt;
        for (const auto &bar : track_bars[track]) rebuilt += bar;
        token_seq[track] = std::move(rebuilt);
    }

    logger.log(mmm::utils::LogLevel::DEBUG, "[Preprocess] Normalized all tracks to " +
                                std::to_string(pad ? max_bars : min_bars) + " bars");
}

} // namespace inference
} // namespace mmm
