#include "inference.h"

#include <algorithm>

namespace mmm::inference {

mmm::sampling::SamplingEngine createEngine(
    mmm::sampling::GenerationConfig& config,
    LibTok::MMM &tokenizer,
    int seed
) {

    mmm::sampling::Sampler sampler;
    if (seed > 0) {
        sampler = mmm::sampling::Sampler(static_cast<std::mt19937::result_type>(seed));
    }

    auto vocab_var = tokenizer.getVocab();
    auto* vocab_ptr = std::get_if<std::map<std::string,int>>(&vocab_var);
    if (!vocab_ptr) {
        throw std::runtime_error("Tokenizer vocab is not a StringIntMap");
    }
    auto& vocab = *vocab_ptr;

    LogitsProcessorList processors = mmm::sampling::createProcessorListFromConfig(
        config,
        vocab["EOS_None"],
        vocab["FillBar_Start"],
        vocab["Bar_None"]
    );

    mmm::sampling::LogitsWarperList warpers = mmm::sampling::createWarperListFromConfig(config);

    mmm::utils::Profiler profiler;

    mmm::sampling::SamplingEngine engine(
        config, 
        processors, 
        warpers, 
        sampler, 
        &profiler
    );

    return engine;
}

LibTok::ScoreType generate(
    mmm::IModel* model, 
    LibTok::MMM &tokenizer,
    PromptConfig &cfg,
    mmm::sampling::SamplingEngine &engine,
    LibTok::ScoreType &score
) {

    auto tokens = tokenizer.encode(score, true, false, {}, false);
    std::vector<LibTok::TokSequence> token_seq = std::get<std::vector<LibTok::TokSequence>>(tokens);

    if (cfg.bar_infilling()) {
        if (cfg.empty()) {
            throw std::invalid_argument("Bar Infilling mode requested but no bar subset provided.");
        }
        infill_bars(token_seq, tokenizer, std::get<mmm::inference::BarInfilling>(cfg.mode), model, engine);
        // convert back to score
        Score result = tokenizer.tokens_to_score(token_seq);
        return result;
    }

    if (cfg.track_infilling()) {
        if (cfg.empty()) {
            throw std::invalid_argument("Track infilling mode requested but no track provided.");
        }
        throw std::invalid_argument("Track infilling is not yet supported.");
    }

    if (cfg.track_sampling()) {
        if (cfg.empty()) {
            throw std::invalid_argument("Track sampling mode requested but no track provided.");
        }
        throw std::invalid_argument("Track sampling is not yet supported.");
    }

    return score;
}

void infill_bars(
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::MMM &tokenizer,
    mmm::inference::BarInfilling &infill_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine
) {
    
    std::vector<int> tracks_to_infill;

    for (auto &[track_idx, subsets] : *infill_config.bars) {
        tracks_to_infill.push_back(track_idx);
    }

    for (auto track_idx : tracks_to_infill) {
        for (auto &subset : subsets) {

            int start_bar_idx = std::get<0>(subset);
            int end_bar_idx = std::get<1>(subset);
            int num_bars_to_infill = end_bar_idx - start_bar_idx;

            engine.updateProcessors("InfillStopLogitsProcessor.reset", 0);
            engine.updateProcessors("InfillStopLogitsProcessor.n_bars_to_infill", num_bars_to_infill);
            engine.updateProcessors("InfillStopLogitsProcessor.n_attribute_controls", std::get<2>(subset).size());

            LibTok::TokSequence input_tokens({}, {}, {}, {}, true);
            std::pair<int,int> token_indices = _adapt_prompt_for_infilling(
                tokenizer,
                track_idx,
                subset, 
                token_seq, 
                input_tokens
            );

            std::vector<int64_t> input_ids;
            input_ids.reserve(input_tokens.size());

            for (int id : input_tokens.ids) {
                input_ids.push_back(static_cast<int64_t>(id));
            }

            if (model->is_cached()) {
                model->reset_cache();
            }

            std::vector<int64_t> generated = engine.generate(input_ids, model);

            int infill_token_start = std::get<0>(token_indices);
            int infill_token_end = std::get<1>(token_indices);

            LibTok::TokSequence generated_seq({}, generated, {}, {}, true);
            generated_seq.complete_sequence();
            tokenizer.decodeTokenIds(generated_seq);

            std::vector<std::string> tokens = generated_seq.tokens;

            auto fillbar_start_it = std::find(tokens.begin(), tokens.end(), "FillBar_Start");
            if (fillbar_start_it == tokens.end()) {
                throw std::runtime_error("FillBar_Start token not found.");
            }
            size_t fillbar_start_idx = std::distance(tokens.begin(), fillbar_start_it);

            // Find first Bar_None after FillBar_Start
            auto first_bar_none_it = std::find(tokens.begin() + fillbar_start_idx + 1, tokens.end(), "Bar_None");
            if (first_bar_none_it == tokens.end()) {
                throw std::runtime_error("No Bar_None token found after FillBar_Start.");
            }
            size_t first_bar_none_idx = std::distance(tokens.begin(), first_bar_none_it);

            // Find last Bar_None after FillBar_Start using reverse iterators
            auto last_bar_none_rit = std::find(tokens.rbegin(), std::make_reverse_iterator(tokens.begin() + fillbar_start_idx), "Bar_None");
            if (last_bar_none_rit == tokens.rend()) {
                throw std::runtime_error("Last Bar_None not found.");
            }
            size_t last_bar_none_idx = std::distance(tokens.begin(), last_bar_none_rit.base()) - 1;

            if (last_bar_none_idx <= first_bar_none_idx) {
                throw std::runtime_error("Only one Bar_None found or invalid ordering.");
            }

            // Count Bar_None tokens between FillBar_Start and end
            int bar_none_count = std::count(tokens.begin() + fillbar_start_idx + 1, tokens.end(), "Bar_None");
            if (bar_none_count != num_bars_to_infill + 1) {
                throw std::runtime_error(
                    "Expected " + std::to_string(num_bars_to_infill + 1) + " Bar_None tokens after FillBar_Start, but found " + std::to_string(bar_none_count)
                );
            }

            // Extract infilled content between first and last Bar_None (exclusive)
            std::vector<std::string> infilled_content(
                tokens.begin() + first_bar_none_idx,
                tokens.begin() + last_bar_none_idx
            );

            LibTok::TokSequence infilled_seq(infilled_content);
            infilled_seq.complete_sequence();

            token_seq[track_idx] = token_seq[track_idx].slice(0, infill_token_start) + infilled_seq + token_seq[track_idx].sliceUntilEnd(infill_token_end);
            
        }
    }

}

std::pair<int,int> _adapt_prompt_for_infilling(
    LibTok::MMM &tokenizer,
    int track_idx, 
    BarSubset &subset,
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::TokSequence &input_tokens
) {
    int num_context_bars = 8;
    
    int start_bar_idx = std::get<0>(subset);
    int end_bar_idx = std::get<1>(subset);

    tokenizer.decodeTokenIds(token_seq[track_idx]);
    std::vector<LibTok::TokSequence> bar_subseqs_infill = token_seq[track_idx].splitPerBars();
    int num_bars = bar_subseqs_infill.size();
    int context_start_idx = std::max(0, start_bar_idx - num_context_bars);
    int context_end_idx = std::min(start_bar_idx + num_context_bars, num_bars );
    LibTok::TokSequence seq_to_infill({}, {}, {}, {}, false);
    int infill_start_token_idx = 0;
    for (int bar_idx = context_start_idx; bar_idx < start_bar_idx; bar_idx++) {
        seq_to_infill += bar_subseqs_infill[bar_idx];
        infill_start_token_idx += bar_subseqs_infill[bar_idx].size();
    }
    int infill_end_token_idx = infill_start_token_idx;
    for (int bar_idx = start_bar_idx; bar_idx < end_bar_idx; bar_idx++) {
        LibTok::TokSequence infill_bar_seq({"Infill_Bar"});
        infill_bar_seq.complete_sequence();
        seq_to_infill += infill_bar_seq;
        infill_end_token_idx += bar_subseqs_infill[bar_idx].size();
    }
    for (int bar_idx = end_bar_idx; bar_idx < context_end_idx; bar_idx++) {
        seq_to_infill += bar_subseqs_infill[bar_idx];
    }
    tokenizer.encodeTokenIds(seq_to_infill);

    for (int i = 0; i < token_seq.size(); i++) {
        if (i == track_idx) {
            input_tokens += seq_to_infill;
        } else {
            std::vector<LibTok::TokSequence> bar_subseqs = token_seq[i].splitPerBars();
            int tokseq_len = token_seq[i].size();
            input_tokens += token_seq[i].slice(0,2);
            for (int bar_idx = context_start_idx; bar_idx < context_end_idx; bar_idx++) {
                int tokseq_len = token_seq[i].size();
                input_tokens += bar_subseqs[bar_idx];
            }
            input_tokens += token_seq[i].sliceUntilEnd(tokseq_len - 1);
        }
    }

    for (int i = 0; i < token_seq.size(); i++) {
        if (i == track_idx) {
            input_tokens += seq_to_infill;
        } else {
            auto bar_subseqs = token_seq[i].splitPerBars();
            int num_bars = bar_subseqs.size();
            int tokseq_len = token_seq[i].size();

            input_tokens += token_seq[i].slice(0, 2);

            if (num_bars < context_end_idx || num_bars < context_start_idx) {
                throw std::runtime_error("Context error. Track " + std::to_string(i) + " has " + std::to_string(num_bars) + " bars, but context is from bar " + std::to_string(context_start_idx) + " to " + std::to_string(context_end_idx) + ".");
            }

            for (int bar_idx = context_start_idx; bar_idx < context_end_idx; bar_idx++) {
                input_tokens += bar_subseqs[bar_idx];
            }

            input_tokens += token_seq[i].sliceUntilEnd(tokseq_len - 1);
        }
    }

    LibTok::TokSequence fill_bar_start_seq({"FillBar_Start"});
    fill_bar_start_seq.complete_sequence();
    input_tokens += fill_bar_start_seq;

    Control attribute_controls = std::get<2>(subset);
    LibTok::TokSequence control_seq(attribute_controls);
    control_seq.complete_sequence();
    input_tokens += control_seq;

    return std::pair<int,int>(infill_start_token_idx, infill_end_token_idx);
}

}