#include "inference.h"

#include <algorithm>

namespace mmm {

namespace inference {

void infill_bars(
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::MMM &tokenizer,
    const mmm::inference::BarInfilling& infill_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine,
    int context_length,
    mmm::utils::Logger& logger
) {

    logger.log(mmm::utils::LogLevel::INFO, "[InfillBars] Beginning infill");

    for (auto &[track_idx, subsets] : infill_config.bars) {
        logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Track " + std::to_string(track_idx) + ", subsets=" + std::to_string(subsets.size()) + "");
        for (auto &subset : subsets) {

            int start_bar_idx = std::get<0>(subset);
            int end_bar_idx = std::get<1>(subset);
            int num_bars_to_infill = end_bar_idx - start_bar_idx;

            logger.log(mmm::utils::LogLevel::DEBUG,  
                "[InfillBars] Subset start=" + std::to_string(start_bar_idx) 
                + " end=" + std::to_string(end_bar_idx) 
                + " bars_to_infill=" + std::to_string(num_bars_to_infill) + "" 
            );

            engine.updateProcessors("BarInfillStopLogitsProcessor.active", 1);
            logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Activated bar infilling LogitsProcessor");
            engine.updateProcessors("BarInfillStopLogitsProcessor.reset", 0);
            logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Reset LogitsProcessor");
            engine.updateProcessors("BarInfillStopLogitsProcessor.n_bars_to_infill", num_bars_to_infill);
            logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Reset LogitsProcessor bar infill count");

            if (model->is_cached()) {
                logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Resetting model cache");
                model->reset_cache();
            }

            LibTok::TokSequence input_tokens({}, {}, {}, {}, true);
            std::pair<int,int> token_indices = _adapt_prompt_for_infilling(
                tokenizer,
                track_idx,
                subset, 
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

            logger.log(mmm::utils::LogLevel::TRACE, "[InfillBars] Input Tokens:");
            for (auto tok : input_tokens.tokens) {
                logger.log(mmm::utils::LogLevel::TRACE, std::string("   ") + tok);
            } 

            logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Generating " + std::to_string(input_ids.size()) + " input ids");
            std::vector<int64_t> generated = engine.generate(input_ids, model, logger);
            logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Generation complete. Tokens=" + std::to_string(generated.size()) + "");

            int infill_token_start = std::get<0>(token_indices);
            int infill_token_end = std::get<1>(token_indices);

            std::vector<int> gen_ids;
            gen_ids.reserve(generated.size());
            for (auto v : generated) gen_ids.push_back(static_cast<int>(v));

            LibTok::TokSequence generated_seq({}, gen_ids, {}, {}, true);

            if (tokenizer.isTrained()) {
                logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Decoding token ids after generation");
                tokenizer.decodeTokenIds(generated_seq);
            }

            std::vector<std::string> tokens = generated_seq.tokens;

            auto fillbar_start_it = std::find(tokens.begin(), tokens.end(), "FillBar_Start");
            if (fillbar_start_it == tokens.end()) {
                logger.log(mmm::utils::LogLevel::ERROR, "[InfillBars] FillBar_Start token not found.");
                throw std::runtime_error("[InfillBars] FillBar_Start token not found.");
            }
            size_t fillbar_start_idx = std::distance(tokens.begin(), fillbar_start_it);
            logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Found FillBar_Start at idx=" + std::to_string(fillbar_start_idx) + ""); 

            logger.log(mmm::utils::LogLevel::TRACE, "[InfillBars] Generated Tokens:");
            for (auto it = fillbar_start_it; it != tokens.end(); ++it) {
                logger.log(mmm::utils::LogLevel::TRACE, std::string("   ") + *it);
            }

            std::vector<std::string> infilled_content = extractInfilledContent(tokens, fillbar_start_idx, num_bars_to_infill, logger);

            logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Extracted infilled_content size=" + std::to_string(infilled_content.size()) + ""); 

            LibTok::TokSequence infilled_seq(infilled_content);
            tokenizer.complete_sequence(infilled_seq);

            logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Infilled sequence complete. Length=" + std::to_string(infilled_seq.size()) + ""); 

            auto before_seq = token_seq[track_idx].slice(0, infill_token_start);
            auto infilled_seq_copy = infilled_seq; // just to emphasize it's separate
            auto after_seq  = token_seq[track_idx].sliceUntilEnd(infill_token_end);

            // then concatenate as usual
            token_seq[track_idx] = before_seq + infilled_seq_copy + after_seq;
            
            logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Track " + std::to_string(track_idx) + " updated with infilled sequence."); 

        }
    }
    engine.updateProcessors("BarInfillStopLogitsProcessor.active", 0);
    logger.log(mmm::utils::LogLevel::DEBUG, "[InfillBars] Deactivated bar infilling LogitsProcessor");

    logger.log(mmm::utils::LogLevel::INFO, "[InfillBars] Infill finished");

}

std::pair<int,int> _adapt_prompt_for_infilling(
    LibTok::MMM &tokenizer,
    int track_idx, 
    const BarSubset &subset,
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::TokSequence &input_tokens,
    int context_length,
    mmm::utils::Logger& logger
) {
    
    int start_bar_idx = std::get<0>(subset);
    int end_bar_idx = std::get<1>(subset);

    logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Track=" + std::to_string(track_idx) 
        + " start_bar=" + std::to_string(start_bar_idx) + " end_bar=" 
        + std::to_string(end_bar_idx) + ""); 

    if (tokenizer.isTrained()) {
        tokenizer.decodeTokenIds(token_seq[track_idx]);
    }
    // We split by bars, ignoring Track_Start and Program tokens
    std::vector<LibTok::TokSequence> bar_subseqs_infill = token_seq[track_idx].sliceUntilEnd(2).splitPerBars();
    if (start_bar_idx >= end_bar_idx) {
        throw std::runtime_error("[AdaptPromptForBarInfill] Start bar of infilling must be before end bar. Received start=" + std::to_string(start_bar_idx) + " and end=" + std::to_string(end_bar_idx));
    }
    if (start_bar_idx < 0 || start_bar_idx >= bar_subseqs_infill.size()) {
        throw std::runtime_error("[AdaptPromptForBarInfill] Start bar for infilling must be included in [0," + std::to_string(bar_subseqs_infill.size() - 1) + "].");
    }
    if (end_bar_idx < 1 || end_bar_idx > bar_subseqs_infill.size()) {
        throw std::runtime_error("[AdaptPromptForBarInfill] End bar for infilling must be included in [1," + std::to_string(bar_subseqs_infill.size()) + "].");
    }
    int num_bars = bar_subseqs_infill.size();
    logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Track " + std::to_string(track_idx) + " has " + std::to_string(num_bars) + " bars"); 

    int context_start_idx = std::max(0, start_bar_idx - context_length);
    int context_end_idx = std::min(end_bar_idx + context_length, num_bars );
    logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Context range: " 
        + std::to_string(context_start_idx) + " -> " + std::to_string(context_end_idx) + "");
    LibTok::TokSequence seq_to_infill({}, {}, {}, {}, false);
    // We start of Track_Start and Program tokens
    seq_to_infill += token_seq[track_idx].slice(0,2);
    int infill_start_token_idx = 2;
    for (int bar_idx = context_start_idx; bar_idx < start_bar_idx; bar_idx++) {
        seq_to_infill += bar_subseqs_infill[bar_idx];
        infill_start_token_idx += bar_subseqs_infill[bar_idx].size();
    }
    logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Infill start token index=" + std::to_string(infill_start_token_idx) + "");
    int infill_end_token_idx = infill_start_token_idx;
    for (int bar_idx = start_bar_idx; bar_idx < end_bar_idx; bar_idx++) {
        LibTok::TokSequence infill_bar_seq({"Infill_Bar"});
        tokenizer.complete_sequence(infill_bar_seq);
        seq_to_infill += infill_bar_seq;
        infill_end_token_idx += bar_subseqs_infill[bar_idx].size();
    }
    logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Infill end token index=" + std::to_string(infill_end_token_idx) + "");
    for (int bar_idx = end_bar_idx; bar_idx < context_end_idx; bar_idx++) {
        seq_to_infill += bar_subseqs_infill[bar_idx];
    }
    if (tokenizer.isTrained()) {
        logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Encoding sequence to infill.");
        tokenizer.encodeTokenIds(seq_to_infill);
    } else {
        logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Not encoding infilling sequence, no BPE.");
    }

    logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Iterating tracks.");
    for (int i = 0; i < token_seq.size(); i++) {
        if (i == track_idx) {
            logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Track " + std::to_string(i) + " to infill.");
            input_tokens += seq_to_infill;
        } else {
            LibTok::TokSequence current_track_seq = token_seq[i];
            logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Track " + std::to_string(i) + " for context.");
            if (tokenizer.isTrained()) {
                tokenizer.encodeTokenIds(current_track_seq);
            }
            auto bar_subseqs = current_track_seq.splitPerBars();
            int num_bars = bar_subseqs.size();
            int tokseq_len = current_track_seq.size();

            input_tokens += current_track_seq.slice(0, 2);

            if (num_bars < context_end_idx || num_bars < context_start_idx) {
                throw std::runtime_error("[AdaptPromptForBarInfill] Context error. Track " + std::to_string(i) + " has " + std::to_string(num_bars) + " bars, but context is from bar " + std::to_string(context_start_idx) + " to " + std::to_string(context_end_idx) + ".");
            }

            for (int bar_idx = context_start_idx; bar_idx < context_end_idx; bar_idx++) {
                input_tokens += bar_subseqs[bar_idx];
            }
            
            if (context_end_idx != num_bars) {
                logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Clipping for context ( context end " + std::to_string(context_end_idx) + " < last bar " + std::to_string(num_bars) + " ).");
                input_tokens += current_track_seq.sliceUntilEnd(tokseq_len - 1);
            }
        }
    }

    LibTok::TokSequence fill_bar_start_seq({"FillBar_Start"});
    tokenizer.complete_sequence(fill_bar_start_seq);
    input_tokens += fill_bar_start_seq;

    Controls attribute_controls = std::get<2>(subset);
    LibTok::TokSequence control_seq(attribute_controls);
    tokenizer.complete_sequence(control_seq);
    input_tokens += control_seq;

    LibTok::TokSequence first_bar_start_seq({"Bar_None"});
    tokenizer.complete_sequence(first_bar_start_seq);
    input_tokens += first_bar_start_seq;

    logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Infill token range: " 
        + std::to_string(infill_start_token_idx) + " → " 
        + std::to_string(infill_end_token_idx) + "");
        
    logger.log(mmm::utils::LogLevel::DEBUG, "[AdaptPromptForBarInfill] Final input_tokens size=" 
        + std::to_string(input_tokens.size()) + "");

    return std::pair<int,int>(infill_start_token_idx, infill_end_token_idx);
}

std::vector<std::string> extractInfilledContent(
    const std::vector<std::string>& tokens,
    size_t fillbar_start_idx,
    int num_bars_to_infill,
    mmm::utils::Logger& logger
) {
    std::vector<std::string> infill_tokens;
    std::vector<std::vector<std::string>> bars;

    logger.log(mmm::utils::LogLevel::DEBUG,
        "[InfillExtraction] Starting extraction after FillBar_Start idx=" +
        std::to_string(fillbar_start_idx) +
        ", num_bars_to_infill=" + std::to_string(num_bars_to_infill)
    );

    if (fillbar_start_idx >= tokens.size() - 1) {
        logger.log(mmm::utils::LogLevel::WARN,
            "[InfillExtraction] No tokens after FillBar_Start. Returning Bar_None placeholders."
        );
        return std::vector<std::string>(num_bars_to_infill, "Bar_None");
    }

    // ---------------------------------------------------------------------
    // Parse bars sequentially
    // ---------------------------------------------------------------------
    std::vector<std::string> current_bar;
    bool inside_bar = false;

    for (size_t i = fillbar_start_idx + 1; i < tokens.size(); ++i) {
        const auto& tok = tokens[i];

        if (tok == "Bar_None") {
            // If we're already inside a bar, that means this starts a new one.
            if (inside_bar) {
                bars.push_back(std::move(current_bar));
                current_bar.clear();
            }
            inside_bar = true;
            current_bar.push_back(tok);
        }
        else if (tok == "FillBar_End") {
            // End the current bar and finish.
            if (inside_bar) {
                current_bar.push_back(tok);
                bars.push_back(std::move(current_bar));
                current_bar.clear();
                inside_bar = false;
            }
            logger.log(mmm::utils::LogLevel::DEBUG, "[InfillExtraction] FillBar_End reached.");
            break;
        }
        else {
            // Regular token — append to current bar if we’re inside one.
            if (inside_bar) current_bar.push_back(tok);
        }
    }

    // Add any trailing bar that didn't end in FillBar_End
    if (inside_bar && !current_bar.empty()) {
        bars.push_back(std::move(current_bar));
    }

    logger.log(mmm::utils::LogLevel::DEBUG,
        "[InfillExtraction] Found " + std::to_string(bars.size()) +
        " bar(s) after FillBar_Start."
    );

    // ---------------------------------------------------------------------
    // Handle insufficient/excess bars
    // ---------------------------------------------------------------------
    int generated_bars = static_cast<int>(bars.size());
    int produced_bars = std::min(num_bars_to_infill, generated_bars);

    if (produced_bars < generated_bars) {
        logger.log(mmm::utils::LogLevel::WARN,
            "[InfillExtraction] Model generated too many bars (" +
            std::to_string(generated_bars) + "). Expected " +
            std::to_string(num_bars_to_infill) + "."
        );
    } else if (produced_bars > generated_bars) {
        logger.log(mmm::utils::LogLevel::WARN,
            "[InfillExtraction] Model generated too few bars (" +
            std::to_string(generated_bars) + "). Expected " +
            std::to_string(num_bars_to_infill) + "."
        );
    }

    // Take the first `produced_bars`
    for (int i = 0; i < produced_bars; ++i) {
        infill_tokens.insert(infill_tokens.end(), bars[i].begin(), bars[i].end());
    }

    // If missing, append empty bars
    int missing = num_bars_to_infill - produced_bars;
    if (missing > 0) {
        logger.log(mmm::utils::LogLevel::WARN,
            "[InfillExtraction] Only " + std::to_string(produced_bars) +
            " bars produced; appending " + std::to_string(missing) +
            " empty Bar_None(s)."
        );

        for (int i = 0; i < missing; ++i) {
            infill_tokens.push_back("Bar_None");
        }
    }

    logger.log(mmm::utils::LogLevel::DEBUG,
        "[InfillExtraction] Extraction complete. Total infill tokens=" +
        std::to_string(infill_tokens.size())
    );

    return infill_tokens;
}


}
}