#include "inference.h"

#include <algorithm>

namespace mmm {

namespace inference {

void infill_bars(
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::MMM &tokenizer,
    mmm::inference::BarInfilling &infill_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine,
    int context_length,
    bool verbose
) {

    if (verbose) std::cout << "[InfillBars] Beginning infill\n";

    for (auto &[track_idx, subsets] : infill_config.bars) {
        if (verbose) std::cout << "[InfillBars] Track " << track_idx << ", subsets=" << subsets.size() << "\n";
        for (auto &subset : subsets) {

            int start_bar_idx = std::get<0>(subset);
            int end_bar_idx = std::get<1>(subset);
            int num_bars_to_infill = end_bar_idx - start_bar_idx;

            if (verbose) { 
                std::cout << "[InfillBars] Subset start=" << start_bar_idx 
                << " end=" << end_bar_idx 
                << " bars_to_infill=" << num_bars_to_infill << "\n"; 
            }

            engine.updateProcessors("BarInfillStopLogitsProcessor.active", 1);
            if (verbose) std::cout << "[InfillBars] Activated bar infilling LogitsProcessor\n";
            engine.updateProcessors("BarInfillStopLogitsProcessor.reset", 0);
            if (verbose) std::cout << "[InfillBars] Reset LogitsProcessor\n";
            engine.updateProcessors("BarInfillStopLogitsProcessor.n_bars_to_infill", num_bars_to_infill);
            if (verbose) std::cout << "[InfillBars] Reset LogitsProcessor bar infill count\n";

            if (model->is_cached()) {
                if (verbose) std::cout << "[InfillBars] Resetting model cache\n";
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
                verbose
            );

            std::vector<int64_t> input_ids;
            input_ids.reserve(input_tokens.size());

            for (int id : input_tokens.ids) {
                input_ids.push_back(static_cast<int64_t>(id));
            }

            if (verbose) std::cout << "[InfillBars] Generating " << input_ids.size() << " input ids\n";
            std::vector<int64_t> generated = engine.generate(input_ids, model, verbose);
            if (verbose) std::cout << "[InfillBars] Generation complete. Tokens=" << generated.size() << "\n";

            int infill_token_start = std::get<0>(token_indices);
            int infill_token_end = std::get<1>(token_indices);

            std::vector<int> gen_ids;
            gen_ids.reserve(generated.size());
            for (auto v : generated) gen_ids.push_back(static_cast<int>(v));

            LibTok::TokSequence generated_seq({}, gen_ids, {}, {}, true);

            if (tokenizer.isTrained()) {
                if (verbose) {
                    std::cout << "[InfillBars] Decoding token ids after generation\n";
                }
                tokenizer.decodeTokenIds(generated_seq);
            }

            std::vector<std::string> tokens = generated_seq.tokens;

            auto fillbar_start_it = std::find(tokens.begin(), tokens.end(), "FillBar_Start");
            if (fillbar_start_it == tokens.end()) {
                throw std::runtime_error("[InfillBars] FillBar_Start token not found.");
            }
            size_t fillbar_start_idx = std::distance(tokens.begin(), fillbar_start_it);
            if (verbose) { 
                std::cout << "[InfillBars] Found FillBar_Start at idx=" << fillbar_start_idx << "\n"; 
            }

            std::vector<std::string> infilled_content = extractInfilledContent(tokens, fillbar_start_idx, num_bars_to_infill, verbose);

            if (verbose) { 
                std::cout << "[InfillBars] Extracted infilled_content size=" << infilled_content.size() << "\n"; 
            }

            LibTok::TokSequence infilled_seq(infilled_content);
            tokenizer.complete_sequence(infilled_seq);

            if (verbose) { 
                std::cout << "[InfillBars] Infilled sequence complete. Length=" << infilled_seq.size() << "\n"; 
            }

            auto before_seq = token_seq[track_idx].slice(0, infill_token_start);
            auto infilled_seq_copy = infilled_seq; // just to emphasize it's separate
            auto after_seq  = token_seq[track_idx].sliceUntilEnd(infill_token_end);

            // then concatenate as usual
            token_seq[track_idx] = before_seq + infilled_seq_copy + after_seq;
            
            if (verbose) { 
                std::cout << "[InfillBars] Track " << track_idx << " updated with infilled sequence.\n"; 
            }

        }
    }
    engine.updateProcessors("BarInfillStopLogitsProcessor.active", 0);
    if (verbose) std::cout << "[InfillBars] Deactivated bar infilling LogitsProcessor\n";

    if (verbose) std::cout << "[InfillBars] Infill finished\n";

}

std::pair<int,int> _adapt_prompt_for_infilling(
    LibTok::MMM &tokenizer,
    int track_idx, 
    BarSubset &subset,
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::TokSequence &input_tokens,
    int context_length,
    bool verbose
) {
    
    int start_bar_idx = std::get<0>(subset);
    int end_bar_idx = std::get<1>(subset);

    if (verbose) { 
        std::cout << "[AdaptPromptForBarInfill] Track=" << track_idx 
        << " start_bar=" << start_bar_idx << " end_bar=" 
        << end_bar_idx << "\n"; 
    }

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
    if (verbose) { 
        std::cout << "[AdaptPromptForBarInfill] Track " << track_idx << " has " << num_bars << " bars\n"; 
    }

    int context_start_idx = std::max(0, start_bar_idx - context_length);
    int context_end_idx = std::min(start_bar_idx + context_length, num_bars );
    if (verbose) { 
        std::cout << "[AdaptPromptForBarInfill] Context range: " 
        << context_start_idx << " -> " << context_end_idx << "\n"; 
    }
    LibTok::TokSequence seq_to_infill({}, {}, {}, {}, false);
    // We start of Track_Start and Program tokens
    seq_to_infill += token_seq[track_idx].slice(0,2);
    int infill_start_token_idx = 2;
    for (int bar_idx = context_start_idx; bar_idx < start_bar_idx; bar_idx++) {
        seq_to_infill += bar_subseqs_infill[bar_idx];
        infill_start_token_idx += bar_subseqs_infill[bar_idx].size();
    }
    if (verbose) { std::cout << "[AdaptPromptForBarInfill] Infill start token index=" << infill_start_token_idx << "\n";}
    int infill_end_token_idx = infill_start_token_idx;
    for (int bar_idx = start_bar_idx; bar_idx < end_bar_idx; bar_idx++) {
        LibTok::TokSequence infill_bar_seq({"Infill_Bar"});
        tokenizer.complete_sequence(infill_bar_seq);
        seq_to_infill += infill_bar_seq;
        infill_end_token_idx += bar_subseqs_infill[bar_idx].size();
    }
    if (verbose) { std::cout << "[AdaptPromptForBarInfill] Infill end token index=" << infill_end_token_idx << "\n";}
    for (int bar_idx = end_bar_idx; bar_idx < context_end_idx; bar_idx++) {
        seq_to_infill += bar_subseqs_infill[bar_idx];
    }
    if (tokenizer.isTrained()) {
        if (verbose) { std::cout << "[AdaptPromptForBarInfill] Encoding sequence to infill" << "\n";}
        tokenizer.encodeTokenIds(seq_to_infill);
    } else {
        if (verbose) { std::cout << "[AdaptPromptForBarInfill] Not encoding infilling sequence, no BPE" << "\n";}
    }

    for (int i = 0; i < token_seq.size(); i++) {
        if (i == track_idx) {
            input_tokens += seq_to_infill;
        } else {
            if (tokenizer.isTrained()) {
                tokenizer.encodeTokenIds(token_seq);
            }
            auto bar_subseqs = token_seq[i].splitPerBars();
            int num_bars = bar_subseqs.size();
            int tokseq_len = token_seq[i].size();

            input_tokens += token_seq[i].slice(0, 2);

            if (num_bars < context_end_idx || num_bars < context_start_idx) {
                throw std::runtime_error("[AdaptPromptForBarInfill] Context error. Track " + std::to_string(i) + " has " + std::to_string(num_bars) + " bars, but context is from bar " + std::to_string(context_start_idx) + " to " + std::to_string(context_end_idx) + ".");
            }

            for (int bar_idx = context_start_idx; bar_idx < context_end_idx; bar_idx++) {
                input_tokens += bar_subseqs[bar_idx];
            }
            
            if (context_end_idx != num_bars) {
                input_tokens += token_seq[i].sliceUntilEnd(tokseq_len - 1);
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

    if (verbose) { 
        std::cout << "[AdaptPromptForBarInfill] Infill token range: " 
        << infill_start_token_idx << " → " 
        << infill_end_token_idx << "\n"; 
        std::cout << "[AdaptPromptForBarInfill] Final input_tokens size=" 
        << input_tokens.size() << "\n"; 
    }

    return std::pair<int,int>(infill_start_token_idx, infill_end_token_idx);
}

std::vector<std::string> extractInfilledContent(
    const std::vector<std::string>& tokens,
    size_t fillbar_start_idx,
    int num_bars_to_infill,
    bool verbose
) {
    std::vector<std::string> infill_tokens;
    std::vector<std::vector<std::string>> bars;

    if (verbose) {
        std::cout << "[InfillExtraction] Starting extraction after FillBar_Start idx="
                  << fillbar_start_idx << ", num_bars_to_infill=" << num_bars_to_infill << "\n";
    }

    if (fillbar_start_idx >= tokens.size() - 1) {
        if (verbose) std::cerr << "[InfillExtraction] No tokens after FillBar_Start.\n";
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
            if (verbose) std::cout << "  [InfillExtraction] FillBar_End reached.\n";
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

    if (verbose) {
        std::cout << "[InfillExtraction] Found " << bars.size() << " bar(s) after FillBar_Start.\n";
    }

    // ---------------------------------------------------------------------
    // Handle insufficient/excess bars
    // ---------------------------------------------------------------------
    int generated_bars = static_cast<int>(bars.size());
    int produced_bars = std::min(num_bars_to_infill, generated_bars);

    if (produced_bars < generated_bars) {
        if (verbose) std::cout << "[InfillExtraction] Model generated too many bars!\n";
    } else if (produced_bars > generated_bars) {
        if (verbose) std::cout << "[InfillExtraction] Model generated too few barss!\n";
    }

    // Take the first `produced_bars`
    for (int i = 0; i < produced_bars; ++i) {
        infill_tokens.insert(infill_tokens.end(), bars[i].begin(), bars[i].end());
    }

    // If missing, append empty bars
    int missing = num_bars_to_infill - produced_bars;
    if (missing > 0) {
        if (verbose) {
            std::cerr << "[InfillExtraction] Only " << produced_bars
                      << " bars produced; appending " << missing
                      << " empty Bar_None(s).\n";
        }
        for (int i = 0; i < missing; ++i) {
            infill_tokens.push_back("Bar_None");
        }
    }

    return infill_tokens;
}

}
}