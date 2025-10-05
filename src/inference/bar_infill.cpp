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
            engine.updateProcessors("BarInfillStopLogitsProcessor.n_attribute_controls", std::get<2>(subset).size());
            if (verbose) std::cout << "[InfillBars] Reset LogitsProcessor attribute control count\n";

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

            tokenizer.complete_sequence(generated_seq);
            if (tokenizer.isTrained()) {
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

    // collect indices of Bar_None after the FillBar_Start
    std::vector<size_t> bidx;
    for (size_t i = fillbar_start_idx + 1; i < tokens.size(); ++i) {
        if (tokens[i] == "Bar_None") bidx.push_back(i);
    }

    if (bidx.empty()) {
        if (verbose) {
            std::cerr << "[InfillExtraction] No Bar_None found after FillBar_Start; "
                         "falling back to " << num_bars_to_infill << " empty bars.\n";
        }
        // produce exactly num_bars_to_infill empty bars (represented as consecutive Bar_None tokens)
        return std::vector<std::string>(num_bars_to_infill, std::string("Bar_None"));
    }

    bool last_is_bar_none = (!tokens.empty() && tokens.back() == "Bar_None");
    // number of completed bars produced by the model:
    // if last token is Bar_None, final Bar_None is a terminator -> completed bars = k-1
    // otherwise completed bars = k
    int k = static_cast<int>(bidx.size());
    int generated_bars = last_is_bar_none ? (k - 1) : k;
    if (generated_bars < 0) generated_bars = 0;

    int produced_bars = std::min(num_bars_to_infill, generated_bars);

    if (verbose) {
        std::cout << "[InfillExtraction] Found " << k << " Bar_None tokens after FillBar_Start"
                  << " (last_is_bar_none=" << last_is_bar_none << "), "
                  << "generated_bars=" << generated_bars
                  << ", will produce " << produced_bars << " filled bars (requested "
                  << num_bars_to_infill << ").\n";
    }

    // begin at first Bar_None
    size_t idx_begin = bidx[0];
    size_t idx_end = idx_begin; // exclusive

    if (produced_bars == 0) {
        // Extract nothing, then append empty bars below
        idx_end = idx_begin;
    } else {
        // Choose end index:
        // - If the model ended with a terminating Bar_None (last_is_bar_none=true),
        //   then the p-th bar's end boundary is at bidx[produced_bars] (exists because produced_bars <= k-1).
        // - If not, and produced_bars < k, use bidx[produced_bars].
        // - If not, and produced_bars == k, use tokens.size().
        if (last_is_bar_none) {
            // safe because produced_bars <= generated_bars = k-1 -> bidx[produced_bars] exists
            idx_end = bidx[produced_bars];
        } else {
            if (produced_bars < k) idx_end = bidx[produced_bars];
            else idx_end = tokens.size();
        }
    }

    // Copy the token range [idx_begin, idx_end)
    if (idx_begin < idx_end && idx_end <= tokens.size()) {
        infill_tokens.insert(infill_tokens.end(), tokens.begin() + idx_begin, tokens.begin() + idx_end);
    }

    int missing = num_bars_to_infill - produced_bars;
    if (missing > 0) {
        if (verbose) {
            std::cerr << "[InfillExtraction] Only " << produced_bars << " bars produced; "
                      << "appending " << missing << " empty Bar_None(s).\n";
        }
        // Append `missing` Bar_None tokens (consecutive Bar_None = empty bars)
        for (int i = 0; i < missing; ++i) infill_tokens.push_back("Bar_None");
    }

    if (!last_is_bar_none) {
        // If the model didn't end with Bar_None, that likely means it stopped for another reason (max_new_tokens).
        // Warn the user so they can adjust generation length if desired.
        if (verbose) {
            std::cerr << "[InfillExtraction] Warning: model output did not end with a final Bar_None; "
                      << "the trailing content (if any) is preserved as the last bar's content.\n";
        }
    }

    return infill_tokens;
}

}
}