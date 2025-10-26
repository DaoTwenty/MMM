#pragma once

#include "token_utils.h"

namespace mmm::utils {

void deconstructTokenToBaseIds(
    int token_id,
    LibTok::MMM& tokenizer,
    int base_vocab_size,
    std::unordered_map<int, std::unordered_set<int>>& memo, // cache
    std::unordered_set<int>& recursion_stack, // for cycle prevention
    std::unordered_set<int>& out_base_ids)
{
    // Already computed → reuse result
    if (memo.count(token_id)) {
        out_base_ids.insert(memo[token_id].begin(), memo[token_id].end());
        return;
    }

    // Cycle prevention
    if (recursion_stack.count(token_id)) {
        return;
    }

    recursion_stack.insert(token_id);

    std::unordered_set<int> base_ids;

    if (token_id < base_vocab_size) {
        base_ids.insert(token_id);
    } else {
        LibTok::TokSequence seq({}, {token_id}, {}, {}, true);
        tokenizer.decodeTokenIds(seq);
        tokenizer.complete_sequence(seq);

        if (seq.ids.empty() || (seq.ids.size() == 1 && seq.ids[0] == token_id)) {
            base_ids.insert(token_id);
        } else {
            for (int sub_id : seq.ids) {
                deconstructTokenToBaseIds(sub_id, tokenizer, base_vocab_size, memo, recursion_stack, base_ids);
            }
        }
    }

    // Save and propagate
    memo[token_id] = base_ids;
    out_base_ids.insert(base_ids.begin(), base_ids.end());
    recursion_stack.erase(token_id);
}

void buildContainMaps(
    LibTok::MMM& tokenizer,
    int base_vocab_size,
    const SpecialTokens& special_tokens,
    Logger& logger,
    std::unordered_map<int, bool>& contains_bar,
    std::unordered_map<int, bool>& contains_track_end)
{
    int vocab_size = tokenizer.getVocabSize();
    std::unordered_map<int, std::unordered_set<int>> memo;
    for (int id = 0; id < vocab_size; ++id) {
        std::unordered_set<int> recursion_stack;
        std::unordered_set<int> base_ids;

        deconstructTokenToBaseIds(id, tokenizer, base_vocab_size, memo, recursion_stack, base_ids);

        bool has_bar = base_ids.count(special_tokens.bar_none);
        bool has_track_end = base_ids.count(special_tokens.track_end);

        contains_bar[id] = has_bar;
        contains_track_end[id] = has_track_end;
    }
}

} // namespace mmm::utils