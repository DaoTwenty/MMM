#pragma once

#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "mmm.h"
#include "tok_sequence.h"
#include "utils.h"
#include "logger.h"

namespace mmm::utils {

struct SpecialTokens;  

void deconstructTokenToBaseIds(
    int token_id,
    LibTok::MMM& tokenizer,
    int base_vocab_size,
    std::unordered_map<int, std::unordered_set<int>>& memo, 
    std::unordered_set<int>& recursion_stack, 
    std::unordered_set<int>& out_base_ids);

void buildContainMaps(
    LibTok::MMM& tokenizer,
    int base_vocab_size,
    const SpecialTokens& special_tokens,
    Logger& logger,
    std::unordered_map<int, bool>& contains_bar,
    std::unordered_map<int, bool>& contains_track_end);

} // namespace mmm::utils