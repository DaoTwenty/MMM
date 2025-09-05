#pragma once

#include <string>

#include "model.h"

namespace mmm {

class TokenizationConfig {
public:

    std::string tokenization;
    LibTok::TokenizationConfig tokenizer_config;
    int vocab_size = -1;

};

class Baseline {
public:

Baseline(std::string name, TokenizationConfig tokenization_config);
~Baseline();

void setSeed(int seed);
LibTok::MusicTokenizer createTokenizer();

int pad_token_id();
int bos_token_id();
int eos_token_id();
std::vector<int> special_tokens_ids();

operator std::string() {return m_name;}

private:

    std::string m_name;
    int m_seed;
    TokenizationConfig m_tokenization_config;

};

}