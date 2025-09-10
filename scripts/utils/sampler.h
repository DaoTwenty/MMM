#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace mmm {

namespace sampling {

class Sampler {
public:
    Sampler(std::mt19937::result_type seed = std::random_device{}());

    // Multinomial sampling
    int64_t sample(const std::vector<float>& logits);

    // Greedy decoding
    int64_t argmax(const std::vector<float>& logits);
private:
    std::mt19937 rng_;
};

}

}
