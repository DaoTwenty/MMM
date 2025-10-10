#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>

namespace mmm {

namespace sampling {

class Sampler {
public:
    Sampler(int seed = std::random_device{}());
    // Multinomial sampling
    int64_t sample(const std::vector<float>& logits);

    // Greedy decoding
    int64_t argmax(const std::vector<float>& logits);

    int getSeed() const {
        return seed_;
    }

    void setSeed(int new_seed) {
        seed_ = new_seed;
        rng_.seed(static_cast<std::mt19937::result_type>(new_seed));
    }

private:
    int seed_;
    std::mt19937 rng_;
};

}

}
