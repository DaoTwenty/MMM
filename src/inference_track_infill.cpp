#include "inference.h"

#include <algorithm>

namespace mmm {

namespace inference {

void infill_tracks(
    std::vector<LibTok::TokSequence> &token_seq,
    LibTok::MMM &tokenizer,                               
    mmm::inference::TrackInfilling &infill_config,
    mmm::IModel* model,
    mmm::sampling::SamplingEngine &engine,
    bool verbose
) {}   

}
}