#pragma once

#include "model.h"
#include "config.h" 

namespace mmm {
namespace inference {

void generate(CausalLM model, GenerationConfig gen_config);

}
}