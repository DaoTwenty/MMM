#pragma once

#include <fstream>
#include <nlohmann/json.hpp>

#include "config.h"
#include "promptconfig.h"
#include "inference.h"

using Controls = std::vector<std::string>;
using BarSubset = std::tuple<int, int, Controls>;

namespace mmm::utils {

inline void from_json(const nlohmann::json& j, mmm::sampling::GenerationConfig& cfg) {
    j.at("do_sample").get_to(cfg.do_sample);
    j.at("min_new_tokens").get_to(cfg.min_new_tokens);
    j.at("max_new_tokens").get_to(cfg.max_new_tokens);
    j.at("pad_token_id").get_to(cfg.pad_token_id);
    j.at("repetition_penalty").get_to(cfg.repetition_penalty);
    j.at("temperature").get_to(cfg.temperature);
    j.at("top_k").get_to(cfg.top_k);
    j.at("top_p").get_to(cfg.top_p);
}

inline void loadGenerationConfigFromJson(const std::string& path, mmm::sampling::GenerationConfig& cfg) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open GenerationConfig JSON: " + path);
    nlohmann::json j;
    f >> j;
    from_json(j, cfg);
}

inline void from_json(const nlohmann::json& j, mmm::inference::BarInfilling& cfg) {
    for (auto it = j.begin(); it != j.end(); ++it) {
        int track_idx = std::stoi(it.key());
        std::vector<BarSubset> subsets;
        for (auto& item : it.value()) {
            Controls controls = item.at("controls").get<Controls>();
            int start = item.at("start_bar").get<int>();
            int end = item.at("end_bar").get<int>();
            subsets.emplace_back(start, end, controls);
        }
        cfg.bars[track_idx] = subsets;
    }
}

inline void from_json(const nlohmann::json& j, mmm::inference::TrackInfilling& cfg) {
    for (auto it = j.begin(); it != j.end(); ++it) {
        int track_idx = std::stoi(it.key());
        mmm::inference::TrackInfilling::Controls controls = it.value().get<mmm::inference::TrackInfilling::Controls>();
        cfg.tracks[track_idx] = controls;
    }
}

inline void from_json(const nlohmann::json& j, mmm::inference::TrackSampling& cfg) {
    for (const auto& item : j) {
        int program = item.at("program").get<int>();
        std::vector<std::string> controls = item.at("controls").get<std::vector<std::string>>();
        cfg.tracks.emplace_back(program, controls);
    }
}

inline void loadPromptConfigFromJson(const std::string& path, mmm::inference::PromptConfig& cfg) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open PromptConfig JSON: " + path);

    nlohmann::json j;
    f >> j;

    std::string mode = j.at("mode").get<std::string>();

    if (mode == "BarInfilling") {
        mmm::inference::BarInfilling barCfg;
        from_json(j.at("config"), barCfg);
        cfg.mode = barCfg;
    } 
    else if (mode == "TrackInfilling") {
        mmm::inference::TrackInfilling trackInfCfg;
        from_json(j.at("config"), trackInfCfg);
        cfg.mode = trackInfCfg;
    } 
    else if (mode == "TrackSampling") {
        mmm::inference::TrackSampling trackSampCfg;
        from_json(j.at("config"), trackSampCfg);
        cfg.mode = trackSampCfg;
    } 
    else {
        throw std::runtime_error("Unknown PromptConfig mode: " + mode);
    }

    //if (j.contains("context_length")) cfg.context_length = j.at("context_length").get<int>();
    //if (j.contains("bars_per_step")) cfg.bars_per_step = j.at("bars_per_step").get<int>();
    //if (j.contains("tracks_per_step")) cfg.tracks_per_step = j.at("tracks_per_step").get<int>();

}


inline mmm::sampling::LogitsWarperList createWarperListFromConfig(const mmm::sampling::GenerationConfig& config) { 
    mmm::sampling::LogitsWarperList warpers; 
    // Temperature scaling 
    if (config.temperature != 1.0f) { 
        warpers.addWarper(std::make_shared<mmm::sampling::TemperatureLogitsWarper>(config.temperature)); 
    } 
    // Choose between Top-K and Top-P (mutually exclusive) 
    if (config.top_k > 0 && config.top_p >= 1.0f) { 
        warpers.addWarper(std::make_shared<mmm::sampling::TopKLogitsWarper>(config.top_k)); 
    } else if (config.top_p < 1.0f) { 
        warpers.addWarper(std::make_shared<mmm::sampling::TopPLogitsWarper>(config.top_p)); 
    } 
    return warpers; 
}

inline mmm::sampling::LogitsProcessorList createProcessorListFromConfig(const mmm::sampling::GenerationConfig& config, int eos_token_id, int infill_token_id, int bar_token_id, int track_start_token_id, int track_end_token_id) { 
    mmm::sampling::LogitsProcessorList processors; 
    // Repetition penalty 
    if (config.repetition_penalty != 1.0f) { 
        processors.addProcessor( std::make_shared<mmm::sampling::RepetitionPenaltyLogitsProcessor>(config.repetition_penalty)); 
    } 
    // Min length processor (forbid EOS until at least min length) 
    if (config.max_new_tokens > 0) { 
        processors.addProcessor( std::make_shared<mmm::sampling::MaxLengthLogitsProcessor>(config.max_new_tokens, eos_token_id)); 
    } 

    if (config.min_new_tokens > 0) { 
        processors.addProcessor( std::make_shared<mmm::sampling::MinLengthLogitsProcessor>(config.min_new_tokens, eos_token_id)); 
    }
    // Include Generation Mode processors  
    auto bar_infill_stop = std::make_shared<mmm::sampling::BarInfillStopLogitsProcessor>( infill_token_id, bar_token_id, eos_token_id); 
    bar_infill_stop->updateConfig("BarInfillStopLogitsProcessor.active", 0);
    processors.addProcessor(bar_infill_stop); 

    auto track_sample_stop = std::make_shared<mmm::sampling::TrackSampleStopLogitsProcessor>( bar_token_id, track_start_token_id, track_end_token_id, eos_token_id); 
    track_sample_stop->updateConfig("TrackSampleStopLogitsProcessor.active", 0);
    processors.addProcessor(track_sample_stop);
    return processors;
}

} // namespace mmm::utils