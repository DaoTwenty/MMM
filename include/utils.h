#pragma once

#include <fstream>
#include <nlohmann/json.hpp>

#include "generationconfig.h"
#include "promptconfig.h"
#include "inference.h"

using Controls = std::vector<std::string>;
using BarSubset = std::tuple<int, int, Controls>;

namespace mmm::utils {

inline void from_json(const nlohmann::json& j, mmm::sampling::GenerationConfig& cfg) {
    j.at("do_sample").get_to(cfg.do_sample);
    j.at("max_new_tokens").get_to(cfg.max_new_tokens);
    j.at("attempts").get_to(cfg.attempts);
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

inline void to_json(nlohmann::json& j, const mmm::inference::BarInfilling& cfg) {
    j = nlohmann::json::object();
    for (const auto& [track_idx, subsets] : cfg.bars) {
        j[std::to_string(track_idx)] = nlohmann::json::array();
        for (const auto& [start, end, controls] : subsets) {
            nlohmann::json item;
            item["start_bar"] = start;
            item["end_bar"] = end;
            item["controls"] = controls;
            j[std::to_string(track_idx)].push_back(item);
        }
    }
}

inline void from_json(const nlohmann::json& j, mmm::inference::TrackSampling& cfg) {
    for (const auto& item : j) {
        int program = item.at("program").get<int>();
        std::vector<std::string> controls = item.at("controls").get<std::vector<std::string>>();
        cfg.tracks.emplace_back(program, controls);
    }
}

inline void to_json(nlohmann::json& j, const mmm::inference::TrackSampling& cfg) {
    j = nlohmann::json::array();
    for (const auto& [program, controls] : cfg.tracks) {
        nlohmann::json item;
        item["program"] = program;
        item["controls"] = controls;
        j.push_back(item);
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
    } else if (mode == "TrackSampling") {
        mmm::inference::TrackSampling trackSampCfg;
        from_json(j.at("config"), trackSampCfg);
        cfg.mode = trackSampCfg;
    } 
    else {
        throw std::runtime_error("Unknown PromptConfig mode: " + mode);
    }

    if (j.contains("context_length")) cfg.context_length = j.at("context_length").get<int>();
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

struct SpecialTokens {
    int eos_none        = -1;
    int fillbar_start   = -1;
    int fillbar_end     = -1;
    int bar_none        = -1;
    int infill_track    = -1;
    int track_start     = -1;
    int track_end       = -1;

    // Optional: constructor from a vocab map
    SpecialTokens(const std::unordered_map<std::string,int>& vocab) {
        eos_none      = vocab.at("EOS_None");
        fillbar_start = vocab.at("FillBar_Start");
        fillbar_end   = vocab.at("FillBar_End");
        bar_none      = vocab.at("Bar_None");
        infill_track  = vocab.at("Infill_Track");
        track_start   = vocab.at("Track_Start");
        track_end     = vocab.at("Track_End");
    }

    void print() const {
        std::cout << "[SpecialTokens]\n";
        std::cout << "  EOS_None      = " << eos_none << "\n";
        std::cout << "  FillBar_Start = " << fillbar_start << "\n";
        std::cout << "  FillBar_End   = " << fillbar_end << "\n";
        std::cout << "  Bar_None      = " << bar_none << "\n";
        std::cout << "  Infill_Track  = " << infill_track << "\n";
        std::cout << "  Track_Start   = " << track_start << "\n";
        std::cout << "  Track_End     = " << track_end << "\n";
    }
};

inline mmm::sampling::LogitsProcessorList createProcessorListFromConfig(const mmm::sampling::GenerationConfig& config, const SpecialTokens& tokens) { 
    mmm::sampling::LogitsProcessorList processors; 
    // Repetition penalty 
    if (config.repetition_penalty != 1.0f) { 
        processors.addProcessor( std::make_shared<mmm::sampling::RepetitionPenaltyLogitsProcessor>(config.repetition_penalty)); 
    } 

    // Include Generation Mode processors  
    auto bar_infill_stop = std::make_shared<mmm::sampling::BarInfillStopLogitsProcessor>( tokens.fillbar_start, tokens.fillbar_end, tokens.bar_none, tokens.eos_none); 
    bar_infill_stop->updateConfig("BarInfillStopLogitsProcessor.active", 0);
    processors.addProcessor(bar_infill_stop); 

    auto track_sample_stop = std::make_shared<mmm::sampling::TrackSampleStopLogitsProcessor>( tokens.bar_none, tokens.track_start, tokens.track_end, tokens.eos_none); 
    track_sample_stop->updateConfig("TrackSampleStopLogitsProcessor.active", 0);
    processors.addProcessor(track_sample_stop);
    return processors;
}

} // namespace mmm::utils