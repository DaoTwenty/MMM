#pragma once

#include <variant>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>

using Controls = std::vector<std::string>;
using BarSubset = std::tuple<int, int, Controls>;

namespace mmm {
namespace inference {

struct BarInfilling;

inline std::ostream& operator<<(std::ostream& os, const BarInfilling& cfg);

// ----------------- BarInfilling -----------------
struct BarInfilling {
    std::unordered_map<int, std::vector<BarSubset>> bars;

    bool empty() const { return bars.empty(); }

    void print() const { std::cout << *this; } // delegate to operator<<
};

// free function for streaming
inline std::ostream& operator<<(std::ostream& os, const BarInfilling& cfg) {
    if (!cfg.bars.empty()) {
        os << "  bars_to_infill:\n";
        for (const auto& [track_idx, subsets] : cfg.bars) {
            os << "    Track " << track_idx << ":\n";
            for (const auto& [start_bar, end_bar, controls] : subsets) {
                os << "      bars [" << start_bar << ", " << end_bar << "), controls=[";
                for (size_t i = 0; i < controls.size(); ++i) {
                    os << controls[i];
                    if (i + 1 < controls.size()) os << ", ";
                }
                os << "]\n";
            }
        }
    } else {
        os << "  bars_to_generate=None\n";
    }
    return os;
}

struct TrackSampling;

inline std::ostream& operator<<(std::ostream& os, const TrackSampling& cfg);

// ----------------- TrackSampling -----------------
struct TrackSampling {
    std::vector<std::pair<int, std::vector<std::string>>> tracks;

    bool empty() const { return tracks.empty(); }

    void print() const { std::cout << *this; } // delegate to operator<<
};

inline std::ostream& operator<<(std::ostream& os, const TrackSampling& cfg) {
    if (!cfg.tracks.empty()) {
        os << "  new_tracks:\n";
        for (const auto& [program, controls] : cfg.tracks) {
            os << "    program=" << program << ", controls=[";
            for (size_t i = 0; i < controls.size(); ++i) {
                os << controls[i];
                if (i + 1 < controls.size()) os << ", ";
            }
            os << "]\n";
        }
    } else {
        os << "  new_tracks=None\n";
    }
    return os;
}

// ----------------- PromptConfig -----------------
using PromptMode = std::variant<BarInfilling, TrackSampling>;

struct PromptConfig;

inline std::ostream& operator<<(std::ostream& os, const PromptConfig& cfg);

struct PromptConfig {
    PromptMode mode;
    int context_length = 4;

    bool empty() const {
        return std::visit([](auto const &m) { return m.empty(); }, mode);
    }

    const auto &bars() const {
        return std::get<BarInfilling>(mode).bars;
    }

    bool bar_infilling() const { return std::holds_alternative<BarInfilling>(mode); }
    bool track_sampling() const { return std::holds_alternative<TrackSampling>(mode); }

    void print() const { std::cout << *this; } // delegate to operator<<
};

inline std::ostream& operator<<(std::ostream& os, const PromptConfig& cfg) {
    if (std::holds_alternative<BarInfilling>(cfg.mode)) {
        os << "Mode: BarInfilling\n";
        os << std::get<BarInfilling>(cfg.mode);
    } else if (std::holds_alternative<TrackSampling>(cfg.mode)) {
        os << "Mode: TrackSampling\n";
        os << std::get<TrackSampling>(cfg.mode);
    }
    return os;
}

} // namespace inference
} // namespace mmm
