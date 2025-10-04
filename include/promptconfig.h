#pragma once

#include <variant>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>

namespace mmm {

namespace inference {


struct BarInfilling {
    using Controls = std::vector<std::string>;
    using BarSubset = std::tuple<int, int, Controls>;
    std::unordered_map<int, std::vector<BarSubset>> bars;

    bool empty() const {
        return bars.empty();
    }

    void print() const {
        if (!bars.empty()) {
            std::cout << "  bars_to_infill:\n";
            for (const auto& [track_idx, subsets] : bars) {
                std::cout << "    Track " << track_idx << ":\n";
                for (const auto& [start_bar, end_bar, controls] : subsets) {
                    std::cout << "      bars [" << start_bar << ", " << end_bar << ")"
                              << ", controls=[";
                    for (size_t i = 0; i < controls.size(); ++i) {
                        std::cout << controls[i];
                        if (i + 1 < controls.size()) std::cout << ", ";
                    }
                    std::cout << "]\n";
                }
            }
        } else {
            std::cout << "  bars_to_generate=None\n";
        }
    }
};

struct TrackInfilling {
    using Controls = std::vector<std::string>;
    std::unordered_map<int, Controls> tracks;

    bool empty() const {
        return tracks.empty();
    }

    void print() const {
        if (!tracks.empty()) {
            std::cout << " track_to_infill:\n";
            for (const auto& [track_idx, controls] : tracks) {
                std::cout << "    Track " << track_idx << ":\n";
                std::cout << " controls=[";
                for (size_t i = 0; i < controls.size(); ++i) {
                    std::cout << controls[i];
                    if (i + 1 < controls.size()) std::cout << ", ";
                }
                std::cout << "]\n";
            } 
        } else {
            std::cout << "  tracks_to_generate=None\n";
        }
    }
};

struct TrackSampling {
    std::vector<std::pair<int, std::vector<std::string>>> tracks;

    bool empty() const {
        return tracks.empty();
    }

    void print() const {
        if (!tracks.empty()) {
            std::cout << "  new_tracks:\n";
            for (const auto& [program, controls] : tracks) {
                std::cout << "    program=" << program << ", controls=[";
                for (size_t i = 0; i < controls.size(); ++i) {
                    std::cout << controls[i];
                    if (i + 1 < controls.size()) std::cout << ", ";
                }
                std::cout << "]\n";
            }
        } else {
            std::cout << "  new_tracks=None\n";
        }
    }
};

using PromptMode = std::variant<BarInfilling, TrackInfilling,  TrackSampling>;

struct PromptConfig {
    PromptMode mode;
    // TODO: Implement
    //int context_length = 4;
    //int bars_per_step = 1;
    //int tracks_per_step = 1;

    bool empty() const {
        return std::visit([](auto const &m) {
            return m.empty();  // calls the respective struct's empty()
        }, mode);
    }

    const auto &bars() const {
        return std::get<BarInfilling>(mode).bars;
    }

    bool bar_infilling() {
        return std::holds_alternative<BarInfilling>(mode);
    }

    bool track_infilling() {
        return std::holds_alternative<TrackInfilling>(mode);
    }

    bool track_sampling() {
        return std::holds_alternative< TrackSampling>(mode);
    }

    void print() const {
        if (std::holds_alternative<BarInfilling>(mode)) {
            std::cout << "Mode: BarInfilling\n";
        } else if (std::holds_alternative<TrackInfilling>(mode)) {
            std::cout << "Mode: TrackInfilling\n";
        } else if (std::holds_alternative< TrackSampling>(mode)) {
            std::cout << "Mode: TrackSampling\n";
        }
        std::visit([](auto const &m) { m.print(); }, mode);
    }
};

}

}