// pybind_inference.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine.h"
#include "model.h"
#include "promptconfig.h"
#include "mmm.h"
#include "tok_sequence.h"
#include "utils.h"
#include "inference.h"
#include "model.h"
#include "logger.h"

namespace py = pybind11;

static nlohmann::json py_to_json(const py::handle& obj) {
    if (py::isinstance<py::bool_>(obj)) {
        return obj.cast<bool>();
    } else if (py::isinstance<py::int_>(obj)) {
        return obj.cast<int>();
    } else if (py::isinstance<py::float_>(obj)) {
        return obj.cast<double>();
    } else if (py::isinstance<py::str>(obj)) {
        return obj.cast<std::string>();
    } else if (py::isinstance<py::dict>(obj)) {
        nlohmann::json j = nlohmann::json::object();
        for (auto item : obj.cast<py::dict>()) {
            j[py::str(item.first)] = py_to_json(item.second);
        }
        return j;
    } else if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        nlohmann::json j = nlohmann::json::array();
        for (auto item : obj.cast<py::iterable>()) {
            j.push_back(py_to_json(item));
        }
        return j;
    } else if (obj.is_none()) {
        return nullptr;
    } else {
        // Fallback: string representation
        return py::str(obj);
    }
}

py::object json_to_py(const nlohmann::json &j) {
    if (j.is_object()) {
        py::dict d;
        for (auto it = j.begin(); it != j.end(); ++it)
            d[py::str(it.key())] = json_to_py(it.value());
        return d;
    } else if (j.is_array()) {
        py::list l;
        for (auto &el : j) l.append(json_to_py(el));
        return l;
    } else if (j.is_string()) {
        return py::str(j.get<std::string>());
    } else if (j.is_boolean()) {
        return py::bool_(j.get<bool>());
    } else if (j.is_number_integer()) {
        return py::int_(j.get<int>());
    } else if (j.is_number_unsigned()) {
        return py::int_(j.get<unsigned>());
    } else if (j.is_number_float()) {
        return py::float_(j.get<double>());
    } else if (j.is_null()) {
        return py::none();
    }
    throw std::runtime_error("Unsupported JSON type");
}

class LoggerGlobal {
public:
    static mmm::utils::Logger& instance() {
        static mmm::utils::Logger logger(mmm::utils::LogLevel::INFO);
        return logger;
    }

    static void setLogLevel(mmm::utils::LogLevel level) {
        instance().setMaxLevel(level);
    }

    static void setMedium(mmm::utils::LogLevel level, mmm::utils::LogMedium medium, const std::string& filename = "") {
        instance().setMedium(level, medium, filename);
    }
};

PYBIND11_MODULE(mmm, m) {
    m.doc() = "Python bindings for MMM inference";

    m.def("generate",
    [](mmm::IModel* model,
       LibTok::MMM &tokenizer,
       mmm::inference::PromptConfig &prompt_config,
       mmm::sampling::SamplingEngine &sampling_engine,
       LibTok::ScoreType &score) {
           return mmm::inference::generate(
            model, 
            tokenizer, 
            prompt_config, 
            sampling_engine, 
            score, 
            LoggerGlobal::instance()
        );
       },
    py::arg("model"),
    py::arg("tokenizer"),
    py::arg("prompt_config"),
    py::arg("sampling_engine"),
    py::arg("score"),
    "Run generation");
    
    // ----------------- GenerationConfig -----------------
    py::class_<mmm::sampling::GenerationConfig>(m, "GenerationConfig")
    .def(py::init<>())

    // Constructor with arguments (all fields)
    .def(py::init<bool, int, int, int, float, float, float, int>(),
         py::arg("do_sample") = true,
         py::arg("max_new_tokens") = 100,
         py::arg("attempts") = 3,
         py::arg("pad_token_id") = 0,
         py::arg("repetition_penalty") = 1.0f,
         py::arg("temperature") = 1.0f,
         py::arg("top_p") = 1.0f,
         py::arg("top_k") = 0
    )

    .def_readwrite("do_sample", &mmm::sampling::GenerationConfig::do_sample)
    .def_readwrite("max_new_tokens", &mmm::sampling::GenerationConfig::max_new_tokens)
    .def_readwrite("attempts", &mmm::sampling::GenerationConfig::attempts)
    .def_readwrite("pad_token_id", &mmm::sampling::GenerationConfig::pad_token_id)
    .def_readwrite("repetition_penalty", &mmm::sampling::GenerationConfig::repetition_penalty)
    .def_readwrite("temperature", &mmm::sampling::GenerationConfig::temperature)
    .def_readwrite("top_k", &mmm::sampling::GenerationConfig::top_k)
    .def_readwrite("top_p", &mmm::sampling::GenerationConfig::top_p)
    .def_static("from_json_str", [](const std::string &json_str) {
        nlohmann::json j = nlohmann::json::parse(json_str);
        mmm::sampling::GenerationConfig cfg;
        mmm::utils::from_json(j, cfg);
        return cfg;
    })
    .def_static("from_dict", [](const py::dict &py_cfg) {
        nlohmann::json j = nlohmann::json::object();
        for (auto item : py_cfg) {
            std::string key = py::str(item.first);
            j[key] = py_to_json(item.second);
        }
        mmm::sampling::GenerationConfig cfg;
        mmm::utils::from_json(j, cfg);
        return cfg;
    })
    .def_static("from_json_file", [](const std::string &path) {
        mmm::sampling::GenerationConfig cfg;
        mmm::utils::loadGenerationConfigFromJson(path, cfg);
        return cfg;
    })
    .def("__repr__", [](const mmm::sampling::GenerationConfig &cfg) {
        std::ostringstream oss;
        oss << cfg;
        return oss.str();
    });

    // ----------------- SamplingEngine -----------------
    py::class_<mmm::sampling::SamplingEngine>(m, "SamplingEngine")
    // --- Constructor ---
    .def(
        py::init([](const mmm::sampling::GenerationConfig &config,
                    LibTok::MMM &tokenizer,
                    int seed) {
            return mmm::inference::createEngine(
                const_cast<mmm::sampling::GenerationConfig &>(config),
                tokenizer,
                LoggerGlobal::instance(),
                seed
            );
        }),
        py::arg("config"),
        py::arg("tokenizer"),
        py::arg("seed") = -1,
        "Construct a SamplingEngine from a GenerationConfig and tokenizer."
    )

    // --- Accessors ---
    .def_property(
        "config",
        [](mmm::sampling::SamplingEngine &self) -> const mmm::sampling::GenerationConfig& {
            return self.getConfig();
        },
        [](mmm::sampling::SamplingEngine &self, const mmm::sampling::GenerationConfig &cfg) {
            self.setConfig(cfg);
        },
        "Get or set the GenerationConfig of the engine."
    )

    .def_property(
        "seed",
        [](mmm::sampling::SamplingEngine &self) {
            return self.getSeed();
        },
        [](mmm::sampling::SamplingEngine &self, int new_seed) {
            self.setSeed(new_seed);
        },
        "Get or set the random seed used by the SamplingEngine."
    )

    .def_property_readonly(
        "vocab_size",
        [](const mmm::sampling::SamplingEngine &self) {
            return self.getVocabSize();
        },
        "Return the vocabulary size associated with the engine."
    )

    .def_property_readonly(
        "eos_token_id",
        [](const mmm::sampling::SamplingEngine &self) {
            return self.getEosTokenId();
        },
        "Return the EOS (end-of-sequence) token id."
    )

    // --- String representation ---
    .def("__repr__", [](const mmm::sampling::SamplingEngine &self) {
        std::ostringstream oss;
        oss << "<SamplingEngine vocab_size=" << self.getVocabSize()
            << " eos_token_id=" << self.getEosTokenId()
            << " seed=" << self.getSeed() << ">";
        return oss.str();
    });

    // ----------------- Tokenizer -----------------
    py::class_<LibTok::MMM>(m, "Tokenizer")
        .def(py::init([](const std::string &tokenizer_path) {
            if (tokenizer_path.empty()) {
                throw std::runtime_error("Tokenizer requires a file path or TokenizerConfig.");
            }
            return std::make_unique<LibTok::MMM>(tokenizer_path, false);
        }), py::arg("tokenizer_path"));


    // ----------------- Score -----------------
    py::class_<LibTok::ScoreType>(m, "Score")
        // Default constructor (optionally load from MIDI)
        .def(py::init<>([](const std::string &midi_path = "") {
            if (midi_path.empty()) {
                return LibTok::ScoreType();
            } else {
                return LibTokUtils::loadScoreFromMidi(std::filesystem::path(midi_path));
            }
        }), py::arg("midi_path") = "")

        // Save method placeholder
        .def("save", [](LibTok::ScoreType &self, const std::string &path) {
            LibTokUtils::saveMidiFromScore(self, std::filesystem::path(path));
        }, py::arg("path"));

    // ----------------- PromptConfig -----------------
    py::class_<mmm::inference::PromptConfig>(m, "PromptConfig")
    .def(py::init([](const py::object &mode_obj, int context_length) {
        mmm::inference::PromptConfig cfg;
        cfg.context_length = context_length;

        // Determine mode type from python object
        if (py::isinstance<py::dict>(mode_obj)) {
            py::dict d = mode_obj.cast<py::dict>();

            // BarInfilling mode
            if (d.contains("bars")) {
                mmm::inference::BarInfilling barCfg;

                for (auto item : d["bars"].cast<py::dict>()) {
                    int track_idx = py::cast<int>(item.first);
                    py::list subsets = py::cast<py::list>(item.second);

                    std::vector<BarSubset> vec_subsets;
                    for (auto subset_obj : subsets) {
                        py::tuple t = py::cast<py::tuple>(subset_obj);
                        int start_bar = py::cast<int>(t[0]);
                        int end_bar = py::cast<int>(t[1]);
                        std::vector<std::string> controls = t[2].cast<std::vector<std::string>>();
                        vec_subsets.emplace_back(start_bar, end_bar, controls);
                    }

                    barCfg.bars[track_idx] = vec_subsets;
                }

                cfg.mode = barCfg;

            // TrackSampling mode
            } else if (d.contains("tracks")) {
                mmm::inference::TrackSampling trackCfg;

                py::list tracks_list = d["tracks"].cast<py::list>();
                for (auto track_obj : tracks_list) {
                    py::tuple t = py::cast<py::tuple>(track_obj);
                    int program = py::cast<int>(t[0]);
                    std::vector<std::string> controls = t[1].cast<std::vector<std::string>>();
                    trackCfg.tracks.emplace_back(program, controls);
                }

                cfg.mode = trackCfg;

            } else {
                throw std::runtime_error("Unknown mode dict keys");
            }
        } else {
            throw std::runtime_error("Mode must be a dict");
        }
        return cfg;
    }), py::arg("mode"), py::arg("context_length") = 4)

    // Properties
    .def_property("context_length",
                  [](const mmm::inference::PromptConfig &cfg) { return cfg.context_length; },
                  [](mmm::inference::PromptConfig &cfg, int val) { cfg.context_length = val; })
    .def("bar_infilling", &mmm::inference::PromptConfig::bar_infilling)
    .def("track_sampling", &mmm::inference::PromptConfig::track_sampling)
    .def("empty", &mmm::inference::PromptConfig::empty)
    .def("bars", [](const mmm::inference::PromptConfig &cfg) -> const auto& {
        if (!cfg.bar_infilling()) throw std::runtime_error("Not in bar_infilling mode");
        return cfg.bars();
    })

    // __str__/__repr__
    .def("__str__", [](const mmm::inference::PromptConfig &cfg) {
        std::ostringstream oss;
        oss << cfg;
        return oss.str();
    })

    // ----------------- JSON/Dict factories -----------------
    .def_static("from_json_str", [](const std::string &json_str) {
        nlohmann::json j = nlohmann::json::parse(json_str);
        mmm::inference::PromptConfig cfg;
        std::string mode = j.at("mode").get<std::string>();
        if (mode == "BarInfilling") {
            mmm::inference::BarInfilling barCfg;
            mmm::utils::from_json(j.at("config"), barCfg);
            cfg.mode = barCfg;
        } else if (mode == "TrackSampling") {
            mmm::inference::TrackSampling trackCfg;
            mmm::utils::from_json(j.at("config"), trackCfg);
            cfg.mode = trackCfg;
        } else {
            throw std::runtime_error("Unknown PromptConfig mode: " + mode);
        }
        if (j.contains("context_length")) cfg.context_length = j.at("context_length").get<int>();
        return cfg;
    })
    .def_static("from_json_file", [](const std::string &path) {
        mmm::inference::PromptConfig cfg;
        mmm::utils::loadPromptConfigFromJson(path, cfg);
        return cfg;
    })
    .def_static("from_dict", [](const py::dict &py_cfg) {
        nlohmann::json j = py_to_json(py_cfg);   // recursive conversion
        mmm::inference::PromptConfig cfg;
        std::string mode = j.at("mode").get<std::string>();
        if (mode == "BarInfilling") {
            mmm::inference::BarInfilling barCfg;
            mmm::utils::from_json(j.at("config"), barCfg);
            cfg.mode = barCfg;
        } else if (mode == "TrackSampling") {
            mmm::inference::TrackSampling trackCfg;
            mmm::utils::from_json(j.at("config"), trackCfg);
            cfg.mode = trackCfg;
        } else {
            throw std::runtime_error("Unknown PromptConfig mode in dict");
        }
        if (j.contains("context_length")) cfg.context_length = j.at("context_length").get<int>();
        return cfg;
    })

    // ----------------- Serialization -----------------
    .def("to_dict", [](const mmm::inference::PromptConfig &cfg) {
        nlohmann::json j;
        j["context_length"] = cfg.context_length;
        if (cfg.bar_infilling()) {
            j["mode"] = "BarInfilling";
            nlohmann::json cfg_j;
            mmm::utils::to_json(cfg_j, std::get<mmm::inference::BarInfilling>(cfg.mode));
            j["config"] = cfg_j;
        } else if (cfg.track_sampling()) {
            j["mode"] = "TrackSampling";
            nlohmann::json cfg_j;
            mmm::utils::to_json(cfg_j, std::get<mmm::inference::TrackSampling>(cfg.mode));
            j["config"] = cfg_j;
        }
        return json_to_py(j);  // <- fully recursive conversion
    })
    .def("to_json", [](const mmm::inference::PromptConfig &cfg) {
        // repeat the same lambda logic
        nlohmann::json j;
        j["context_length"] = cfg.context_length;
        if (cfg.bar_infilling()) {
            j["mode"] = "BarInfilling";
            nlohmann::json cfg_j;
            mmm::utils::to_json(cfg_j, std::get<mmm::inference::BarInfilling>(cfg.mode));
            j["config"] = cfg_j;
        } else if (cfg.track_sampling()) {
            j["mode"] = "TrackSampling";
            nlohmann::json cfg_j;
            mmm::utils::to_json(cfg_j, std::get<mmm::inference::TrackSampling>(cfg.mode));
            j["config"] = cfg_j;
        }
        return j.dump(4);
    })
    .def("save_json", [](const mmm::inference::PromptConfig &cfg, const std::string &path) {
        std::ofstream f(path);
        if (!f.is_open()) throw std::runtime_error("Cannot open file: " + path);
        nlohmann::json j;
        j["context_length"] = cfg.context_length;
        if (cfg.bar_infilling()) {
            j["mode"] = "BarInfilling";
            nlohmann::json cfg_j;
            mmm::utils::to_json(cfg_j, std::get<mmm::inference::BarInfilling>(cfg.mode));
            j["config"] = cfg_j;
        } else if (cfg.track_sampling()) {
            j["mode"] = "TrackSampling";
            nlohmann::json cfg_j;
            mmm::utils::to_json(cfg_j, std::get<mmm::inference::TrackSampling>(cfg.mode));
            j["config"] = cfg_j;
        }
        f << j.dump(4);
    });

    py::class_<mmm::ModelConfig>(m, "ModelConfig")
    .def(py::init<>())

    // Constructor with args
#ifdef USE_ONNX
    .def(py::init<const std::string&, bool, bool, int>(),
#endif
#ifdef USE_TORCH
    .def(py::init<const std::string&, bool, int>(),
#endif
        py::arg("model"),
        py::arg("cached") = false,
#ifdef USE_ONNX
        py::arg("coreml") = false,
#endif
        py::arg("vocab_size") = 0)

    // --- Fields ---
    .def_readwrite("model", &mmm::ModelConfig::path)
    .def_readwrite("cached", &mmm::ModelConfig::cached)
#ifdef USE_ONNX
    .def_readwrite("coreml", &mmm::ModelConfig::coreml)
#endif
    .def_readwrite("vocab_size", &mmm::ModelConfig::vocab_size)

    // --- Constructors from JSON ---
    .def_static("from_file", [](const std::string &path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open JSON file: " + path);
        nlohmann::json j;
        f >> j;
        mmm::ModelConfig cfg;
        cfg.path = j.at("model").get<std::string>();
        if (j.contains("cached")) cfg.cached = j.at("cached").get<bool>();
#ifdef USE_ONNX
        if (j.contains("coreml")) cfg.coreml = j.at("coreml").get<bool>();
#endif
        if (j.contains("vocab_size")) cfg.vocab_size = j.at("vocab_size").get<int>();
        return cfg;
    })
    .def_static("from_json", [](const std::string &json_str) {
        nlohmann::json j = nlohmann::json::parse(json_str);
        mmm::ModelConfig cfg;
        cfg.path = j.at("model").get<std::string>();
        if (j.contains("cached")) cfg.cached = j.at("cached").get<bool>();
#ifdef USE_ONNX
        if (j.contains("coreml")) cfg.coreml = j.at("coreml").get<bool>();
#endif
        if (j.contains("vocab_size")) cfg.vocab_size = j.at("vocab_size").get<int>();
        return cfg;
    })

    // --- from_dict (Python dict) ---
    .def_static("from_dict", [](const py::dict &d) {
        mmm::ModelConfig cfg;
        if (d.contains("model")) cfg.path = d["model"].cast<std::string>();
        if (d.contains("cached")) cfg.cached = d["cached"].cast<bool>();
#ifdef USE_ONNX
        if (d.contains("coreml")) cfg.coreml = d["coreml"].cast<bool>();
#endif
        if (d.contains("vocab_size")) cfg.vocab_size = d["vocab_size"].cast<int>();
        return cfg;
    }, "Create a ModelConfig from a Python dictionary.")

    // --- to_dict (return Python dict) ---
    .def("to_dict", [](const mmm::ModelConfig &cfg) {
        py::dict d;
        d["model"] = cfg.path;
        d["cached"] = cfg.cached;
#ifdef USE_ONNX
        d["coreml"] = cfg.coreml;
#endif
        d["vocab_size"] = cfg.vocab_size;
        return d;
    }, "Return a Python dictionary representation of this ModelConfig.")

    // --- to_json (return JSON string) ---
    .def("to_json", [](const mmm::ModelConfig &cfg, int indent = 2) {
        nlohmann::json j;
        j["model"] = cfg.path;
        j["cached"] = cfg.cached;
#ifdef USE_ONNX
        j["coreml"] = cfg.coreml;
#endif
        j["vocab_size"] = cfg.vocab_size;
        return j.dump(indent);
    }, py::arg("indent") = 2, "Return a JSON string representation of this ModelConfig.")

    // --- save_json (write to file) ---
    .def("save_json", [](const mmm::ModelConfig &cfg, const std::string &path, int indent = 2) {
        nlohmann::json j;
        j["model"] = cfg.path;
        j["cached"] = cfg.cached;
#ifdef USE_ONNX
        j["coreml"] = cfg.coreml;
#endif
        j["vocab_size"] = cfg.vocab_size;
        std::ofstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open file for writing: " + path);
        f << j.dump(indent);
    }, py::arg("path"), py::arg("indent") = 2, "Save this ModelConfig to a JSON file.")
    
    // --- repr for nice printing ---
    .def("__repr__", [](const mmm::ModelConfig &cfg) {
        std::ostringstream oss;
        oss << "<ModelConfig path='" << cfg.path
            << "' cached=" << std::boolalpha << cfg.cached
#ifdef USE_ONNX
            << " coreml=" << std::boolalpha << cfg.coreml
#endif
            << " vocab_size=" << cfg.vocab_size << ">";
        return oss.str();
    });

    // IModel opaque factory with constructor
    py::class_<mmm::IModel, std::shared_ptr<mmm::IModel>>(m, "Model")
        .def(py::init([](const mmm::ModelConfig &cfg) -> std::shared_ptr<mmm::IModel> {
#ifdef USE_ONNX
            if (cfg.cached) {
                return std::make_shared<mmm::CausalLMCached>(cfg.path, cfg.vocab_size, cfg.coreml);
            } else {
                return std::make_shared<mmm::CausalLM>(cfg.path, cfg.vocab_size, cfg.coreml);
            }
#endif
#ifdef USE_TORCH
        if (cfg.cached) {
            return std::make_shared<mmm::CausalLMCached>(cfg.path, cfg.vocab_size);
        } else {
            return std::make_shared<mmm::CausalLM>(cfg.path, cfg.vocab_size);
        }
#endif
        }), py::arg("cfg"), "Construct a model from ModelConfig");

    // --- Expose LogLevel enum
    py::enum_<mmm::utils::LogLevel>(m, "LogLevel")
        .value("FATAL", mmm::utils::LogLevel::FATAL)
        .value("ERROR", mmm::utils::LogLevel::ERROR)
        .value("WARN",  mmm::utils::LogLevel::WARN)
        .value("INFO",  mmm::utils::LogLevel::INFO)
        .value("DEBUG", mmm::utils::LogLevel::DEBUG)
        .value("TRACE", mmm::utils::LogLevel::TRACE)
        .export_values();

    // --- Expose LogMedium enum
    py::enum_<mmm::utils::LogMedium>(m, "LogMedium")
        .value("NONE", mmm::utils::LogMedium::NONE)
        .value("CONSOLE", mmm::utils::LogMedium::CONSOLE)
        .value("FILE", mmm::utils::LogMedium::FILE)
        .value("BOTH", mmm::utils::LogMedium::BOTH)
        .export_values();

    // --- Expose static LoggerGlobal API
    m.def("set_log_level", &LoggerGlobal::setLogLevel, py::arg("level"),
          "Set global log level for the library");

    m.def("set_log_medium", &LoggerGlobal::setMedium, 
          py::arg("level"), py::arg("medium"), py::arg("filename") = "",
          "Set the output medium (console, file, both) for a given log level");

}
