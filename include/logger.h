#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace mmm::utils {

enum class LogLevel {
    FATAL = 0,
    ERROR = 1,
    WARN  = 2,
    INFO  = 3,
    DEBUG = 4,
    TRACE = 5
};

enum class LogMedium {
    NONE,
    CONSOLE,
    FILE,
    BOTH
};

class Logger {
public:
    explicit Logger(LogLevel maxLevel = LogLevel::INFO)
        : maxLevel(maxLevel) {}

    ~Logger() {
        for (auto &f : fileStreams) {
            if (f.second && f.second->is_open())
                f.second->close();
        }
    }

    void setMaxLevel(LogLevel level) { maxLevel = level; }

    void setMedium(LogLevel level, LogMedium medium, const std::string& filename = "") {
        mediums[level] = medium;
        if ((medium == LogMedium::FILE || medium == LogMedium::BOTH) && !filename.empty()) {
            std::lock_guard<std::mutex> lock(mu);
            fileStreams[level] = std::make_unique<std::ofstream>(filename, std::ios::app);
        }
    }

    void log(LogLevel level, const std::string& msg) {
        if ((int)level > (int)maxLevel) return;

        std::lock_guard<std::mutex> lock(mu);
        LogMedium medium = mediums.count(level) ? mediums[level] : LogMedium::CONSOLE;

        std::string fullMsg = format(level, msg);

        switch (medium) {
            case LogMedium::CONSOLE:
                std::cout << fullMsg << std::endl;
                break;
            case LogMedium::FILE:
                if (fileStreams.count(level) && fileStreams[level]->is_open())
                    (*fileStreams[level]) << stripAnsi(fullMsg) << std::endl;
                break;
            case LogMedium::BOTH:
                std::cout << fullMsg << std::endl;
                if (fileStreams.count(level) && fileStreams[level]->is_open())
                    (*fileStreams[level]) << stripAnsi(fullMsg) << std::endl;
                break;
            case LogMedium::NONE:
            default:
                break;
        }
    }

private:
    LogLevel maxLevel;
    std::unordered_map<LogLevel, LogMedium> mediums;
    std::unordered_map<LogLevel, std::unique_ptr<std::ofstream>> fileStreams;
    std::mutex mu;

    // ANSI codes for colors and bold
    static std::string colorCode(LogLevel level) {
        switch(level) {
            case LogLevel::FATAL: return "\033[1;41m"; // bold red background
            case LogLevel::ERROR: return "\033[1;31m"; // bold red
            case LogLevel::WARN:  return "\033[1;33m"; // bold yellow
            case LogLevel::INFO:  return "\033[1;32m"; // bold green
            case LogLevel::DEBUG: return "\033[1;34m"; // bold blue
            case LogLevel::TRACE: return "\033[1;37m"; // bold white
            default: return "\033[0m"; // reset
        }
    }

    static std::string prefix(LogLevel level) {
        switch (level) {
            case LogLevel::FATAL: return "[FATAL] ";
            case LogLevel::ERROR: return "[ERROR] ";
            case LogLevel::WARN:  return "[WARN] ";
            case LogLevel::INFO:  return "[INFO] ";
            case LogLevel::DEBUG: return "[DEBUG] ";
            case LogLevel::TRACE: return "[TRACE] ";
            default: return "";
        }
    }

    static std::string colorBoldPrefix(LogLevel level) {
        return colorCode(level) + prefix(level) + "\033[0m";
    }

    std::string format(LogLevel level, const std::string& msg) {
        return colorBoldPrefix(level) + msg; // colored + bold prefix
    }

    static std::string stripAnsi(const std::string& str) {
        std::string out;
        bool esc = false;
        for (char c : str) {
            if (c == '\033') { esc = true; continue; }
            if (esc) {
                if (c == 'm') esc = false;
                continue;
            }
            out += c;
        }
        return out;
    }
};

} // namespace mmm::utils
