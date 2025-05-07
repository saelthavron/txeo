#ifndef LOGGER_H
#define LOGGER_H
#pragma once

#include <string>

namespace txeo {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class Logger {
  public:
    Logger(const Logger &) = delete;
    Logger(Logger &&) = delete;
    Logger &operator=(const Logger &) = delete;
    Logger &operator=(Logger &&) = delete;
    virtual ~Logger() = default;

    void log(txeo::LogLevel level, const std::string &message);

    void turn_on() { _is_turned_on = true; };
    void turn_off() { _is_turned_on = false; };

    [[nodiscard]] txeo::LogLevel output_level() const { return _output_level; }
    void set_output_level(txeo::LogLevel output_level) { _output_level = output_level; }

    void debug(const std::string &message);
    void info(const std::string &message);
    void warning(const std::string &message);
    void error(const std::string &message);

  protected:
    Logger() = default;
    bool _is_turned_on{true};
    txeo::LogLevel _output_level{txeo::LogLevel::DEBUG};

    static std::string log_level_str(txeo::LogLevel level);
    virtual void write(txeo::LogLevel level, const std::string &message) = 0;
};

} // namespace txeo
#endif