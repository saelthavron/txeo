#ifndef LOGGERCONSOLE_H
#define LOGGERCONSOLE_H
#pragma once

#include "txeo/Logger.h"

#include <mutex>

namespace txeo {

class LoggerConsole : public txeo::Logger {
  public:
    LoggerConsole(const LoggerConsole &) = delete;
    LoggerConsole &operator=(const LoggerConsole &) = delete;
    LoggerConsole(LoggerConsole &&) noexcept = delete;
    LoggerConsole &operator=(LoggerConsole &&) noexcept = delete;
    ~LoggerConsole() = default;

    static LoggerConsole &instance();

  private:
    LoggerConsole() = default;
    void write(txeo::LogLevel level, const std::string &message) override;
    std::mutex _mutex;
};
} // namespace txeo
#endif