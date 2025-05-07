#ifndef LOGGERCONSOLE_H
#define LOGGERCONSOLE_H
#pragma once

#include "txeo/Logger.h"

#include <mutex>

namespace txeo {

class LoggerConsole : public txeo::Logger {
  public:
    LoggerConsole() = default;
    LoggerConsole(const LoggerConsole &) = delete;
    LoggerConsole &operator=(const LoggerConsole &) = delete;
    LoggerConsole(LoggerConsole &&) noexcept = delete;
    LoggerConsole &operator=(LoggerConsole &&) noexcept = delete;
    ~LoggerConsole() = default;

    static LoggerConsole &instance();

  private:
    void write(txeo::LogLevel level, std::string message) override;
    std::mutex _mutex;
};
} // namespace txeo
#endif