#ifndef LOGGERFILE_H
#define LOGGERFILE_H
#pragma once

#include "txeo/Logger.h"

namespace txeo {

class LoggerFile : public txeo::Logger {
  public:
    LoggerFile() = default;
    LoggerFile(const LoggerFile &) = delete;
    LoggerFile(LoggerFile &&) = delete;
    LoggerFile &operator=(const LoggerFile &) = delete;
    LoggerFile &operator=(LoggerFile &&) = delete;
    ~LoggerFile() = default;

  private:
    void write(txeo::LogLevel level, std::string message) override;
    std::mutex _mutex;
};

} // namespace txeo

#endif