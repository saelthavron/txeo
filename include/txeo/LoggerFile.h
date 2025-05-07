#ifndef LOGGERFILE_H
#define LOGGERFILE_H
#pragma once

#include "txeo/Logger.h"

#include <filesystem>
#include <fstream>

namespace txeo {

class LoggerFile : public txeo::Logger {
  public:
    LoggerFile(const LoggerFile &) = delete;
    LoggerFile(LoggerFile &&) = delete;
    LoggerFile &operator=(const LoggerFile &) = delete;
    LoggerFile &operator=(LoggerFile &&) = delete;
    ~LoggerFile();

    bool open_file(const std::filesystem::path &file_path);

    void close_file();

    static LoggerFile &instance();

  private:
    LoggerFile() = default;
    std::ofstream _wf;
    void write(txeo::LogLevel level, const std::string &message) override;
    std::mutex _mutex;
};

class LoggerFileError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif