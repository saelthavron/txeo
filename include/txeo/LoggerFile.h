#ifndef LOGGERFILE_H
#define LOGGERFILE_H
#pragma once

#include "txeo/Logger.h"

#include <filesystem>
#include <fstream>

namespace txeo {

/**
 * @class LoggerFile
 * @brief Singleton logger implementation for file output
 *
 * Inherits from txeo::Logger and provides thread-safe file logging capabilities.
 * Uses RAII for resource management and follows the singleton pattern.
 *
 * @note The logger must be explicitly opened with open_file() before use
 * @note All logging operations are thread-safe through internal mutex
 *
 * **Example Usage:**
 * @code
 * try {
 *     auto& logger = LoggerFile::instance();
 *     logger.open_file("app.log");
 *     logger.info("System initialized");
 *     logger.warning("Low memory detected");
 *     logger.close_file();
 * } catch(const LoggerFileError& e) {
 *     std::cerr << "Logging failed: " << e.what() << std::endl;
 * }
 * @endcode
 */
class LoggerFile : public txeo::Logger {
  public:
    LoggerFile(const LoggerFile &) = delete;
    LoggerFile(LoggerFile &&) = delete;
    LoggerFile &operator=(const LoggerFile &) = delete;
    LoggerFile &operator=(LoggerFile &&) = delete;
    ~LoggerFile();

    /**
     * @brief Open log file for writing
     * @param file_path Path to log file (will be created if not exists)
     * @return true if file was successfully opened
     * @exception LoggerFileError Thrown if file opening fails
     * @note Appends to existing file by default
     *
     * **Example Usage:**
     * @code
     * if(!logger.open_file("debug.log")) {
     *     // Handle open failure
     * }
     * @endcode
     */
    bool open_file(const std::filesystem::path &file_path);

    /**
     * @brief Close the current log file
     *
     * **Example Usage:**
     * @code
     * logger.close_file(); // Explicit close
     * @endcode
     */
    void close_file();

    /**
     * @brief Get singleton instance
     * @return Reference to the singleton LoggerFile instance
     *
     * **Example Usage:**
     * @code
     * auto& logger = LoggerFile::instance();
     * @endcode
     */
    static LoggerFile &instance();

  private:
    LoggerFile() = default;
    std::ofstream _wf;
    void write(txeo::LogLevel level, const std::string &message) override;
    std::mutex _mutex;
};

/**
 * @class LoggerFileError
 * @brief Exception class for file logging errors
 *
 * Thrown during file operations or write failures. Inherits from std::runtime_error.
 *
 * **Example Usage:**
 * @code
 * try {
 *     logger.debug("Sensor reading");
 * } catch(const LoggerFileError& e) {
 *     handle_error(e.what());
 * }
 * @endcode
 */
class LoggerFileError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif
