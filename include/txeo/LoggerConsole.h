#ifndef LOGGERCONSOLE_H
#define LOGGERCONSOLE_H
#pragma once

#include "txeo/Logger.h"

#include <mutex>

namespace txeo {

/**
 * @class LoggerConsole
 * @brief Thread-safe singleton logger for console output
 *
 * Provides colored console logging with timestamp and level information.
 * Inherits from txeo::Logger and implements thread-safe output operations.
 *
 * @note Automatically enabled on instantiation
 * @note Thread-safe through internal mutex locking
 *
 * **Example Usage:**
 * @code
 * // Get singleton instance
 * auto& console_logger = LoggerConsole::instance();
 *
 * // Configure logging
 * console_logger.set_output_level(LogLevel::INFO);
 *
 * // Log messages
 * console_logger.debug("Debug data");    // Won't print with INFO level
 * console_logger.warning("Low memory!"); // Will print with warning color
 * @endcode
 */
class LoggerConsole : public txeo::Logger {
  public:
    LoggerConsole(const LoggerConsole &) = delete;
    LoggerConsole &operator=(const LoggerConsole &) = delete;
    LoggerConsole(LoggerConsole &&) noexcept = delete;
    LoggerConsole &operator=(LoggerConsole &&) noexcept = delete;
    ~LoggerConsole() = default;

    /**
     * @brief Access the singleton instance
     * @return Reference to the global LoggerConsole instance
     *
     * **Example Usage:**
     * @code
     * // Typical usage pattern
     * LoggerConsole::instance().info("System initialized");
     * @endcode
     */
    static LoggerConsole &instance();

  private:
    LoggerConsole() = default;
    void write(txeo::LogLevel level, const std::string &message) override;
    std::mutex _mutex;
};
} // namespace txeo
#endif