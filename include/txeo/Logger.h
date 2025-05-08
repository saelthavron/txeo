#ifndef LOGGER_H
#define LOGGER_H
#pragma once

#include <string>

namespace txeo {

/**
 * @enum LogLevel
 * @brief Defines severity levels for log messages
 *
 * Ordered from most verbose to most critical:
 * @li DEBUG - Diagnostic information for developers
 * @li INFO - General operational messages
 * @li WARNING - Indicates potential issues
 * @li ERROR - Critical problems requiring attention
 */
enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

/**
 * @class Logger
 * @brief Abstract base class for logging subsystems
 *
 * Provides common interface and functionality for concrete loggers.
 * Supports severity filtering and global enable/disable.
 *
 * @note Inherit from this class to create specific logger implementations
 *
 * **Example Usage:**
 * @code
 * class ConsoleLogger : public Logger {
 * protected:
 *     void write(LogLevel level, const std::string& message) override {
 *         std::cout << "[" << log_level_str(level) << "] " << message << "\n";
 *     }
 * };
 *
 * ConsoleLogger logger;
 * logger.set_output_level(LogLevel::INFO);
 * logger.info("Application started");
 * logger.debug("This won't be shown"); // Filtered by output level
 * @endcode
 */
class Logger {
  public:
    Logger(const Logger &) = delete;
    Logger(Logger &&) = delete;
    Logger &operator=(const Logger &) = delete;
    Logger &operator=(Logger &&) = delete;
    virtual ~Logger() = default;

    /**
     * @brief Main logging method
     * @param level Severity level of the message
     * @param message Content to log
     *
     * @note Respects both output level and enabled state
     */
    void log(txeo::LogLevel level, const std::string &message);

    /**
     * @brief Enable logging operations
     */
    void turn_on() { _is_turned_on = true; };

    /**
     * @brief Disable all logging output
     */
    void turn_off() { _is_turned_on = false; };

    /**
     * @brief Get current output level threshold
     * @return Active LogLevel filter
     */
    [[nodiscard]] txeo::LogLevel output_level() const { return _output_level; }

    /**
     * @brief Set minimum logging level to output
     * @param output_level Messages below this level will be filtered
     *
     * **Example Usage:**
     * @code
     * logger.set_output_level(LogLevel::WARNING); // Only show WARNING+
     * @endcode
     */
    void set_output_level(txeo::LogLevel output_level) { _output_level = output_level; }

    /**
     * @brief Log DEBUG level message
     * @param message Diagnostic information
     */
    void debug(const std::string &message);

    /**
     * @brief Log INFO level message
     * @param message Operational status update
     */
    void info(const std::string &message);

    /**
     * @brief Log WARNING level message
     * @param message Potential issue notification
     */
    void warning(const std::string &message);

    /**
     * @brief Log ERROR level message
     * @param message Critical error report
     */
    void error(const std::string &message);

  protected:
    Logger() = default;
    bool _is_turned_on{true};
    txeo::LogLevel _output_level{txeo::LogLevel::DEBUG};

    static std::string log_level_str(txeo::LogLevel level);

    /**
     * @brief Abstract write operation
     * @param level Message severity level
     * @param message Formatted log content
     *
     * @note Must be implemented in derived classes
     */
    virtual void write(txeo::LogLevel level, const std::string &message) = 0;
};

} // namespace txeo
#endif