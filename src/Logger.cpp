#include "txeo/Logger.h"

namespace txeo {

void Logger::log(txeo::LogLevel level, const std::string &message) {
  if (_is_turned_on && static_cast<int>(level) >= static_cast<int>(_output_level))
    this->write(level, message);
}

void Logger::debug(const std::string &message) {
  this->log(txeo::LogLevel::DEBUG, message);
}

void Logger::info(const std::string &message) {
  this->log(txeo::LogLevel::INFO, message);
}

void Logger::warning(const std::string &message) {
  this->log(txeo::LogLevel::WARNING, message);
}

void Logger::error(const std::string &message) {
  this->log(txeo::LogLevel::ERROR, message);
}

std::string Logger::log_level_str(txeo::LogLevel level) {
  switch (level) {
  case txeo::LogLevel::DEBUG:
    return "[~] DEBUG";
  case txeo::LogLevel::INFO:
    return "[✓] INFO";
  case txeo::LogLevel::WARNING:
    return "[!] WARNING";
  case txeo::LogLevel::ERROR:
    return "[x] ERROR";
  }
}

} // namespace txeo