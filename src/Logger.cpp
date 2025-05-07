#include "txeo/Logger.h"

namespace txeo {

void Logger::log(txeo::LogLevel level, const std::string &message) {
  if (_is_turned_on || static_cast<int>(level) >= static_cast<int>(_output_level))
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

} // namespace txeo