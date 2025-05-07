#include "txeo/LoggerConsole.h"
#include "txeo/Logger.h"
#include "txeo/detail/utils.h"

#include <iostream>
#include <mutex>

namespace txeo {

LoggerConsole &txeo::LoggerConsole::instance() {
  static LoggerConsole logger;
  return logger;
}

void LoggerConsole::write(txeo::LogLevel level, std::string message) {
  std::lock_guard<std::mutex> ld{_mutex};
  std::cout << "[" << txeo::detail::current_time() << "] - " << txeo::Logger::log_level_str(level)
            << ": " << message << std::endl;
}

} // namespace txeo