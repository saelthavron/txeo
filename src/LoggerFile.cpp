#include "txeo/LoggerFile.h"
#include <mutex>

namespace txeo {

LoggerFile &txeo::LoggerFile::instance() {
  static LoggerFile logger{};
  return logger;
}

LoggerFile::~LoggerFile() {
  if (_wf.is_open())
    _wf.close();
}

bool LoggerFile::open_file(const std::filesystem::path &file_path) {
  std::lock_guard<std::mutex> lg{_mutex};

  if (!this->_is_turned_on)
    return false;

  if (_wf.is_open())
    return true;

  _wf = std::ofstream{file_path};
  if (!_wf.is_open())
    throw txeo::LoggerFileError("Log file could not be opened");

  return _wf.is_open();
}

void LoggerFile::close_file() {
  std::lock_guard<std::mutex> lg{_mutex};
  if (_wf.is_open())
    _wf.close();
}

void LoggerFile::write(txeo::LogLevel level, const std::string &message) {
  std::lock_guard<std::mutex> lg{_mutex};
  if (!this->_is_turned_on)
    return;

  if (!_wf.is_open())
    throw txeo::LoggerFileError("Log file not opened");

  _wf << "[" << txeo::detail::current_time() << "] - " << txeo::Logger::log_level_str(level) << ": "
      << message << std::endl;
}

} // namespace txeo