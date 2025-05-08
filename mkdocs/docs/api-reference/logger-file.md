# LoggerFile

Singleton logger for writing formatted log messages to a file.

---

## Overview

The `LoggerFile` class is a thread-safe singleton that implements the `txeo::Logger` interface for persistent file-based logging. It supports log-level filtering, timestamped messages, and mutex-protected writes. The log file must be explicitly opened before use.

This class is ideal for server logs, diagnostics, or long-running processes where console logging is insufficient.

---

## Features

* ✅ Thread-safe using internal `std::mutex`
* ✅ Singleton access (`LoggerFile::instance()`)
* ✅ Timestamped log messages
* ✅ RAII for automatic file cleanup
* ✅ Custom error handling with `LoggerFileError`

---

## Header

```cpp
#include "txeo/LoggerFile.h"
```

---

## Usage Example

```cpp
#include "txeo/LoggerFile.h"

int main() {
    try {
        auto &logger = txeo::LoggerFile::instance();
        logger.open_file("log.txt");

        logger.info("Application started");
        logger.warning("Disk space low");

        logger.close_file();
    } catch (const txeo::LoggerFileError& e) {
        std::cerr << "Logging error: " << e.what() << std::endl;
    }
}
```

---

## Member Functions

### `static LoggerFile &instance()`

Returns the singleton instance of the logger.

```cpp
auto& logger = LoggerFile::instance();
```

### `bool open_file(const std::filesystem::path &file_path)`

Opens a file for logging output. Throws `LoggerFileError` if it fails.

### `void close_file()`

Closes the current log file. Automatically called in destructor if still open.

### `void write(LogLevel level, const std::string &message)`

Writes a message to the open file with timestamp and log level formatting.

> ⚠️ This function is not intended to be called directly; use `log()`, `info()`, `warning()`, etc.

---

## Exception: `LoggerFileError`

Thrown when a file cannot be opened or logging fails.

```cpp
try {
    logger.open_file("invalid/path/log.txt");
} catch (const txeo::LoggerFileError& err) {
    std::cerr << err.what();
}
```

---

For detailed API references, see individual method documentation at [txeo::LoggerFile](https://txeo-doc.netlify.app/classtxeo_1_1_logger_file.html).
