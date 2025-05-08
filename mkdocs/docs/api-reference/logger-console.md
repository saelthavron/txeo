# LoggerConsole

Thread-safe singleton logger for console output with colored messages and timestamp formatting.

---

## Overview

The `LoggerConsole` class is a concrete implementation of the `txeo::Logger` interface that writes log messages to the terminal. It formats messages with timestamps, log levels, and optional color codes, and ensures thread safety using an internal mutex.

This logger is implemented as a singleton, meaning only one instance exists throughout the program. It is ideal for simple real-time diagnostics, debugging, or CLI tools.

---

## Features

* ✅ Thread-safe via `std::mutex`
* ✅ Singleton access pattern (`LoggerConsole::instance()`)
* ✅ Colored log levels (if enabled in your implementation)
* ✅ Timestamped output
* ✅ Level-based filtering (via `set_output_level()`)

---

## Header

```cpp
#include "txeo/LoggerConsole.h"
```

---

## Usage Example

```cpp
#include "txeo/LoggerConsole.h"

int main() {
    auto &logger = txeo::LoggerConsole::instance();

    logger.set_output_level(txeo::LogLevel::INFO);

    logger.debug("Debug message");    // Will not be shown if level is INFO
    logger.info("Info message");      // ✅ Visible
    logger.warning("Warning!");       // ✅ Visible
    logger.error("Fatal error!");     // ✅ Visible
}
```

---

## Member Functions

### `static LoggerConsole &instance()`

Returns the singleton instance of the `LoggerConsole`.

#### Example

```cpp
LoggerConsole::instance().info("Logger ready!");
```

---

### `void write(LogLevel level, const std::string &message)`

Overrides the base class `Logger::write()` to print a formatted log message to the console, thread-safely.

> ⚠️ This method is **not meant to be called directly**. Use `log()`, `info()`, `warning()` etc. from the base class.

---

## Thread Safety

All output is protected with a `std::mutex` to ensure that concurrent log writes do not interleave.

---

For detailed API references, see individual method documentation at [txeo::LoggerConsole](https://txeo-doc.netlify.app/classtxeo_1_1_logger_console.html).
