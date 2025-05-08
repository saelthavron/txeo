# Logger

Abstract base class for all logging implementations in the **txeo** project.

---

## Overview

The `Logger` class defines the standard interface and behavior for logging in **txeo**. It supports log-level filtering, message formatting, and enabling/disabling logging globally. This base class should be inherited by specific logger implementations such as `LoggerConsole` or `LoggerFile`.

---

## Features

* ✅ Log-level filtering (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
* ✅ Toggle logging globally (enable/disable)
* ✅ Convenience methods: `info()`, `error()`, `debug()`, `warning()`
* ✅ Virtual `write()` method for subclass customization

---

## Header

```cpp
#include "txeo/Logger.h"
```

---

## Usage Example

```cpp
class MyConsoleLogger : public txeo::Logger {
protected:
    void write(txeo::LogLevel level, const std::string &message) override {
        std::cout << "[" << log_level_str(level) << "] " << message << std::endl;
    }
};

MyConsoleLogger logger;
logger.set_output_level(txeo::LogLevel::INFO);
logger.info("System initialized");
logger.debug("Debug skipped"); // Will be filtered
```

---

## Member Functions

### `void log(LogLevel level, const std::string &message)`

Writes a message if logging is enabled and the level is above threshold.

### `void turn_on()` / `void turn_off()`

Globally enables or disables all logging.

### `LogLevel output_level() const` / `void set_output_level(LogLevel)`

Controls the minimum level required for messages to be shown.

### Level-Specific Helpers

* `void debug(const std::string&)`
* `void info(const std::string&)`
* `void warning(const std::string&)`
* `void error(const std::string&)`

Shortcut methods to log with the corresponding severity level.

### `virtual void write(LogLevel level, const std::string &message)`

Pure virtual method that must be implemented by subclasses.

### `static std::string log_level_str(LogLevel)`

Converts a `LogLevel` to a human-readable string.

---

## Enum: `LogLevel`

The `LogLevel` enum controls severity classification.

```cpp
enum class LogLevel {
    DEBUG,    // Developer diagnostics
    INFO,     // General status messages
    WARNING,  // Warnings about possible problems
    ERROR     // Errors requiring immediate attention
};
```

---

For detailed API references, see individual method documentation at [txeo::Logger](https://txeo-doc.netlify.app/classtxeo_1_1_logger.html).

