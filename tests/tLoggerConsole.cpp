#include "txeo/LoggerConsole.h"

#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

namespace txeo {

TEST(LoggerConsoleTest, DefaultOutputLevelIsAll) {
  auto &logger = LoggerConsole::instance();
  ASSERT_EQ(logger.output_level(), LogLevel::DEBUG);
}

TEST(LoggerConsoleTest, SetOutputLevel) {
  auto &logger = LoggerConsole::instance();
  logger.set_output_level(LogLevel::INFO);
  ASSERT_EQ(logger.output_level(), LogLevel::INFO);
  logger.set_output_level(LogLevel::DEBUG);
}

TEST(LoggerConsoleTest, WhenTurnedOff_NoMessagesLogged) {
  auto &logger = LoggerConsole::instance();
  logger.turn_off();

  std::stringstream buffer;
  auto old_buf = std::cout.rdbuf(buffer.rdbuf());
  logger.info("Should not appear");
  std::cout.rdbuf(old_buf);

  EXPECT_TRUE(buffer.str().empty());
  logger.turn_on();
}

TEST(LoggerConsoleTest, LogLevelLowerThanOutputLevelNotLogged) {
  auto &logger = LoggerConsole::instance();
  logger.set_output_level(LogLevel::WARNING);

  std::stringstream buffer{};
  auto old_buf = std::cout.rdbuf(buffer.rdbuf());
  logger.info("Info message");
  std::cout.rdbuf(old_buf);

  EXPECT_TRUE(buffer.str().empty());
  logger.set_output_level(LogLevel::DEBUG);
}

TEST(LoggerConsoleTest, LogLevelEqualToOrHigherThanOutputLevelIsLogged) {
  auto &logger = LoggerConsole::instance();
  logger.set_output_level(LogLevel::INFO);

  std::stringstream buffer;
  auto old_buf = std::cout.rdbuf(buffer.rdbuf());
  logger.info("Info message");
  logger.warning("Warning message");
  std::cout.rdbuf(old_buf);

  EXPECT_NE(buffer.str().find("Info message"), std::string::npos);
  EXPECT_NE(buffer.str().find("Warning message"), std::string::npos);
  logger.set_output_level(LogLevel::DEBUG);
}

TEST(LoggerConsoleTest, LogMessagesContainCorrectLevelStrings) {
  auto &logger = LoggerConsole::instance();

  std::stringstream buffer;
  auto old_buf = std::cout.rdbuf(buffer.rdbuf());
  logger.debug("Debug");
  logger.info("Info");
  logger.warning("Warning");
  logger.error("Error");
  std::cout.rdbuf(old_buf);

  const std::string output = buffer.str();
  std::cout << output << std::endl;
  EXPECT_NE(output.find("DEBUG"), std::string::npos);
  EXPECT_NE(output.find("INFO"), std::string::npos);
  EXPECT_NE(output.find("WARNING"), std::string::npos);
  EXPECT_NE(output.find("ERROR"), std::string::npos);
}

TEST(LoggerConsoleTest, SingletonInstanceIsSameAcrossCalls) {
  auto &logger1 = LoggerConsole::instance();
  auto &logger2 = LoggerConsole::instance();
  ASSERT_EQ(&logger1, &logger2);
}

} // namespace txeo