#include "txeo/LoggerFile.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace txeo {
namespace fs = std::filesystem;

const fs::path TEST_LOG = "test_logfile.log";

std::string read_last_line(const fs::path &path) {
  std::ifstream file(path);
  std::string line, last_line;
  while (std::getline(file, line)) {
    if (!line.empty())
      last_line = line;
  }
  return last_line;
}

TEST(LoggerFileTest, SingletonInstanceConsistency) {
  LoggerFile &logger1 = LoggerFile::instance();
  LoggerFile &logger2 = LoggerFile::instance();
  ASSERT_EQ(&logger1, &logger2);
}

TEST(LoggerFileTest, FileCreationOnFirstOpen) {
  if (fs::exists(TEST_LOG))
    fs::remove(TEST_LOG);

  auto &logger = LoggerFile::instance();
  ASSERT_TRUE(logger.open_file(TEST_LOG));
  ASSERT_TRUE(fs::exists(TEST_LOG));
}

TEST(LoggerFileTest, WriteToClosedFileThrows) {
  auto &logger = LoggerFile::instance();
  logger.turn_on();

  logger.close_file();

  EXPECT_THROW({ logger.info("Should throw"); }, LoggerFileError);
}

TEST(LoggerFileTest, LogLevelStringsInOutput) {
  auto &logger = LoggerFile::instance();
  logger.turn_on();
  logger.set_output_level(LogLevel::DEBUG);

  if (!logger.open_file(TEST_LOG)) {
    FAIL() << "Failed to open log file";
  }

  const std::string test_msg = "LEVEL_TEST_";
  logger.debug(test_msg + "debug");
  logger.info(test_msg + "info");
  logger.warning(test_msg + "warning");
  logger.error(test_msg + "error");

  std::ifstream file(TEST_LOG);
  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  EXPECT_NE(content.find("DEBUG"), std::string::npos);
  EXPECT_NE(content.find("INFO"), std::string::npos);
  EXPECT_NE(content.find("WARNING"), std::string::npos);
  EXPECT_NE(content.find("ERROR"), std::string::npos);
}

TEST(LoggerFileTest, LogLevelFiltering) {
  auto &logger = LoggerFile::instance();
  if (!logger.open_file(TEST_LOG))
    FAIL() << "Failed to open log file";
  logger.turn_on();
  logger.set_output_level(LogLevel::WARNING);

  const std::string test_msg = "FILTER_TEST_";
  logger.info(test_msg + "should_not_appear");
  logger.warning(test_msg + "should_appear");

  std::string content = read_last_line(TEST_LOG);
  EXPECT_EQ(content.find(test_msg + "should_not_appear"), std::string::npos);
  EXPECT_NE(content.find(test_msg + "should_appear"), std::string::npos);
}

TEST(LoggerFileTest, ToggleLogging) {
  auto &logger = LoggerFile::instance();
  if (!logger.open_file(TEST_LOG))
    FAIL() << "Failed to open log file";

  logger.turn_off();

  const std::string test_msg = "TOGGLE_TEST";
  logger.error(test_msg);

  std::string content = read_last_line(TEST_LOG);
  EXPECT_EQ(content.find(test_msg), std::string::npos);

  logger.turn_on();
  logger.error(test_msg);
  content = read_last_line(TEST_LOG);
  EXPECT_NE(content.find(test_msg), std::string::npos);
}

TEST(LoggerFileTest, FinalCleanup) {
  if (fs::exists(TEST_LOG)) {
    ASSERT_TRUE(fs::remove(TEST_LOG));
  }
}

// This test only woks if executed alone
//
// TEST(LoggerFileTest, ConcurrentWriteSafety) {
//   auto &logger = LoggerFile::instance();
//   if (!logger.open_file("test_logfile_t.log"))
//     FAIL() << "Failed to open log file";

//   logger.turn_on();

//   constexpr int NUM_THREADS = 10;
//   std::vector<std::thread> threads;
//   std::vector<std::string> messages;

//   for (int i = 0; i < NUM_THREADS; ++i) {
//     messages.push_back("THREAD_" + std::to_string(i));
//   }

//   for (const auto &msg : messages) {
//     threads.emplace_back([&logger, msg]() { logger.info(msg); });
//   }

//   for (auto &t : threads) {
//     t.join();
//   }

//   std::ifstream file("test_logfile_t.log");
//   std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

//   std::cout << content << std::endl;
//   for (const auto &msg : messages) {
//     EXPECT_NE(content.find(msg), std::string::npos);
//   }
// }

} // namespace txeo