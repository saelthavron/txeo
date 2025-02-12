#include "txeo/TensorIO.h"
#include "txeo/TensorShape.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace txeo {
namespace {

namespace fs = std::filesystem;

void create_test_file(const std::string &path, const std::string &content) {
  std::ofstream file{path};
  file << content;
}

class TensorIOTest : public ::testing::Test {
  protected:
    void SetUp() override { fs::create_directory(test_dir); }

    void TearDown() override { fs::remove_all(test_dir); }

    const std::string test_dir = "test_data";
};

TEST_F(TensorIOTest, InstanceReadWrite2DInt) {
  const std::string path = test_dir + "/test_int.csv";
  const Tensor<int> original(txeo::TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});

  TensorIO io(path);
  io.write_text_file(original);

  auto loaded = io.read_text_file<int>();
  EXPECT_EQ(original.shape(), loaded.shape());
  for (size_t i = 0; i < original.dim(); ++i) {
    EXPECT_EQ(original.data()[i], loaded.data()[i]);
  }
}

TEST_F(TensorIOTest, StaticReadWrite2DFloat) {
  const std::string path = test_dir + "/test_float.csv";
  const Tensor<float> original(txeo::TensorShape({3, 2}), {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f});

  TensorIO::write_textfile(original, path);

  auto loaded = TensorIO::read_textfile<float>(path);
  EXPECT_EQ(original.shape(), loaded.shape());
  for (size_t i = 0; i < original.dim(); ++i) {
    EXPECT_FLOAT_EQ(original.data()[i], loaded.data()[i]);
  }
}

TEST_F(TensorIOTest, ReadWithHeader) {
  const std::string path = test_dir + "/header_test.csv";
  create_test_file(path, "col1,col2,col3\n1,2,3\n4,5,6");

  auto tensor = TensorIO::read_textfile<int>(path, ',', true);

  EXPECT_EQ(tensor.shape(), TensorShape({2, 3}));
  EXPECT_EQ(tensor(1, 2), 6);
}

TEST_F(TensorIOTest, WritePrecision) {
  const std::string path = test_dir + "/precision_test.csv";
  const Tensor<double> original(txeo::TensorShape({1, 3}), {1.23456789, 2.34567891, 3.45678912});

  TensorIO::write_textfile(original, 3, path);

  std::ifstream file(path);
  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_EQ(content, "1.235,2.346,3.457");
}

TEST_F(TensorIOTest, InvalidDimensionWrite) {
  const std::string path = test_dir + "/invalid.csv";
  Tensor<float> tensor({1, 2, 3});

  EXPECT_THROW(TensorIO::write_textfile(tensor, path), TensorIOError);
}

TEST_F(TensorIOTest, FileNotFoundRead) {
  EXPECT_THROW(TensorIO::read_textfile<int>("nonexistent.csv"), TensorIOError);
}

TEST_F(TensorIOTest, InvalidDataFormat) {
  const std::string path = test_dir + "/invalid_data.csv";
  create_test_file(path, "1,2.5,three\n4,5,6");

  EXPECT_THROW(TensorIO::read_textfile<int>(path), TensorIOError);
}

TEST_F(TensorIOTest, EmptyFile) {
  const std::string path = test_dir + "/empty.csv";
  create_test_file(path, "");

  EXPECT_THROW(TensorIO::read_textfile<float>(path), TensorIOError);
}

TEST_F(TensorIOTest, InconsistentRowLength) {
  const std::string path = test_dir + "/inconsistent.csv";
  create_test_file(path, "1,2,3\n4,5");

  EXPECT_THROW(TensorIO::read_textfile<int>(path), TensorIOError);
}

TEST_F(TensorIOTest, FloatingPointRequires) {
  // Should only compile for floating point types
  const Tensor<float> float_tensor({2, 2});

  EXPECT_TRUE((requires { TensorIO::write_textfile(float_tensor, 2, "test.csv"); }));
}

} // namespace
} // namespace txeo