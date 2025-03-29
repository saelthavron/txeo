#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>

#include "txeo/Matrix.h"
#include "txeo/MatrixIO.h"
#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"

namespace txeo {
namespace {

namespace fs = std::filesystem;

void create_test_file(const std::string &path, const std::string &content) {
  std::ofstream file{path};
  file << content;
}

class MatrixIOTest : public ::testing::Test {
  protected:
    void SetUp() override { fs::create_directory(test_dir); }

    void TearDown() override { fs::remove_all(test_dir); }

    const std::string test_dir = "test_data";
};

TEST_F(MatrixIOTest, InstanceReadWrite2DInt) {
  const std::string path = test_dir + "/test_int.csv";
  const Matrix<int> original(2, 3, {1, 2, 3, 4, 5, 6});

  MatrixIO io(path);
  io.write_text_file(original);

  auto loaded = io.read_text_file<int>();
  EXPECT_EQ(original.shape(), loaded.shape());
  for (size_t i = 0; i < original.dim(); ++i) {
    EXPECT_EQ(original.data()[i], loaded.data()[i]);
  }
}

TEST_F(MatrixIOTest, StaticReadWrite2DFloat) {
  const std::string path = test_dir + "/test_float.csv";
  const Matrix<float> original(3, 2, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f});

  MatrixIO::write_textfile(original, path);

  auto loaded = MatrixIO::read_textfile<float>(path);
  EXPECT_EQ(original.shape(), loaded.shape());
  for (size_t i = 0; i < original.dim(); ++i) {
    EXPECT_FLOAT_EQ(original.data()[i], loaded.data()[i]);
  }
}

TEST_F(MatrixIOTest, ReadWithHeader) {
  const std::string path = test_dir + "/header_test.csv";
  create_test_file(path, "col1,col2,col3\n1,2,3\n4,5,6");

  auto tensor = MatrixIO::read_textfile<int>(path, ',', true);

  EXPECT_EQ(tensor.shape(), TensorShape({2, 3}));
  EXPECT_EQ(tensor(1, 2), 6);
}

TEST_F(MatrixIOTest, WritePrecision) {
  const std::string path = test_dir + "/precision_test.csv";
  const Matrix<double> original(1, 3, {1.23456789, 2.34567891, 3.45678912});

  MatrixIO::write_textfile(original, 3, path);

  std::ifstream file(path);
  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  EXPECT_EQ(content, "1.235,2.346,3.457");
}

TEST_F(MatrixIOTest, FileNotFoundRead) {
  EXPECT_THROW(MatrixIO::read_textfile<int>("nonexistent.csv"), MatrixIOError);
}

TEST_F(MatrixIOTest, InvalidDataFormat) {
  const std::string path = test_dir + "/invalid_data.csv";
  create_test_file(path, "1,2.5,three\n4,5,6");

  EXPECT_THROW(MatrixIO::read_textfile<int>(path), MatrixIOError);
}

TEST_F(MatrixIOTest, EmptyFile) {
  const std::string path = test_dir + "/empty.csv";
  create_test_file(path, "");

  EXPECT_THROW(MatrixIO::read_textfile<float>(path), MatrixIOError);
}

TEST_F(MatrixIOTest, InconsistentRowLength) {
  const std::string path = test_dir + "/inconsistent.csv";
  create_test_file(path, "1,2,3\n4,5");

  EXPECT_THROW(MatrixIO::read_textfile<int>(path), MatrixIOError);
}

TEST_F(MatrixIOTest, FloatingPointRequires) {
  const Matrix<float> float_tensor(2, 2);

  EXPECT_TRUE((requires { MatrixIO::write_textfile(float_tensor, 2, "test.csv"); }));
}

TEST_F(MatrixIOTest, OneHotEncode) {
  std::filesystem::path input_path{"../../../../tests/test_data/files/insurance.csv"};
  std::filesystem::path output_path{test_dir + "/test_insurance_one_hot.csv"};
  auto io = MatrixIO::one_hot_encode_text_file(input_path, ',', true, output_path);
  auto matrix = io.read_text_file<float>(true);

  EXPECT_EQ(matrix.dim(), 55);
  EXPECT_EQ(matrix(3, 1), 1.0f);
  EXPECT_EQ(matrix(4, 9), 0.0f);
  EXPECT_THROW(MatrixIO::one_hot_encode_text_file("nofile.txt", ',', true, output_path),
               MatrixIOError);

  const std::string path = test_dir + "/inconsistent.csv";
  create_test_file(path, "1,2,3\n4,5");

  EXPECT_THROW(MatrixIO::one_hot_encode_text_file(path, ',', false, output_path), MatrixIOError);
}

} // namespace
} // namespace txeo