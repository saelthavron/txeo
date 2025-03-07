#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include <gtest/gtest.h>

TEST(MatrixTest, ParameterizedConstructor) {
  txeo::Matrix<int> matrix(2, 3);
  EXPECT_EQ(matrix.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(matrix.size(), 6);
}

TEST(MatrixTest, ParameterizedConstructorWithFillValue) {
  txeo::Matrix<int> matrix(2, 3, 5);
  EXPECT_EQ(matrix.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(matrix(0, 0), 5);
  EXPECT_EQ(matrix(1, 2), 5);
}

TEST(MatrixTest, ParameterizedConstructorWithInitializerList) {
  txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});
  EXPECT_EQ(matrix.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(matrix(0, 0), 1);
  EXPECT_EQ(matrix(0, 1), 2);
  EXPECT_EQ(matrix(0, 2), 3);
  EXPECT_EQ(matrix(1, 0), 4);
  EXPECT_EQ(matrix(1, 1), 5);
  EXPECT_EQ(matrix(1, 2), 6);
}

TEST(MatrixTest, ConstructorWithNestedInitializerList) {
  txeo::Matrix<int> matrix({{1, 2, 3}, {4, 5, 6}});
  EXPECT_EQ(matrix.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(matrix(0, 0), 1);
  EXPECT_EQ(matrix(0, 1), 2);
  EXPECT_EQ(matrix(0, 2), 3);
  EXPECT_EQ(matrix(1, 0), 4);
  EXPECT_EQ(matrix(1, 1), 5);
  EXPECT_EQ(matrix(1, 2), 6);
}

TEST(MatrixTest, CopyConstructor) {
  txeo::Matrix<int> matrix1(2, 3, {1, 2, 3, 4, 5, 6});
  txeo::Matrix<int> matrix2(matrix1);
  EXPECT_EQ(matrix2.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(matrix2(0, 0), 1);
  EXPECT_EQ(matrix2(1, 2), 6);
}

TEST(MatrixTest, MoveConstructor) {
  txeo::Matrix<int> matrix1(2, 3, {1, 2, 3, 4, 5, 6});
  txeo::Matrix<int> matrix2(std::move(matrix1));
  EXPECT_EQ(matrix2.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(matrix2(0, 0), 1);
  EXPECT_EQ(matrix2(1, 2), 6);
}

TEST(MatrixTest, CopyAssignmentOperator) {
  txeo::Matrix<int> matrix1(2, 3, {1, 2, 3, 4, 5, 6});
  txeo::Matrix<int> matrix2(1, 1);
  matrix2 = matrix1;
  EXPECT_EQ(matrix2.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(matrix2(0, 0), 1);
  EXPECT_EQ(matrix2(1, 2), 6);
}

TEST(MatrixTest, MoveAssignmentOperator) {
  txeo::Matrix<int> matrix1(2, 3, {1, 2, 3, 4, 5, 6});
  txeo::Matrix<int> matrix2(1, 1);
  matrix2 = std::move(matrix1);
  EXPECT_EQ(matrix2.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(matrix2(0, 0), 1);
  EXPECT_EQ(matrix2(1, 2), 6);
}

TEST(MatrixTest, MoveConstructorFromTensor) {
  txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});

  txeo::Matrix<int> matrix(std::move(tensor));

  EXPECT_EQ(matrix.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(matrix(0, 0), 1);
  EXPECT_EQ(matrix(0, 1), 2);
  EXPECT_EQ(matrix(0, 2), 3);
  EXPECT_EQ(matrix(1, 0), 4);
  EXPECT_EQ(matrix(1, 1), 5);
  EXPECT_EQ(matrix(1, 2), 6);

  txeo::Tensor<int> cube({1, 1, 1}, {1});
  EXPECT_THROW(txeo::Matrix<int>(std::move(cube)), txeo::MatrixError);
}
