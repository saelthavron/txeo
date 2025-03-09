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

TEST(MatrixTest, ToMatrixValid2DTensor) {

  txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
  txeo::Tensor<int> tensor2({2, 3}, {1, 2, 3, 4, 5, 6});

  auto result = txeo::Matrix<int>::to_matrix(std::move(tensor));
  auto result2 = txeo::Matrix<int>::to_matrix(tensor2);

  EXPECT_EQ(result.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(result2.shape(), txeo::TensorShape({2, 3}));

  EXPECT_EQ(result(0, 0), 1);
  EXPECT_EQ(result(1, 2), 6);
  EXPECT_EQ(result(0, 1), 2);
  EXPECT_EQ(result2(0, 0), 1);
  EXPECT_EQ(result2(1, 2), 6);
  EXPECT_EQ(result2(0, 1), 2);
}

TEST(MatrixTest, ToMatrixReshape) {

  txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});

  matrix.reshape({3, 2});
  EXPECT_EQ(matrix(2, 1), 6);
  EXPECT_EQ(matrix(1, 1), 4);
  EXPECT_THROW(matrix.reshape({1, 2, 3}), txeo::MatrixError);
}

TEST(MatrixTest, ToMatrixInvalid1DTensor) {

  txeo::Tensor<int> tensor({6}, {1, 2, 3, 4, 5, 6});

  EXPECT_THROW(txeo::Matrix<int>::to_matrix(std::move(tensor)), txeo::MatrixError);
}

TEST(MatrixTest, ToMatrixInvalid3DTensor) {

  txeo::Tensor<int> tensor({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  EXPECT_THROW(txeo::Matrix<int>::to_matrix(std::move(tensor)), txeo::MatrixError);
}

TEST(MatrixTest, ToMatrixEmptyTensor) {

  txeo::Tensor<int> tensor(txeo::TensorShape({}));

  EXPECT_THROW(txeo::Matrix<int>::to_matrix(std::move(tensor)), txeo::MatrixError);
}