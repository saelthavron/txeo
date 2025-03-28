#include <gtest/gtest.h>
#include <initializer_list>
#include <utility>
#include <vector>

#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/TensorOp.h"
#include "txeo/TensorShape.h"
#include "txeo/Vector.h"
#include "txeo/types.h"

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

TEST(MatrixTest, ToTensorRvalue) {
  txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});
  txeo::Tensor<int> tensor = txeo::Matrix<int>::to_tensor(std::move(matrix));

  ASSERT_EQ(tensor.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(tensor(0, 0), 1);
  EXPECT_EQ(tensor(0, 1), 2);
  EXPECT_EQ(tensor(0, 2), 3);
  EXPECT_EQ(tensor(1, 0), 4);
  EXPECT_EQ(tensor(1, 1), 5);
  EXPECT_EQ(tensor(1, 2), 6);
}

TEST(MatrixTest, ToTensorConstRef) {
  const txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});
  txeo::Tensor<int> tensor = txeo::Matrix<int>::to_tensor(matrix);

  ASSERT_EQ(tensor.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(tensor(0, 0), 1);
  EXPECT_EQ(tensor(0, 1), 2);
  EXPECT_EQ(tensor(0, 2), 3);
  EXPECT_EQ(tensor(1, 0), 4);
  EXPECT_EQ(tensor(1, 1), 5);
  EXPECT_EQ(tensor(1, 2), 6);
}

TEST(MatrixTest, Normalization) {

  txeo::Matrix<double> mat(3, 3, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  mat.normalize_columns(txeo::NormalizationType::MIN_MAX);
  txeo::Matrix<double> resp(3, 3, {0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1});
  EXPECT_TRUE(mat == resp);

  txeo::Matrix<double> mat1(3, 3, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  mat1.normalize_rows(txeo::NormalizationType::Z_SCORE);
  txeo::Matrix<double> resp1(3, 3, {-1, 0, 1, -1, 0, 1, -1, 0, 1});
  EXPECT_TRUE(mat1 == resp1);
}

TEST(MatrixTest, AdditionMatrixMatrix) {
  txeo::Matrix<int> m1(2, 2, {1, 2, 3, 4});
  txeo::Matrix<int> m2(2, 2, {5, 6, 7, 8});
  auto result = m1 + m2;

  EXPECT_EQ(result.shape(), txeo::TensorShape({2, 2}));
  EXPECT_EQ(result.data()[0], 6);
  EXPECT_EQ(result.data()[1], 8);
  EXPECT_EQ(result.data()[2], 10);
  EXPECT_EQ(result.data()[3], 12);
}

TEST(MatrixTest, AdditionMatrixScalar) {
  txeo::Matrix<double> m(2, 2, {1.5, 2.5, 3.5, 4.5});
  auto result = m + 2.5;

  EXPECT_EQ(result.shape(), txeo::TensorShape({2, 2}));
  EXPECT_DOUBLE_EQ(result.data()[0], 4.0);
  EXPECT_DOUBLE_EQ(result.data()[3], 7.0);
}

TEST(MatrixTest, SubtractionMatrixMatrix) {
  txeo::Matrix<float> m1(1, 3, {10.0f, 20.0f, 30.0f});
  txeo::Matrix<float> m2(1, 3, {1.0f, 2.0f, 3.0f});
  auto result = m1 - m2;

  EXPECT_EQ(result.shape(), txeo::TensorShape({1, 3}));
  EXPECT_FLOAT_EQ(result.data()[1], 18.0f);
}

TEST(MatrixTest, SubtractionMatrixScalar) {
  txeo::Matrix<int> m(2, 2, {5, 10, 15, 20});
  auto result = m - 3;

  EXPECT_EQ(result.data()[0], 2);
  EXPECT_EQ(result.data()[1], 7);
  EXPECT_EQ(result.data()[2], 12);
  EXPECT_EQ(result.data()[3], 17);
}

TEST(MatrixTest, SubtractionScalarMatrix) {
  txeo::Matrix<int> m(2, 2, {1, 2, 3, 4});
  auto result = 10 - m;

  EXPECT_EQ(result.data()[0], 9);
  EXPECT_EQ(result.data()[1], 8);
  EXPECT_EQ(result.data()[2], 7);
  EXPECT_EQ(result.data()[3], 6);
}

TEST(MatrixTest, MultiplicationMatrixScalar) {
  txeo::Matrix<int> m(2, 3, {2, 3, 4, 5, 6, 7});
  auto result = m * 3;

  txeo::Matrix<int> t1(2, 3, {1, 2, 3, 4, 5, 6});

  EXPECT_EQ(result.data()[0], 6);
  EXPECT_EQ(result.data()[1], 9);
  EXPECT_EQ(result.data()[2], 12);
  EXPECT_EQ(result.data()[3], 15);
  EXPECT_EQ(result.data()[4], 18);
  EXPECT_EQ(result.data()[5], 21);
  EXPECT_TRUE((3 * t1) == txeo::Matrix<int>(2, 3, {3, 6, 9, 12, 15, 18}));
}

TEST(MatrixTest, DivisionMatrixScalar) {
  txeo::Matrix<double> m(1, 4, {10.0, 20.0, 30.0, 40.0});
  auto result = m / 2.0;

  EXPECT_DOUBLE_EQ(result.data()[2], 15.0);
}

TEST(MatrixTest, DivisionScalarMatrix) {
  txeo::Matrix<int> m(2, 2, {2, 4, 5, 10});
  auto result = 100 / m;

  EXPECT_EQ(result.data()[0], 50);
  EXPECT_EQ(result.data()[1], 25);
  EXPECT_EQ(result.data()[2], 20);
  EXPECT_EQ(result.data()[3], 10);
}

TEST(MatrixTest, DefaultMatrixOperations) {
  txeo::Matrix<float> defa;
  auto result_add = defa + 5.0f;
  auto result_mul = defa * 2.0f;

  EXPECT_EQ(result_add(0, 0), 5.0f);
  EXPECT_EQ(result_mul(0, 0), 0.0f);
}

TEST(MatrixTest, MixedOperations) {
  txeo::Matrix<int> m1(2, 2, {5, 10, 15, 20});
  txeo::Matrix<int> m2(2, 2, {1, 2, 3, 4});

  auto result = (m1 - m2) * 2 + 10;

  EXPECT_EQ(result.data()[0], 18);
  EXPECT_EQ(result.data()[1], 26);
  EXPECT_EQ(result.data()[2], 34);
  EXPECT_EQ(result.data()[3], 42);
}

TEST(MatrixTest, FloatingPointPrecision) {
  txeo::Matrix<double> m(1, 2, {1.0, 2.0});
  auto result = m / 3.0;

  EXPECT_DOUBLE_EQ(result.data()[0], 1.0 / 3.0);
  EXPECT_DOUBLE_EQ(result.data()[1], 2.0 / 3.0);
}

TEST(MatrixTest, BooleanMatrixOperations) {
  txeo::Matrix<bool> m1(2, 2, {true, false, true, false});
  txeo::Matrix<bool> m2(2, 2, {true, true, false, false});
  auto result = m1 - m2;

  EXPECT_EQ(result.data()[0], false);
  EXPECT_EQ(result.data()[1], true);
  EXPECT_EQ(result.data()[2], true);
  EXPECT_EQ(result.data()[3], false);
}

TEST(MatrixTest, Transpose2x3Matrix) {
  txeo::Matrix<int> mat(2, 3, {1, 2, 3, 4, 5, 6});
  mat.transpose();

  ASSERT_EQ(mat.row_size(), 3);
  ASSERT_EQ(mat.col_size(), 2);
  EXPECT_EQ(mat(0, 0), 1);
  EXPECT_EQ(mat(0, 1), 4);
  EXPECT_EQ(mat(1, 0), 2);
  EXPECT_EQ(mat(2, 1), 6);
}

TEST(MatrixTest, Transpose1x1Matrix) {
  txeo::Matrix<float> mat(1, 1, {3.14f});
  mat.transpose();

  ASSERT_EQ(mat.row_size(), 1);
  ASSERT_EQ(mat.col_size(), 1);
  EXPECT_FLOAT_EQ(mat(0, 0), 3.14f);
}

TEST(MatrixTest, TransposeRowVectorToColumn) {
  txeo::Matrix<double> mat(1, 3, {1.1, 2.2, 3.3});
  mat.transpose();

  ASSERT_EQ(mat.row_size(), 3);
  ASSERT_EQ(mat.col_size(), 1);
  EXPECT_DOUBLE_EQ(mat(0, 0), 1.1);
  EXPECT_DOUBLE_EQ(mat(1, 0), 2.2);
  EXPECT_DOUBLE_EQ(mat(2, 0), 3.3);
}

TEST(MatrixTest, MatrixMatrixMultiplication) {
  txeo::Matrix<int> a(2, 3, {1, 2, 3, 4, 5, 6});
  txeo::Matrix<int> b(3, 2, {7, 8, 9, 10, 11, 12});
  txeo::Matrix<int> result = a.dot(b);

  ASSERT_EQ(result.row_size(), 2);
  ASSERT_EQ(result.col_size(), 2);
  EXPECT_EQ(result(0, 0), 1 * 7 + 2 * 9 + 3 * 11);
  EXPECT_EQ(result(0, 1), 1 * 8 + 2 * 10 + 3 * 12);
  EXPECT_EQ(result(1, 0), 4 * 7 + 5 * 9 + 6 * 11);
  EXPECT_EQ(result(1, 1), 4 * 8 + 5 * 10 + 6 * 12);
}

TEST(MatrixTest, MatrixVectorMultiplication) {
  txeo::Matrix<int> mat(2, 3, {1, 2, 3, 4, 5, 6});
  txeo::Vector<int> vec({7, 8, 9});
  txeo::Tensor<int> result = mat.dot(vec);

  ASSERT_EQ(result.shape().axis_dim(0), 2);
  EXPECT_EQ(result(0, 0), 1 * 7 + 2 * 8 + 3 * 9);
  EXPECT_EQ(result(1, 0), 4 * 7 + 5 * 8 + 6 * 9);
}

TEST(MatrixTest, InvalidDimensionsThrow) {
  txeo::Matrix<int> a(2, 3);
  txeo::Matrix<int> b(2, 3);
  txeo::Vector<int> vec(2);

  EXPECT_THROW(a.dot(b), txeo::TensorOpError);

  EXPECT_THROW(a.dot(vec), txeo::TensorOpError);
}

TEST(MatrixTest, SquareMatrixMultiplication) {
  txeo::Matrix<int> a(2, 2, {1, 2, 3, 4});
  txeo::Matrix<int> b(2, 2, {5, 6, 7, 8});
  txeo::Matrix<int> result = a.dot(b);

  ASSERT_EQ(result.row_size(), 2);
  ASSERT_EQ(result.col_size(), 2);
  EXPECT_EQ(result(0, 0), 1 * 5 + 2 * 7);
  EXPECT_EQ(result(0, 1), 1 * 6 + 2 * 8);
  EXPECT_EQ(result(1, 0), 3 * 5 + 4 * 7);
  EXPECT_EQ(result(1, 1), 3 * 6 + 4 * 8);
}

TEST(MatrixTest, IdentityMatrixMultiplication) {
  txeo::Matrix<int> identity(2, 2, {1, 0, 0, 1});
  txeo::Matrix<int> mat(2, 3, {5, 6, 7, 8, 9, 10});
  txeo::Matrix<int> result = identity.dot(mat);

  ASSERT_EQ(result.row_size(), 2);
  ASSERT_EQ(result.col_size(), 3);
  EXPECT_EQ(result(0, 0), 5);
  EXPECT_EQ(result(1, 2), 10);
}
