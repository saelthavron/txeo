#include "txeo/Tensor.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorShape.h"
#include <gtest/gtest.h>

namespace txeo {

TEST(TensorFuncTest, PowerElemOperation) {
  Tensor<float> t1({2, 2}, {2.0f, 3.0f, 4.0f, 5.0f});

  auto result = TensorFunc<float>::power_elem(t1, 2.0f);

  ASSERT_EQ(result.shape(), txeo::TensorShape({2, 2}));
  EXPECT_FLOAT_EQ(result(0, 0), 4.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 25.0f);
}

TEST(TensorFuncTest, PowerElemByOperation) {
  Tensor<double> t1({3}, {2.0, 3.0, 4.0});

  TensorFunc<double>::power_elem_by(t1, 3.0);

  EXPECT_DOUBLE_EQ(t1(0), 8.0);
  EXPECT_DOUBLE_EQ(t1(1), 27.0);
  EXPECT_DOUBLE_EQ(t1(2), 64.0);
}

TEST(TensorFuncTest, PowerElemSpecialCases) {
  Tensor<float> t1({2}, {4.0f, 9.0f});

  auto result1 = TensorFunc<float>::power_elem(t1, 0.0f);
  EXPECT_FLOAT_EQ(result1(0), 1.0f);
  EXPECT_FLOAT_EQ(result1(1), 1.0f);

  auto result2 = TensorFunc<float>::power_elem(t1, 0.5f);
  EXPECT_FLOAT_EQ(result2(0), 2.0f);
  EXPECT_FLOAT_EQ(result2(1), 3.0f);
}

TEST(TensorFuncTest, NegativeExponent) {
  Tensor<double> t1({2}, {2.0, 3.0});

  auto result = TensorFunc<double>::power_elem(t1, -1.0);
  EXPECT_DOUBLE_EQ(result(0), 0.5);
  EXPECT_DOUBLE_EQ(result(1), 1.0 / 3.0);
}

TEST(TensorFuncTest, PowerElem) {
  txeo::Tensor<int> tensor({2}, {2, 3});
  EXPECT_NO_THROW(txeo::TensorFunc<int>::power_elem(tensor, 2));
}

TEST(TensorFuncTest, Abs) {
  Tensor<int> tensor({4}, {-1, 2, -3, 4});
  auto result = TensorFunc<int>::abs(tensor);
  EXPECT_EQ(result.shape(), TensorShape({4}));
  EXPECT_EQ(result(0), 1);
  EXPECT_EQ(result(1), 2);
  EXPECT_EQ(result(2), 3);
  EXPECT_EQ(result(3), 4);
}

TEST(TensorFuncTest, Square) {
  txeo::Tensor<int> tensor1D({3}, {1, 2, 3});
  auto result1D = TensorFunc<int>::square(tensor1D);
  EXPECT_EQ(result1D.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(result1D(0), 1);
  EXPECT_EQ(result1D(1), 4);
  EXPECT_EQ(result1D(2), 9);

  txeo::Tensor<double> tensor2D({2, 2}, {1.0, 2.0, 3.0, 4.0});
  auto result2D = TensorFunc<double>::square(tensor2D);
  EXPECT_EQ(result2D.shape(), txeo::TensorShape({2, 2}));
  EXPECT_DOUBLE_EQ(result2D(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(result2D(0, 1), 4.0);
  EXPECT_DOUBLE_EQ(result2D(1, 0), 9.0);
  EXPECT_DOUBLE_EQ(result2D(1, 1), 16.0);
}

TEST(TensorFuncTest, SquareBy) {
  txeo::Tensor<int> tensor1D({3}, {1, 2, 3});
  TensorFunc<int>::square_by(tensor1D);
  EXPECT_EQ(tensor1D(0), 1);
  EXPECT_EQ(tensor1D(1), 4);
  EXPECT_EQ(tensor1D(2), 9);

  txeo::Tensor<double> tensor2D({2, 2}, {1.0, 2.0, 3.0, 4.0});
  TensorFunc<double>::square_by(tensor2D);
  EXPECT_DOUBLE_EQ(tensor2D(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(tensor2D(0, 1), 4.0);
  EXPECT_DOUBLE_EQ(tensor2D(1, 0), 9.0);
  EXPECT_DOUBLE_EQ(tensor2D(1, 1), 16.0);
}

TEST(TensorFuncTest, Sqrt) {
  txeo::Tensor<double> tensor1D({3}, {1.0, 4.0, 9.0});
  auto result1D = TensorFunc<double>::sqrt(tensor1D);
  EXPECT_EQ(result1D.shape(), txeo::TensorShape({3}));
  EXPECT_DOUBLE_EQ(result1D(0), 1.0);
  EXPECT_DOUBLE_EQ(result1D(1), 2.0);
  EXPECT_DOUBLE_EQ(result1D(2), 3.0);

  txeo::Tensor<double> tensor2D({2, 2}, {16.0, 25.0, 36.0, 49.0});
  auto result2D = TensorFunc<double>::sqrt(tensor2D);
  EXPECT_EQ(result2D.shape(), txeo::TensorShape({2, 2}));
  EXPECT_DOUBLE_EQ(result2D(0, 0), 4.0);
  EXPECT_DOUBLE_EQ(result2D(0, 1), 5.0);
  EXPECT_DOUBLE_EQ(result2D(1, 0), 6.0);
  EXPECT_DOUBLE_EQ(result2D(1, 1), 7.0);
}

TEST(TensorFuncTest, SqrtBy) {
  txeo::Tensor<double> tensor1D({3}, {1.0, 4.0, 9.0});
  TensorFunc<double>::sqrt_by(tensor1D);
  EXPECT_DOUBLE_EQ(tensor1D(0), 1.0);
  EXPECT_DOUBLE_EQ(tensor1D(1), 2.0);
  EXPECT_DOUBLE_EQ(tensor1D(2), 3.0);

  txeo::Tensor<double> tensor2D({2, 2}, {16.0, 25.0, 36.0, 49.0});
  TensorFunc<double>::sqrt_by(tensor2D);
  EXPECT_DOUBLE_EQ(tensor2D(0, 0), 4.0);
  EXPECT_DOUBLE_EQ(tensor2D(0, 1), 5.0);
  EXPECT_DOUBLE_EQ(tensor2D(1, 0), 6.0);
  EXPECT_DOUBLE_EQ(tensor2D(1, 1), 7.0);
}

TEST(TensorFuncTest, AbsBy) {
  txeo::Tensor<int> tensor1D({3}, {-1, 2, -3});
  TensorFunc<int>::abs_by(tensor1D);
  EXPECT_EQ(tensor1D(0), 1);
  EXPECT_EQ(tensor1D(1), 2);
  EXPECT_EQ(tensor1D(2), 3);

  txeo::Tensor<double> tensor2D({2, 2}, {-1.0, 2.0, -3.0, 4.0});
  TensorFunc<double>::abs_by(tensor2D);
  EXPECT_DOUBLE_EQ(tensor2D(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(tensor2D(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(tensor2D(1, 0), 3.0);
  EXPECT_DOUBLE_EQ(tensor2D(1, 1), 4.0);
}

TEST(TensorFuncTest, EmptyTensorHandling) {
  Tensor<float> empty({0});
  Tensor<float> t1({2}, {1.0f, 2.0f});

  EXPECT_THROW(TensorFunc<float>::power_elem(empty, 2.0f), TensorFuncError);
}

TEST(TensorFuncTest, PermuteValidAxes) {

  txeo::Tensor<int> tensor({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  auto result = TensorFunc<int>::permute(tensor, {1, 2, 0});

  EXPECT_EQ(result.shape(), txeo::TensorShape({3, 4, 2}));

  EXPECT_EQ(result(0, 0, 0), 1);
  EXPECT_EQ(result(2, 3, 1), 24);
  EXPECT_EQ(result(1, 2, 1), 19);
  EXPECT_EQ(result(1, 0, 0), 5);
}

TEST(TensorFuncTest, PermuteInvalidAxes) {

  txeo::Tensor<int> tensor({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  EXPECT_THROW(TensorFunc<int>::permute(tensor, {1, 2}), txeo::TensorFuncError);

  EXPECT_THROW(TensorFunc<int>::permute(tensor, {1, 2, 3}), txeo::TensorFuncError);
}

TEST(TensorFuncTest, PermuteByValidAxes) {

  txeo::Tensor<int> tensor({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  TensorFunc<int>::permute_by(tensor, {1, 2, 0});

  EXPECT_EQ(tensor.shape(), txeo::TensorShape({3, 4, 2}));

  EXPECT_EQ(tensor(0, 0, 0), 1);
  EXPECT_EQ(tensor(2, 3, 1), 24);
  EXPECT_EQ(tensor(1, 2, 1), 19);
  EXPECT_EQ(tensor(1, 0, 0), 5);
}

TEST(TensorFuncTest, PermuteByInvalidAxes) {

  txeo::Tensor<int> tensor({2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  EXPECT_THROW(TensorFunc<int>::permute_by(tensor, {1, 2}), txeo::TensorFuncError);

  EXPECT_THROW(TensorFunc<int>::permute_by(tensor, {1, 2, 3}), txeo::TensorFuncError);
}

TEST(TensorFuncTest, Transpose) {

  txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});

  auto result = TensorFunc<int>::transpose(matrix);

  EXPECT_EQ(result.shape(), txeo::TensorShape({3, 2}));

  EXPECT_EQ(result(0, 0), 1);
  EXPECT_EQ(result(2, 1), 6);
  EXPECT_EQ(result(1, 0), 2);
}

TEST(TensorFuncTest, TransposeBy) {

  txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});

  TensorFunc<int>::transpose_by(matrix);

  EXPECT_EQ(matrix.shape(), txeo::TensorShape({3, 2}));

  EXPECT_EQ(matrix(0, 0), 1);
  EXPECT_EQ(matrix(2, 1), 6);
  EXPECT_EQ(matrix(1, 0), 2);
}

TEST(TensorFuncTest, NormalizationAll) {

  txeo::Tensor<double> tens1({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  txeo::TensorFunc<double>::normalize_by(tens1, txeo::NormalizationType::MIN_MAX);
  txeo::Tensor<double> resp1({3, 3}, {0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1});
  EXPECT_TRUE(tens1 == resp1);

  txeo::Tensor<double> tens2({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  txeo::TensorFunc<double>::normalize_by(tens2, txeo::NormalizationType::Z_SCORE);
  EXPECT_NEAR(tens2(0, 0), -1.46059, 1e-5);
  EXPECT_NEAR(tens2(1, 1), 0.00000, 1e-5);
  EXPECT_NEAR(tens2(2, 1), 1.09544, 1e-5);

  txeo::Tensor<double> tens3(txeo::TensorShape({0}));
  EXPECT_THROW(txeo::TensorFunc<double>::normalize_by(tens3, txeo::NormalizationType::Z_SCORE),
               txeo::TensorFuncError);

  txeo::Tensor<double> tens4({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  auto resp4 = txeo::TensorFunc<double>::normalize(tens4, txeo::NormalizationType::Z_SCORE);
  EXPECT_TRUE(resp4 == tens2);

  txeo::Tensor<double> tens5({1}, {3});
  txeo::TensorFunc<double>::normalize_by(tens5, txeo::NormalizationType::Z_SCORE);
  txeo::Tensor<double> resp5({1}, {3});
  EXPECT_TRUE(resp5 == tens5);
}

TEST(TensorFuncTest, NormalizationAxis) {

  txeo::Tensor<double> tens6({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  txeo::TensorFunc<double>::normalize_by(tens6, 0, txeo::NormalizationType::MIN_MAX);
  txeo::Tensor<double> resp6({3, 3}, {0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1});
  EXPECT_TRUE(tens6 == resp6);

  txeo::Tensor<double> tens7({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  auto tens7_norm = txeo::TensorFunc<double>::normalize(tens7, 0, txeo::NormalizationType::Z_SCORE);
  txeo::Tensor<double> resp7({3, 3}, {-1, -1, -1, 0, 0, 0, 1, 1, 1});
  EXPECT_TRUE(tens7_norm == resp7);
}
} // namespace txeo