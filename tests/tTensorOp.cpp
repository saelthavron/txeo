#include "txeo/Tensor.h"
#include "txeo/TensorOp.h"
#include "txeo/TensorShape.h"
#include <cstdint>
#include <gtest/gtest.h>

namespace txeo {

TEST(TensorOpTest, SumOperationFloat) {
  Tensor<float> t1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> t2({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

  auto result = TensorOp<float>::sum(t1, t2);
  EXPECT_EQ(result.shape().axes_dims(), std::vector<int64_t>({2, 2}));
  EXPECT_FLOAT_EQ(result(0, 0), 6.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 12.0f);
}

TEST(TensorOpTest, SumOperatorFloat) {
  Tensor<float> t1({2}, {1.0f, 2.0f});
  Tensor<float> t2({2}, {3.0f, 4.0f});

  auto result = t1 + t2;
  EXPECT_EQ(result.shape().axes_dims(), std::vector<int64_t>({2}));
  EXPECT_FLOAT_EQ(result(0), 4.0f);
  EXPECT_FLOAT_EQ(result(1), 6.0f);
}

TEST(TensorOpTest, SumByOperationFloat) {
  Tensor<float> t1({3}, {1.0f, 2.0f, 3.0f});
  Tensor<float> t2({3}, {4.0f, 5.0f, 6.0f});

  TensorOp<float>::sum_by(t1, t2);
  EXPECT_FLOAT_EQ(t1(0), 5.0f);
  EXPECT_FLOAT_EQ(t1(1), 7.0f);
  EXPECT_FLOAT_EQ(t1(2), 9.0f);
}

TEST(TensorOpTest, SubtractOperationDouble) {
  Tensor<double> t1({2, 2}, {5.0, 6.0, 7.0, 8.0});
  Tensor<double> t2({2, 2}, {1.0, 2.0, 3.0, 4.0});

  auto result = TensorOp<double>::subtract(t1, t2);
  EXPECT_EQ(result.shape().axes_dims(), std::vector<int64_t>({2, 2}));
  EXPECT_DOUBLE_EQ(result(0, 0), 4.0);
  EXPECT_DOUBLE_EQ(result(1, 1), 4.0);
}

TEST(TensorOpTest, SubtractOperatorDouble) {
  Tensor<double> t1({2}, {5.0, 6.0});
  Tensor<double> t2({2}, {2.0, 3.0});

  auto result = t1 - t2;
  EXPECT_DOUBLE_EQ(result(0), 3.0);
  EXPECT_DOUBLE_EQ(result(1), 3.0);
}

TEST(TensorOpTest, SubtractByOperationDouble) {
  Tensor<double> t1({3}, {10.0, 20.0, 30.0});
  Tensor<double> t2({3}, {1.0, 2.0, 3.0});

  TensorOp<double>::subtract_by(t1, t2);
  EXPECT_DOUBLE_EQ(t1(0), 9.0);
  EXPECT_DOUBLE_EQ(t1(1), 18.0);
  EXPECT_DOUBLE_EQ(t1(2), 27.0);
}

TEST(TensorOpTest, MultiplyScalarFloat) {
  Tensor<float> t1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  auto result = TensorOp<float>::multiply(t1, 2.5f);
  EXPECT_FLOAT_EQ(result(0, 0), 2.5f);
  EXPECT_FLOAT_EQ(result(1, 1), 10.0f);
}

TEST(TensorOpTest, MultiplyOperatorFloat) {
  Tensor<float> t1({2}, {1.5f, 2.5f});

  auto result = t1 * 2.0f;
  EXPECT_FLOAT_EQ(result(0), 3.0f);
  EXPECT_FLOAT_EQ(result(1), 5.0f);
}

TEST(TensorOpTest, MultiplyByScalarDouble) {
  Tensor<double> t1({2, 2}, {1.0, 2.0, 3.0, 4.0});

  TensorOp<double>::multiply_by(t1, 3.0);
  EXPECT_DOUBLE_EQ(t1(0, 0), 3.0);
  EXPECT_DOUBLE_EQ(t1(1, 1), 12.0);
}

TEST(TensorOpTest, ShapeMismatchSum) {
  Tensor<float> t1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> t2({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  EXPECT_THROW(TensorOp<float>::sum(t1, t2), TensorOpError);
  EXPECT_THROW(t1 + t2, TensorOpError);
}

TEST(TensorOpTest, EmptyTensorOperations) {
  Tensor<float> empty({0});
  Tensor<float> t1({2}, {1.0f, 2.0f});

  EXPECT_THROW(TensorOp<float>::sum(empty, t1), TensorOpError);
  EXPECT_THROW(TensorOp<float>::sum_by(empty, t1), TensorOpError);
  EXPECT_THROW(TensorOp<float>::multiply(empty, 2.0f), TensorOpError);
}

TEST(TensorOpTest, InplaceModificationCheck) {
  Tensor<float> original({2}, {1.0f, 2.0f});
  Tensor<float> copy = original;

  TensorOp<float>::sum_by(copy, original);
  EXPECT_FLOAT_EQ(copy(0), 2.0f);
  EXPECT_FLOAT_EQ(copy(1), 4.0f);
}

TEST(TensorOpTest, HadamardProdCorrectness) {
  Tensor<float> t1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> t2({2, 2}, {2.0f, 3.0f, 4.0f, 5.0f});

  auto result = TensorOp<float>::hadamard_prod(t1, t2);

  ASSERT_EQ(result.shape(), txeo::TensorShape({2, 2}));
  EXPECT_FLOAT_EQ(result(0, 0), 2.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 20.0f);
}

TEST(TensorOpTest, HadamardProdByInPlace) {
  Tensor<double> t1({3}, {1.0, 2.0, 3.0});
  Tensor<double> t2({3}, {4.0, 5.0, 6.0});

  TensorOp<double>::hadamard_prod_by(t1, t2);

  EXPECT_DOUBLE_EQ(t1(0), 4.0);
  EXPECT_DOUBLE_EQ(t1(1), 10.0);
  EXPECT_DOUBLE_EQ(t1(2), 18.0);
}

TEST(TensorOpTest, PowerElemOperation) {
  Tensor<float> t1({2, 2}, {2.0f, 3.0f, 4.0f, 5.0f});

  auto result = TensorOp<float>::power_elem(t1, 2.0f);

  ASSERT_EQ(result.shape(), txeo::TensorShape({2, 2}));
  EXPECT_FLOAT_EQ(result(0, 0), 4.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 25.0f);
}

TEST(TensorOpTest, PowerElemByOperation) {
  Tensor<double> t1({3}, {2.0, 3.0, 4.0});

  TensorOp<double>::power_elem_by(t1, 3.0);

  EXPECT_DOUBLE_EQ(t1(0), 8.0);
  EXPECT_DOUBLE_EQ(t1(1), 27.0);
  EXPECT_DOUBLE_EQ(t1(2), 64.0);
}

TEST(TensorOpTest, ShapeMismatchHadamard) {
  Tensor<float> t1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> t2({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  EXPECT_THROW(TensorOp<float>::hadamard_prod(t1, t2), TensorOpError);
  EXPECT_THROW(TensorOp<float>::hadamard_prod_by(t1, t2), TensorOpError);
}

TEST(TensorOpTest, PowerElemSpecialCases) {
  Tensor<float> t1({2}, {4.0f, 9.0f});

  // Test zero exponent
  auto result1 = TensorOp<float>::power_elem(t1, 0.0f);
  EXPECT_FLOAT_EQ(result1(0), 1.0f);
  EXPECT_FLOAT_EQ(result1(1), 1.0f);

  // Test fractional exponent
  auto result2 = TensorOp<float>::power_elem(t1, 0.5f);
  EXPECT_FLOAT_EQ(result2(0), 2.0f);
  EXPECT_FLOAT_EQ(result2(1), 3.0f);
}

TEST(TensorOpTest, EmptyTensorHandling) {
  Tensor<float> empty({0});
  Tensor<float> t1({2}, {1.0f, 2.0f});

  EXPECT_THROW(TensorOp<float>::hadamard_prod(empty, t1), TensorOpError);
  EXPECT_THROW(TensorOp<float>::power_elem(empty, 2.0f), TensorOpError);
}

TEST(TensorOpTest, NegativeExponent) {
  Tensor<double> t1({2}, {2.0, 3.0});

  auto result = TensorOp<double>::power_elem(t1, -1.0);
  EXPECT_DOUBLE_EQ(result(0), 0.5);
  EXPECT_DOUBLE_EQ(result(1), 1.0 / 3.0);
}

TEST(TensorOpTest, InPlaceModificationCheck) {
  Tensor<float> original({2}, {3.0f, 4.0f});
  Tensor<float> copy = original;
  Tensor<float> operand({2}, {2.0f, 0.5f});

  TensorOp<float>::hadamard_prod_by(copy, operand);
  EXPECT_FLOAT_EQ(copy(0), 6.0f);
  EXPECT_FLOAT_EQ(copy(1), 2.0f);
}

} // namespace txeo