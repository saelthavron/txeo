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

  auto result1 = TensorOp<float>::power_elem(t1, 0.0f);
  EXPECT_FLOAT_EQ(result1(0), 1.0f);
  EXPECT_FLOAT_EQ(result1(1), 1.0f);

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

TEST(TensorOpTest, SumTensorScalar) {
  Tensor<int> input({2, 2}, {1, 2, 3, 4});
  Tensor<int> result = TensorOp<int>::sum(input, 5);
  ASSERT_EQ(result.shape(), txeo::TensorShape({2, 2}));
  EXPECT_EQ(result.data()[0], 6);
  EXPECT_EQ(result.data()[1], 7);
  EXPECT_EQ(result.data()[2], 8);
  EXPECT_EQ(result.data()[3], 9);
}

TEST(TensorOpTest, SumByTensorScalar) {
  Tensor<float> input({3}, {1.1f, 2.2f, 3.3f});
  TensorOp<float>::sum_by(input, 10.0f);
  ASSERT_FLOAT_EQ(input.data()[0], 11.1f);
  ASSERT_FLOAT_EQ(input.data()[1], 12.2f);
  ASSERT_FLOAT_EQ(input.data()[2], 13.3f);
}

TEST(TensorOpTest, SubtractTensorScalar) {
  Tensor<double> input({2}, {10.5, 20.5});
  Tensor<double> result = TensorOp<double>::subtract(input, 5.0);
  ASSERT_EQ(result.shape(), txeo::TensorShape({2}));
  EXPECT_DOUBLE_EQ(result.data()[0], 5.5);
  EXPECT_DOUBLE_EQ(result.data()[1], 15.5);
}

TEST(TensorOpTest, SubtractScalarTensor) {
  Tensor<int> input({3}, {3, 5, 7});
  Tensor<int> result = TensorOp<int>::subtract(20, input);
  ASSERT_EQ(result.data()[0], 17);
  ASSERT_EQ(result.data()[1], 15);
  ASSERT_EQ(result.data()[2], 13);
}

TEST(TensorOpTest, SubtractByScalarTensor) {
  Tensor<float> input({2}, {100.0f, 200.0f});
  TensorOp<float>::subtract_by(50.0f, input);
  ASSERT_FLOAT_EQ(input.data()[0], -50.0f);
  ASSERT_FLOAT_EQ(input.data()[1], -150.0f);
}

TEST(TensorOpTest, DivideTensorByScalar) {
  Tensor<double> input({3}, {10.0, 20.0, 30.0});
  Tensor<double> result = TensorOp<double>::divide(input, 2.0);
  ASSERT_DOUBLE_EQ(result.data()[0], 5.0);
  ASSERT_DOUBLE_EQ(result.data()[1], 10.0);
  ASSERT_DOUBLE_EQ(result.data()[2], 15.0);
}

TEST(TensorOpTest, DivideScalarByTensor) {
  Tensor<int> input({2}, {2, 5});
  Tensor<int> result = TensorOp<int>::divide(10, input);
  ASSERT_EQ(result.data()[0], 5);
  ASSERT_EQ(result.data()[1], 2);
}

TEST(TensorOpTest, DivideByTensorScalar) {
  Tensor<float> input({2}, {9.0f, 16.0f});
  TensorOp<float>::divide_by(input, 3.0f);
  ASSERT_FLOAT_EQ(input.data()[0], 3.0f);
  ASSERT_FLOAT_EQ(input.data()[1], 5.3333333f);
}

TEST(TensorOpTest, DivideByScalarTensor) {
  Tensor<double> input({2}, {2.0, 4.0});
  TensorOp<double>::divide_by(8.0, input);
  ASSERT_DOUBLE_EQ(input.data()[0], 4.0);
  ASSERT_DOUBLE_EQ(input.data()[1], 2.0);
}

TEST(TensorOpTest, HadamardDiv) {
  Tensor<int> left({3}, {10, 20, 30});
  Tensor<int> right({3}, {2, 5, 6});
  Tensor<int> result = TensorOp<int>::hadamard_div(left, right);
  ASSERT_EQ(result.data()[0], 5);
  ASSERT_EQ(result.data()[1], 4);
  ASSERT_EQ(result.data()[2], 5);
}

TEST(TensorOpTest, HadamardDivBy) {
  Tensor<float> left({2}, {15.0f, 25.0f});
  Tensor<float> right({2}, {3.0f, 5.0f});
  TensorOp<float>::hadamard_div_by(left, right);
  ASSERT_FLOAT_EQ(left.data()[0], 5.0f);
  ASSERT_FLOAT_EQ(left.data()[1], 5.0f);
}

TEST(TensorOpTest, HandleEmptyTensor) {
  Tensor<int> empty_tensor({0}, {});
  EXPECT_THROW(TensorOp<int>::sum(empty_tensor, 5), TensorOpError);
}

TEST(TensorOpTest, MixedPrecisionOperations) {
  Tensor<double> input({2}, {1.5, 2.5});
  Tensor<double> result = TensorOp<double>::divide(input, 2);
  ASSERT_DOUBLE_EQ(result.data()[0], 0.75);
  ASSERT_DOUBLE_EQ(result.data()[1], 1.25);
}

TEST(TensorOpTest, HadamardDiv2) {
  txeo::Tensor<double> a({2}, {1.0, 2.0});
  txeo::Tensor<double> b({2}, {2.0, 1.0});
  EXPECT_NO_THROW(txeo::TensorOp<double>::hadamard_div(a, b));
}

TEST(TensorOpTest, SubtractByScalar) {
  txeo::Tensor<double> tensor({2}, {3.0, 4.0});
  EXPECT_NO_THROW(txeo::TensorOp<double>::subtract_by(2.0, tensor));
}

TEST(TensorOpTest, HadamardProd) {
  txeo::Tensor<int> a({2}, {2, 3});
  txeo::Tensor<int> b({2}, {1, 2});
  EXPECT_NO_THROW(txeo::TensorOp<int>::hadamard_prod(a, b));
}

TEST(TensorOpTest, PowerElem) {
  txeo::Tensor<int> tensor({2}, {2, 3});
  EXPECT_NO_THROW(txeo::TensorOp<int>::power_elem(tensor, 2));
}

TEST(TensorOpTest, Multiply) {
  txeo::Tensor<long long> tensor({2}, {5LL, 6LL});
  EXPECT_NO_THROW(txeo::TensorOp<long long>::multiply(tensor, 2LL));
}

TEST(TensorOpTest, SubtractByTensor) {
  txeo::Tensor<long long> a({2}, {5LL, 6LL});
  txeo::Tensor<long long> b({2}, {1LL, 2LL});
  EXPECT_NO_THROW(txeo::TensorOp<long long>::subtract_by(a, b));
}

TEST(TensorOpTest, DivideByZeroDimTensor) {
  txeo::Tensor<double> empty_tensor({0});
  EXPECT_THROW(txeo::TensorOp<double>::divide(empty_tensor, 2.0), txeo::TensorOpError);
}

TEST(TensorOpTest, SubtractInvalidShapes) {
  txeo::Tensor<int> a({2}, {1, 2});
  txeo::Tensor<int> b({3}, {1, 2, 3});
  EXPECT_THROW(txeo::TensorOp<int>::subtract(a, b), txeo::TensorOpError);
}

TEST(TensorOpTest, Abs) {
  Tensor<int> tensor({4}, {-1, 2, -3, 4});
  auto result = TensorOp<int>::abs(tensor);
  EXPECT_EQ(result.shape(), TensorShape({4}));
  EXPECT_EQ(result(0), 1);
  EXPECT_EQ(result(1), 2);
  EXPECT_EQ(result(2), 3);
  EXPECT_EQ(result(3), 4);
}

} // namespace txeo