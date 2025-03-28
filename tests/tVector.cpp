#include <gtest/gtest.h>
#include <initializer_list>
#include <utility>
#include <vector>

#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include "txeo/Vector.h"
#include "txeo/types.h"

TEST(VectorTest, ParameterizedConstructor) {
  txeo::Vector<int> vector(3);
  EXPECT_EQ(vector.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(vector.dim(), 3);
}

TEST(VectorTest, ParameterizedConstructorWithFillValue) {
  txeo::Vector<int> vector(3, 5);
  EXPECT_EQ(vector.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(vector(0), 5);
  EXPECT_EQ(vector(1), 5);
  EXPECT_EQ(vector(2), 5);
}

TEST(VectorTest, ParameterizedConstructorWithInitializerList) {
  txeo::Vector<int> vector(3, {1, 2, 3});
  EXPECT_EQ(vector.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(vector(0), 1);
  EXPECT_EQ(vector(1), 2);
  EXPECT_EQ(vector(2), 3);
}

TEST(VectorTest, ConstructorWithInitializerList) {
  txeo::Vector<int> vector({1, 2, 3});
  EXPECT_EQ(vector.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(vector(0), 1);
  EXPECT_EQ(vector(1), 2);
  EXPECT_EQ(vector(2), 3);
}

TEST(VectorTest, MoveConstructorFromTensor) {
  txeo::Tensor<int> tensor({3}, {1, 2, 3});
  txeo::Vector<int> vector(std::move(tensor));
  EXPECT_EQ(vector.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(vector(0), 1);
  EXPECT_EQ(vector(1), 2);
  EXPECT_EQ(vector(2), 3);
}

TEST(VectorTest, CopyConstructor) {
  txeo::Vector<int> vector1(3, {1, 2, 3});
  txeo::Vector<int> vector2(vector1);
  EXPECT_EQ(vector2.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(vector2(0), 1);
  EXPECT_EQ(vector2(1), 2);
  EXPECT_EQ(vector2(2), 3);
}

TEST(VectorTest, MoveConstructor) {
  txeo::Vector<int> vector1(3, {1, 2, 3});
  txeo::Vector<int> vector2(std::move(vector1));
  EXPECT_EQ(vector2.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(vector2(0), 1);
  EXPECT_EQ(vector2(1), 2);
  EXPECT_EQ(vector2(2), 3);
}

TEST(VectorTest, CopyAssignmentOperator) {
  txeo::Vector<int> vector1(3, {1, 2, 3});
  txeo::Vector<int> vector2(1);
  vector2 = vector1;
  EXPECT_EQ(vector2.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(vector2(0), 1);
  EXPECT_EQ(vector2(1), 2);
  EXPECT_EQ(vector2(2), 3);
}

TEST(VectorTest, MoveAssignmentOperator) {
  txeo::Vector<int> vector1(3, {1, 2, 3});
  txeo::Vector<int> vector2(1);
  vector2 = std::move(vector1);
  EXPECT_EQ(vector2.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(vector2(0), 1);
  EXPECT_EQ(vector2(1), 2);
  EXPECT_EQ(vector2(2), 3);
}

TEST(VectorTest, VectorError) {
  txeo::Tensor<int> emptyTensor({1, 2});
  EXPECT_THROW(txeo::Vector<int> vector(std::move(emptyTensor)), txeo::VectorError);
}

TEST(VectorTest, ReshapeValidShape) {

  txeo::Vector<int> vector(6, {1, 2, 3, 4, 5, 6});

  EXPECT_THROW(vector.reshape({2, 3}), txeo::VectorError);
}

TEST(VectorTest, ReshapeInvalidShape) {

  txeo::Vector<int> vector(6, {1, 2, 3, 4, 5, 6});

  EXPECT_THROW(vector.reshape({2, 4}), txeo::VectorError);
}

TEST(VectorTest, ToVectorValid1DTensor) {

  txeo::Tensor<int> tensor({6}, {1, 2, 3, 4, 5, 6});
  txeo::Tensor<int> tensor2({6}, {1, 2, 3, 4, 5, 6});

  auto result = txeo::Vector<int>::to_vector(std::move(tensor));
  auto result2 = txeo::Vector<int>::to_vector(tensor2);

  EXPECT_EQ(result.shape(), txeo::TensorShape({6}));
  EXPECT_EQ(result2.shape(), txeo::TensorShape({6}));

  EXPECT_EQ(result(0), 1);
  EXPECT_EQ(result(5), 6);
  EXPECT_EQ(result(2), 3);
  EXPECT_EQ(result2(0), 1);
  EXPECT_EQ(result2(5), 6);
  EXPECT_EQ(result2(2), 3);
}

TEST(VectorTest, ToVectorInvalid2DTensor) {

  txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});

  EXPECT_THROW(txeo::Vector<int>::to_vector(std::move(tensor)), txeo::VectorError);
}

TEST(VectorTest, ToVectorEmptyTensor) {

  txeo::Tensor<int> tensor(txeo::TensorShape({}));

  EXPECT_THROW(txeo::Vector<int>::to_vector(std::move(tensor)), txeo::VectorError);
}

TEST(VectorTest, ToVectorRvalue) {
  txeo::Tensor<int> tensor({4}, {1, 2, 3, 4});
  txeo::Vector<int> vector = txeo::Vector<int>::to_vector(std::move(tensor));

  ASSERT_EQ(vector.dim(), 4);
  EXPECT_EQ(vector(0), 1);
  EXPECT_EQ(vector(1), 2);
  EXPECT_EQ(vector(2), 3);
  EXPECT_EQ(vector(3), 4);
}

TEST(VectorTest, ToVectorConstRef) {
  const txeo::Tensor<int> tensor({4}, {5, 6, 7, 8});
  txeo::Vector<int> vector = txeo::Vector<int>::to_vector(tensor);

  ASSERT_EQ(vector.dim(), 4);
  EXPECT_EQ(vector(0), 5);
  EXPECT_EQ(vector(1), 6);
  EXPECT_EQ(vector(2), 7);
  EXPECT_EQ(vector(3), 8);
}

TEST(VectorTest, ToTensorRvalue) {
  txeo::Vector<int> vector(4, {9, 10, 11, 12});
  txeo::Tensor<int> tensor = txeo::Vector<int>::to_tensor(std::move(vector));

  ASSERT_EQ(tensor.dim(), 4);
  EXPECT_EQ(tensor(0), 9);
  EXPECT_EQ(tensor(1), 10);
  EXPECT_EQ(tensor(2), 11);
  EXPECT_EQ(tensor(3), 12);
}

TEST(VectorTest, ToTensorConstRef) {
  const txeo::Vector<int> vector(4, {13, 14, 15, 16});
  txeo::Tensor<int> tensor = txeo::Vector<int>::to_tensor(vector);

  ASSERT_EQ(tensor.dim(), 4);
  EXPECT_EQ(tensor(0), 13);
  EXPECT_EQ(tensor(1), 14);
  EXPECT_EQ(tensor(2), 15);
  EXPECT_EQ(tensor(3), 16);
}

TEST(VectorTest, Normalization) {
  txeo::Vector<double> vec({1., 2., 3., 4., 5., 6., 7., 8., 9.});
  txeo::Vector<double> resp({0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1});

  vec.normalize(txeo::NormalizationType::MIN_MAX);
  EXPECT_TRUE(vec == resp);
}

TEST(VectorTest, VectorAddition) {
  txeo::Vector<int> v1({1, 2, 3});
  txeo::Vector<int> v2({4, 5, 6});
  auto result = v1 + v2;

  EXPECT_EQ(result.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(result.data()[0], 5);
  EXPECT_EQ(result.data()[1], 7);
  EXPECT_EQ(result.data()[2], 9);
}

TEST(VectorTest, ScalarAddition) {
  txeo::Vector<double> v({1.5, 2.5, 3.5});
  auto result = v + 2.5;

  EXPECT_EQ(result.shape(), txeo::TensorShape({3}));
  EXPECT_DOUBLE_EQ(result.data()[0], 4.0);
  EXPECT_DOUBLE_EQ(result.data()[1], 5.0);
  EXPECT_DOUBLE_EQ(result.data()[2], 6.0);
}

TEST(VectorTest, VectorSubtraction) {
  txeo::Vector<float> v1({5.0f, 3.0f, 8.0f});
  txeo::Vector<float> v2({1.0f, 2.0f, 3.0f});
  auto result = v1 - v2;

  EXPECT_EQ(result.shape(), txeo::TensorShape({3}));
  EXPECT_FLOAT_EQ(result.data()[0], 4.0f);
  EXPECT_FLOAT_EQ(result.data()[1], 1.0f);
  EXPECT_FLOAT_EQ(result.data()[2], 5.0f);
}

TEST(VectorTest, ScalarSubtractionRight) {
  txeo::Vector<int> v({5, 3, 8});
  auto result = v - 2;

  EXPECT_EQ(result.data()[0], 3);
  EXPECT_EQ(result.data()[1], 1);
  EXPECT_EQ(result.data()[2], 6);
}

TEST(VectorTest, ScalarSubtractionLeft) {
  txeo::Vector<int> v({1, 2, 3});
  auto result = 10 - v;

  EXPECT_EQ(result.data()[0], 9);
  EXPECT_EQ(result.data()[1], 8);
  EXPECT_EQ(result.data()[2], 7);
}

TEST(VectorTest, ScalarMultiplication) {
  txeo::Vector<int> v({2, 3, 4});
  auto result = v * 3;

  EXPECT_EQ(result.data()[0], 6);
  EXPECT_EQ(result.data()[1], 9);
  EXPECT_EQ(result.data()[2], 12);
  EXPECT_TRUE((4 * v) == txeo::Vector<int>({8, 12, 16}));
}

TEST(VectorTest, ScalarDivisionRight) {
  txeo::Vector<double> v({10.0, 20.0, 30.0});
  auto result = v / 2.0;

  EXPECT_DOUBLE_EQ(result.data()[0], 5.0);
  EXPECT_DOUBLE_EQ(result.data()[1], 10.0);
  EXPECT_DOUBLE_EQ(result.data()[2], 15.0);
}

TEST(VectorTest, ScalarDivisionLeft) {
  txeo::Vector<int> v({2, 4, 5});
  auto result = 100 / v;

  EXPECT_EQ(result.data()[0], 50);
  EXPECT_EQ(result.data()[1], 25);
  EXPECT_EQ(result.data()[2], 20);
}

TEST(VectorTest, EmptyVectorOperations) {
  txeo::Vector<float> empty_vec;
  auto result_add = empty_vec + 5.0f;
  auto result_mul = empty_vec * 2.0f;

  EXPECT_EQ(result_add(0), 5.0f);
  EXPECT_EQ(result_mul(0), 0.0f);
}

TEST(VectorTest, MixedPrecisionOperations) {
  txeo::Vector<double> v({1.0, 2.0, 3.0});
  auto result = (v * 2.) - 1.5;

  EXPECT_DOUBLE_EQ(result.data()[0], 0.5);
  EXPECT_DOUBLE_EQ(result.data()[1], 2.5);
  EXPECT_DOUBLE_EQ(result.data()[2], 4.5);
}

TEST(VectorTest, FloatingPointPrecision) {
  txeo::Vector<double> v({1.0, 2.0, 3.0});
  auto result = v / 3.0;

  EXPECT_DOUBLE_EQ(result.data()[0], 1.0 / 3.0);
  EXPECT_DOUBLE_EQ(result.data()[1], 2.0 / 3.0);
  EXPECT_DOUBLE_EQ(result.data()[2], 1.0);
}

TEST(VectorTest, BooleanOperations) {
  txeo::Vector<bool> v1({true, false, true});
  txeo::Vector<bool> v2({true, true, false});
  auto result = v1 - v2;

  EXPECT_EQ(result.data()[0], false);
  EXPECT_EQ(result.data()[1], true);
  EXPECT_EQ(result.data()[2], true);
}
