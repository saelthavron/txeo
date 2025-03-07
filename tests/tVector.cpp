#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include "txeo/Vector.h"
#include <gtest/gtest.h>

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