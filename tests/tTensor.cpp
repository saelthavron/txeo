#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "txeo/Tensor.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorOp.h"
#include "txeo/TensorShape.h"

namespace txeo {

TEST(TensorTest, ShapeConstructor) {
  TensorShape shape({2, 3, 4});
  Tensor<int> t(shape);
  EXPECT_EQ(t.dim(), 24);
  EXPECT_EQ(t.order(), 3);
  EXPECT_EQ(t.shape().axes_dims(), std::vector<int64_t>({2, 3, 4}));

  Tensor<int> tt({2, 3, 4, 5});
  EXPECT_EQ(tt.dim(), 120);
  EXPECT_EQ(tt.order(), 4);
  EXPECT_EQ(tt.shape().axes_dims(), std::vector<int64_t>({2, 3, 4, 5}));

  Tensor<int> ttt({4, 5, 6});
  EXPECT_EQ(ttt.dim(), 120);
  EXPECT_EQ(ttt.order(), 3);
  EXPECT_EQ(ttt.shape().axes_dims(), std::vector<int64_t>({4, 5, 6}));

  auto shp = std::vector<size_t>({4, 5, 6});
  Tensor<int> tttt(shp);
  EXPECT_EQ(tttt.dim(), 120);
  EXPECT_EQ(tttt.order(), 3);
  EXPECT_EQ(tttt.shape().axes_dims(), std::vector<int64_t>({4, 5, 6}));

  auto shp1 = std::vector<size_t>({4, 5, 6});
  Tensor<int> tv(shp1, 5);
  for (size_t i{0}; i < tv.dim(); ++i)
    EXPECT_EQ(tv.data()[i], 5);

  txeo::TensorShape x({100});
  Tensor<int> source(x, 5);
  for (size_t i{0}; i < source.dim(); ++i)
    EXPECT_EQ(source(i), 5);
}

TEST(TensorTest, InitializerListConstructor) {
  Tensor<int> t({{1, 2}, {3, 4}, {5, 6}});
  ASSERT_EQ(t.order(), 2);
  EXPECT_EQ(t.shape().axes_dims(), std::vector<int64_t>({3, 2}));
  EXPECT_EQ(t(0, 0), 1);
  EXPECT_EQ(t(2, 1), 6);
}

TEST(TensorTest, CopyConstructor) {
  Tensor<int> original({{1, 2}, {3, 4}});
  Tensor<int> copy(original);
  EXPECT_EQ(copy.dim(), 4);
  EXPECT_EQ(copy(1, 1), 4);

  original(1, 1) = 5;
  EXPECT_EQ(copy(1, 1), 4);
}

TEST(TensorTest, MoveConstructor) {
  Tensor<int> original({{1, 2}, {3, 4}});
  Tensor<int> moved(std::move(original));
  EXPECT_EQ(moved.dim(), 4);
  EXPECT_EQ(moved(1, 1), 4);
}

TEST(TensorTest, AssignmentOperator) {
  Tensor<int> t1({{1, 2}, {3, 4}});
  Tensor<int> t2({{1}});
  t2 = t1;
  EXPECT_EQ(t2.dim(), 4);
  EXPECT_EQ(t2(0, 1), 2);

  t1(0, 1) = 5;
  EXPECT_EQ(t2(0, 1), 2);
}

TEST(TensorTest, ElementAccess) {
  Tensor<int> t({{1, 2}, {3, 4}});

  EXPECT_EQ(t(0, 0), 1);
  EXPECT_EQ(t(1, 1), 4);

  t(1, 0) = 5;
  EXPECT_EQ(t(1, 0), 5);

  const auto &ct = t;
  EXPECT_EQ(ct(0, 1), 2);
}

TEST(TensorTest, InvalidElementAccess) {
  Tensor<int> t({{1, 2}, {3, 4}});

  EXPECT_THROW(t.at(0), TensorError);
  EXPECT_THROW(t.at(0, 1, 2), TensorError);
  EXPECT_THROW(t.at(2, 0), TensorError);
  EXPECT_THROW(t.at(0, -1), TensorError);
}

TEST(TensorTest, Reshape) {
  Tensor<int> t({{1, 2}, {3, 4}});

  t.reshape({4});
  EXPECT_EQ(t.order(), 1);
  EXPECT_EQ(t.dim(), 4);
  EXPECT_EQ(t(3), 4);

  EXPECT_THROW(t.reshape({5}), TensorError);
}

TEST(TensorTest, Slice) {
  Tensor<int> t(txeo::TensorShape({4, 3}), 0);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      t(i, j) = i * 3 + j;

  auto slice = t.slice(1, 3);
  EXPECT_EQ(slice.order(), 2);
  EXPECT_EQ(slice.shape().axes_dims(), std::vector<int64_t>({2, 3}));
  EXPECT_EQ(slice(0, 0), 3);
  EXPECT_EQ(slice(1, 2), 8);
  EXPECT_THROW(t.slice(1, 10), TensorError);
  EXPECT_THROW(t.slice(11, 10), TensorError);
}

TEST(TensorTest, FillAndAssignment) {
  Tensor<int> t(txeo::TensorShape({2, 2}));
  t.fill(42);
  EXPECT_EQ(t(0, 0), 42);
  EXPECT_EQ(t(1, 1), 42);

  t = 7;
  EXPECT_EQ(t(0, 1), 7);
}

TEST(TensorTest, RandomInitialization) {
  Tensor<double> t(txeo::TensorShape({1000}));
  t.fill_with_uniform_random(0.0, 1.0, 42, 22);

  double max = *std::max_element(t.data(), t.data() + t.dim());
  double min = *std::min_element(t.data(), t.data() + t.dim());

  EXPECT_GE(min, 0.0);
  EXPECT_LE(max, 1.0);
  EXPECT_THROW(t.fill_with_uniform_random(10, 1, 42, 22), TensorError);

  Tensor<double> t1(txeo::TensorShape({0}));
  t1.fill_with_uniform_random(0.0, 1.0, 42, 22);
  EXPECT_EQ(t1.dim(), 0);

  Tensor<double> t2(txeo::TensorShape({1000}));
  t2.fill_with_uniform_random(0.0, 1.0);
  max = *std::max_element(t2.data(), t2.data() + t2.dim());
  min = *std::min_element(t2.data(), t2.data() + t2.dim());

  EXPECT_GE(min, 0.0);
  EXPECT_LE(max, 1.0);
}

TEST(TensorTest, Shuffle) {
  Tensor<double> t(txeo::TensorShape({1000}));
  for (int i = 0; i < 1000; ++i)
    t(i) = i;

  auto original = t.clone();
  t.shuffle();

  EXPECT_NE(std::equal(t.data(), t.data() + t.dim(), original.data()), true);
  std::sort(t.data(), t.data() + t.dim());
  EXPECT_TRUE(std::equal(t.data(), t.data() + t.dim(), original.data()));

  Tensor<double> t1(txeo::TensorShape({0}));
  t1.shuffle();
  EXPECT_EQ(t1.dim(), 0);
}

TEST(TensorTest, Squeeze) {
  Tensor<int> t(txeo::TensorShape({1, 3, 1, 4}));
  t.squeeze();
  EXPECT_EQ(t.order(), 2);
  EXPECT_EQ(t.shape().axes_dims(), std::vector<int64_t>({3, 4}));
}

TEST(TensorTest, Clone) {
  Tensor<int> original(txeo::TensorShape({2, 2}), 5);
  auto clone = original.clone();

  original(0, 0) = 10;
  EXPECT_EQ(clone(0, 0), 5);
  EXPECT_EQ(clone.dim(), 4);
}

TEST(TensorTest, EqualityOperators) {
  Tensor<int> t1(txeo::TensorShape({2, 2}), 5);
  Tensor<int> t2(txeo::TensorShape({2, 2}), 5);
  Tensor<int> t3(txeo::TensorShape({2, 2}), 6);
  Tensor<int> t4(txeo::TensorShape({2, 2, 1}), 6);

  EXPECT_TRUE(t1 == t2);
  EXPECT_FALSE(t1 == t3);
  EXPECT_TRUE(t1 != t3);
  EXPECT_FALSE(t1 == t4);
  EXPECT_TRUE(t1 != t4);
  EXPECT_FALSE(t1 != t2);
}

TEST(TensorTest, ScalarTensor) {
  Tensor<int> t({{1}});
  t = 42;
  EXPECT_EQ(t.dim(), 1);
  EXPECT_EQ(t(), 42);
  EXPECT_THROW(t.at(0), TensorError);
}

TEST(TensorTest, MemoryOperations) {
  Tensor<double> t(txeo::TensorShape({1000}));
  size_t expected_size = 1000 * sizeof(double);
  EXPECT_GE(t.memory_size(), expected_size);
}

TEST(TensorTest, ValidShareOperation) {
  Tensor<int> source(txeo::TensorShape({2, 3}), 5);
  Tensor<int> target(txeo::TensorShape({3, 2}));

  ASSERT_NO_THROW(target.view_of(source, txeo::TensorShape({6})));
  EXPECT_EQ(target.shape().axes_dims(), std::vector<int64_t>({6}));
  EXPECT_EQ(target.dim(), 6);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(target(i), 5);
  }

  Tensor<int> target1(txeo::TensorShape({0}), 5);
  target1.view_of(source, txeo::TensorShape({0}));
  EXPECT_EQ(target1.dim(), 0);
}

TEST(TensorTest, DimensionMismatch) {
  Tensor<int> source(txeo::TensorShape({2, 2}), 4);
  Tensor<int> target(txeo::TensorShape({4}));

  EXPECT_THROW(target.view_of(source, txeo::TensorShape({3})), TensorError);
  EXPECT_THROW(target.view_of(source, txeo::TensorShape({5})), TensorError);
}

TEST(TensorTest, ValidFlatten) {
  Tensor<float> original(txeo::TensorShape({2, 3}), 1.5f);
  auto flattened = original.flatten();

  EXPECT_EQ(flattened.order(), 1);
  EXPECT_EQ(flattened.dim(), 6);
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(flattened(i), 1.5f);
  }
}

TEST(TensorTest, ValidScalarAccess) {
  Tensor<int> scalar(txeo::TensorShape({}));
  scalar = 42;

  EXPECT_EQ(scalar.at(), 42);
  const auto &const_scalar = scalar;
  EXPECT_EQ(const_scalar.at(), 42);
}

TEST(TensorTest, NonScalarAccess) {
  Tensor<int> matrix(txeo::TensorShape({2, 2}), 5);
  EXPECT_THROW(matrix.at(), TensorError);
  const auto &const_matrix = matrix;
  EXPECT_THROW(const_matrix.at(), TensorError);
}

TEST(TensorTest, ScalarAccess) {
  Tensor<double> scalar(txeo::TensorShape({1}), 3.14);
  const auto &const_scalar = scalar;

  EXPECT_DOUBLE_EQ(scalar(), 3.14);
  EXPECT_DOUBLE_EQ(const_scalar(), 3.14);
}

TEST(TensorTest, ValidMove) {
  Tensor<int> source(txeo::TensorShape({3}), {1, 2, 3});
  Tensor<int> target(txeo::TensorShape({1}));

  target = std::move(source);

  EXPECT_EQ(target.dim(), 3);
  EXPECT_EQ(target(0), 1);
}

TEST(TensorTest, ValidConstruction) {
  Tensor<int> t({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});

  EXPECT_EQ(t.order(), 3);
  EXPECT_EQ(t.shape().axes_dims(), std::vector<int64_t>({2, 2, 2}));
  EXPECT_EQ(t(1, 0, 1), 6);
}

TEST(TensorTest, InconsistentDimensions) {
  EXPECT_THROW(Tensor<int>({{{1, 2}, {3}}, {{4, 5}, {6, 7}}}), TensorError);
}

TEST(TensorTest, ValidConstructionShapeValues) {
  Tensor<float> t(txeo::TensorShape({2, 2}), {1.1f, 2.2f, 3.3f, 4.4f});

  EXPECT_EQ(t.order(), 2);
  EXPECT_FLOAT_EQ(t(1, 1), 4.4f);
}

TEST(TensorTest, SizeMismatch) {
  std::vector<int> data{1, 2, 3};
  EXPECT_THROW(Tensor<int>(txeo::TensorShape({2, 2}), data), TensorError);
}

TEST(TensorTest, ZeroDimTensor) {
  Tensor<double> t(txeo::TensorShape({}), 3.14);
  EXPECT_EQ(t.dim(), 1);
  EXPECT_DOUBLE_EQ(t(), 3.14);
}

TEST(TensorTest, MutableIteration) {
  txeo::Tensor<int> tensor(TensorShape({5}), {1, 2, 3, 4, 5});

  auto it = std::begin(tensor);
  EXPECT_EQ(*it, 1);
  ++it;
  EXPECT_EQ(*it, 2);

  it += 2;
  EXPECT_EQ(*it, 4);

  it--;
  EXPECT_EQ(*it, 3);

  it -= 2;
  EXPECT_EQ(*it, 1);
}

TEST(TensorTest, ConstIteration) {
  const txeo::Tensor<int> tensor(TensorShape({5}), {10, 20, 30, 40, 50});

  auto it = std::cbegin(tensor);
  EXPECT_EQ(*it, 10);
  ++it;
  EXPECT_EQ(*it, 20);

  it += 2;
  EXPECT_EQ(*it, 40);

  it--;
  EXPECT_EQ(*it, 30);

  it -= 2;
  EXPECT_EQ(*it, 10);
}

TEST(TensorTest, IteratorComparison) {
  txeo::Tensor<int> tensor(TensorShape({5}), {1, 2, 3, 4, 5});

  auto it1 = std::begin(tensor);
  auto it2 = std::begin(tensor);
  auto it3 = std::end(tensor);

  EXPECT_TRUE(it1 == it2);
  EXPECT_FALSE(it1 == it3);

  ++it1;
  EXPECT_TRUE(it1 != it2);

  EXPECT_TRUE(it1 > it2);
  EXPECT_TRUE(it2 < it1);

  it1 += 3;
  EXPECT_TRUE(it1 >= it2);
  EXPECT_TRUE(it2 <= it1);
}

TEST(TensorTest, AdditionOperator) {
  Tensor<float> t1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> t2({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

  Tensor<float> result = t1 + t2;

  ASSERT_EQ(result.shape(), txeo::TensorShape({2, 2}));
  EXPECT_FLOAT_EQ(result(0, 0), 6.0f);
  EXPECT_FLOAT_EQ(result(0, 1), 8.0f);
  EXPECT_FLOAT_EQ(result(1, 0), 10.0f);
  EXPECT_FLOAT_EQ(result(1, 1), 12.0f);
}

TEST(TensorTest, SubtractionOperator) {
  Tensor<double> t1({3}, {5.0, 6.0, 7.0});
  Tensor<double> t2({3}, {1.0, 2.0, 3.0});

  Tensor<double> result = t1 - t2;

  ASSERT_EQ(result.shape(), txeo::TensorShape({3}));
  EXPECT_DOUBLE_EQ(result(0), 4.0);
  EXPECT_DOUBLE_EQ(result(1), 4.0);
  EXPECT_DOUBLE_EQ(result(2), 4.0);
}

TEST(TensorTest, ScalarMultiplication) {
  Tensor<int> t1({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor<int> result = t1 * 2;
  Tensor<int> result2 = 3 * t1;

  ASSERT_EQ(result.shape(), txeo::TensorShape({2, 3}));
  EXPECT_EQ(result(0, 0), 2);
  EXPECT_EQ(result(1, 2), 12);
  EXPECT_TRUE(result2 == txeo::Tensor<int>({2, 3}, {3, 6, 9, 12, 15, 18}));
}

TEST(TensorTest, CompoundAddition) {
  Tensor<float> t1({2}, {1.5f, 2.5f});
  Tensor<float> t2({2}, {0.5f, 1.5f});

  t1 += t2;

  ASSERT_EQ(t1.shape(), txeo::TensorShape({2}));
  EXPECT_FLOAT_EQ(t1(0), 2.0f);
  EXPECT_FLOAT_EQ(t1(1), 4.0f);
}

TEST(TensorTest, CompoundSubtraction) {
  Tensor<double> t1({3}, {10.0, 20.0, 30.0});
  Tensor<double> t2({3}, {1.0, 2.0, 3.0});

  t1 -= t2;

  ASSERT_EQ(t1.shape(), txeo::TensorShape({3}));
  EXPECT_DOUBLE_EQ(t1(0), 9.0);
  EXPECT_DOUBLE_EQ(t1(1), 18.0);
  EXPECT_DOUBLE_EQ(t1(2), 27.0);
}

TEST(TensorTest, CompoundScalarMultiplication) {
  Tensor<int> t1({2, 2}, {1, 2, 3, 4});

  t1 *= 3;

  ASSERT_EQ(t1.shape(), txeo::TensorShape({2, 2}));
  EXPECT_EQ(t1(0, 0), 3);
  EXPECT_EQ(t1(1, 1), 12);
}

TEST(TensorTest, ShapeMismatchAddition) {
  Tensor<float> t1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor<float> t2({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  EXPECT_THROW(t1 + t2, TensorOpError);
}

TEST(TensorTest, EmptyTensorTest) {
  Tensor<float> empty({0});
  Tensor<float> t1({0});

  EXPECT_THROW(empty + t1, TensorOpError);
  EXPECT_THROW(t1 *= 2.0f, TensorOpError);
}

TEST(TensorTest, ScalarMultiplicationDouble) {
  Tensor<double> t1({2}, {1.5, 2.5});
  Tensor<double> result = t1 * 2.0;

  ASSERT_EQ(result.shape(), txeo::TensorShape({2}));
  EXPECT_DOUBLE_EQ(result(0), 3.0);
  EXPECT_DOUBLE_EQ(result(1), 5.0);
}

TEST(TensorTest, NegativeScalarMultiplication) {
  Tensor<float> t1({3}, {1.0f, 2.0f, 3.0f});
  Tensor<float> result = t1 * -1.0f;

  EXPECT_FLOAT_EQ(result(0), -1.0f);
  EXPECT_FLOAT_EQ(result(1), -2.0f);
  EXPECT_FLOAT_EQ(result(2), -3.0f);
}

TEST(TensorTest, DivisionOperatorTensorScalar) {
  Tensor<int> t({3}, {10, 20, 30});
  Tensor<int> result = t / 2;
  ASSERT_EQ(result.shape(), txeo::TensorShape({3}));
  EXPECT_EQ(result.data()[0], 5);
  EXPECT_EQ(result.data()[1], 10);
  EXPECT_EQ(result.data()[2], 15);
}

TEST(TensorTest, FloatingPointDivision) {
  Tensor<double> t({2}, {7.5, 12.5});
  Tensor<double> result = t / 2.5;
  ASSERT_DOUBLE_EQ(result.data()[0], 3.0);
  ASSERT_DOUBLE_EQ(result.data()[1], 5.0);
}

TEST(TensorTest, SumByScalar) {
  Tensor<float> t({3}, {1.1f, 2.2f, 3.3f});
  t += 10.0f;
  EXPECT_FLOAT_EQ(t.data()[0], 11.1f);
  EXPECT_FLOAT_EQ(t.data()[1], 12.2f);
  EXPECT_FLOAT_EQ(t.data()[2], 13.3f);
}

TEST(TensorTest, SubtractByScalar) {
  Tensor<int> t({4}, {15, 25, 35, 45});
  t -= 5;
  EXPECT_EQ(t.data()[0], 10);
  EXPECT_EQ(t.data()[1], 20);
  EXPECT_EQ(t.data()[2], 30);
  EXPECT_EQ(t.data()[3], 40);
}

TEST(TensorTest, DivideByScalarAndOperator) {
  Tensor<double> t({3}, {9.0, 21.0, 36.0});

  t /= 3.0;
  ASSERT_DOUBLE_EQ(t.data()[0], 3.0);
  ASSERT_DOUBLE_EQ(t.data()[1], 7.0);
  ASSERT_DOUBLE_EQ(t.data()[2], 12.0);

  t /= 2.0;
  ASSERT_DOUBLE_EQ(t.data()[0], 1.5);
  ASSERT_DOUBLE_EQ(t.data()[1], 3.5);
  ASSERT_DOUBLE_EQ(t.data()[2], 6.0);
}

TEST(TensorTest, OperationChaining) {
  Tensor<int> t({2}, {100, 200});
  ((t += 50) -= 75) /= 5;
  EXPECT_EQ(t.data()[0], (100 + 50 - 75) / 5);
  EXPECT_EQ(t.data()[1], (200 + 50 - 75) / 5);
}

TEST(TensorTest, DivideByOne) {
  Tensor<int> t({3}, {5, 10, 15});
  t /= 1;
  EXPECT_EQ(t.data()[0], 5);
  EXPECT_EQ(t.data()[1], 10);
  EXPECT_EQ(t.data()[2], 15);
}

TEST(TensorTest, EmptyTensorTest2) {
  Tensor<float> t({0});
  EXPECT_THROW(t += 10.0f, TensorOpError);
}

TEST(TensorTest, IntegerDivisionTruncation) {
  Tensor<int> t({3}, {7, 11, 15});
  t /= 2;
  EXPECT_EQ(t.data()[0], 3);
  EXPECT_EQ(t.data()[1], 5);
  EXPECT_EQ(t.data()[2], 7);
}

TEST(TensorTest, AdditionOperator2) {
  txeo::Tensor<int> t1({2, 2}, {1, 2, 3, 4});
  auto result1 = t1 + 5;
  EXPECT_EQ(result1.shape(), txeo::TensorShape({2, 2}));
  EXPECT_EQ(result1.data()[0], 6);
  EXPECT_EQ(result1.data()[1], 7);
  EXPECT_EQ(result1.data()[2], 8);
  EXPECT_EQ(result1.data()[3], 9);

  txeo::Tensor<float> t2({3}, {-1.5f, 0.0f, 3.5f});
  auto result2 = t2 + 2.5f;
  EXPECT_FLOAT_EQ(result2.data()[0], 1.0f);
  EXPECT_FLOAT_EQ(result2.data()[1], 2.5f);
  EXPECT_FLOAT_EQ(result2.data()[2], 6.0f);
}

TEST(TensorTest, SubtractionOperatorTensorScalar) {
  txeo::Tensor<double> t1({2}, {5.0, 10.0});
  auto result1 = t1 - 2.5;
  EXPECT_EQ(result1.shape(), txeo::TensorShape({2}));
  EXPECT_DOUBLE_EQ(result1.data()[0], 2.5);
  EXPECT_DOUBLE_EQ(result1.data()[1], 7.5);

  txeo::Tensor<int> t2({3}, {8, 5, 3});
  auto result2 = t2 - 0;
  EXPECT_EQ(result2.data()[0], 8);
  EXPECT_EQ(result2.data()[1], 5);
  EXPECT_EQ(result2.data()[2], 3);
}

TEST(TensorTest, SubtractionOperatorScalarTensor) {
  txeo::Tensor<int> t1({2, 2}, {2, 3, 4, 5});
  auto result1 = 10 - t1;
  EXPECT_EQ(result1.data()[0], 8);
  EXPECT_EQ(result1.data()[1], 7);
  EXPECT_EQ(result1.data()[2], 6);
  EXPECT_EQ(result1.data()[3], 5);

  txeo::Tensor<float> t2({3}, {1.5f, 3.0f, 4.5f});
  auto result2 = 5.0f - t2;
  EXPECT_FLOAT_EQ(result2.data()[0], 3.5f);
  EXPECT_FLOAT_EQ(result2.data()[1], 2.0f);
  EXPECT_FLOAT_EQ(result2.data()[2], 0.5f);
}

TEST(TensorTest, DivisionOperatorTensorScalar2) {
  txeo::Tensor<double> t1({4}, {10.0, 20.0, 30.0, 40.0});
  auto result1 = t1 / 2.0;
  EXPECT_EQ(result1.data()[0], 5.0);
  EXPECT_EQ(result1.data()[1], 10.0);
  EXPECT_EQ(result1.data()[2], 15.0);
  EXPECT_EQ(result1.data()[3], 20.0);

  txeo::Tensor<int> t2({3}, {15, 25, 35});
  auto result2 = t2 / 5;
  EXPECT_EQ(result2.data()[0], 3);
  EXPECT_EQ(result2.data()[1], 5);
  EXPECT_EQ(result2.data()[2], 7);

  txeo::Tensor<float> t3({1}, {5.0f});
  EXPECT_THROW(t3 / 0.0f, txeo::TensorOpError);
}

TEST(TensorTest, DivisionOperatorScalarTensor) {
  txeo::Tensor<int> t1({3}, {2, 5, 10});
  auto result1 = 100 / t1;
  EXPECT_EQ(result1.data()[0], 50);
  EXPECT_EQ(result1.data()[1], 20);
  EXPECT_EQ(result1.data()[2], 10);

  txeo::Tensor<double> t2({2}, {4.0, 2.5});
  auto result2 = 10.0 / t2;
  EXPECT_DOUBLE_EQ(result2.data()[0], 2.5);
  EXPECT_DOUBLE_EQ(result2.data()[1], 4.0);

  txeo::Tensor<float> t3({2}, {5.0f, 0.0f});
  EXPECT_THROW(10.0f / t3, txeo::TensorOpError);
}

TEST(TensorTest, ShapePreservation) {
  txeo::Tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});

  auto add_result = t + 2;
  EXPECT_EQ(add_result.shape(), txeo::TensorShape({2, 3}));

  auto sub_result = 10 - t;
  EXPECT_EQ(sub_result.shape(), txeo::TensorShape({2, 3}));

  auto div_result = t / 1;
  EXPECT_EQ(div_result.shape(), txeo::TensorShape({2, 3}));
}

TEST(TensorTest, Immutability) {
  txeo::Tensor<int> orig({2}, {5, 10});
  auto result = orig + 3;

  EXPECT_EQ(orig.data()[0], 5);
  EXPECT_EQ(orig.data()[1], 10);
  EXPECT_EQ(result.data()[0], 8);
  EXPECT_EQ(result.data()[1], 13);
}

TEST(TensorTest, IncreaseDimensionAddsNewAxis) {
  Tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  t.increase_dimension(1, -1);

  Tensor<int> resp({2, 4}, {1, 2, 3, -1, 4, 5, 6, -1});

  EXPECT_TRUE(t == resp);
}

TEST(TensorTest, PowerRaisesElements) {
  Tensor<float> t({2}, {2.0f, 3.0f});
  t.power(3.0f);

  EXPECT_FLOAT_EQ(t(0), 8.0f);
  EXPECT_FLOAT_EQ(t(1), 27.0f);
}

TEST(TensorTest, SquareSquaresElements) {
  Tensor<double> t({3}, {3.0, 4.0, -5.0});
  t.square();

  EXPECT_DOUBLE_EQ(t(0), 9.0);
  EXPECT_DOUBLE_EQ(t(1), 16.0);
  EXPECT_DOUBLE_EQ(t(2), 25.0);
}

TEST(TensorTest, SqrtComputesSquareRoot) {
  Tensor<float> t({4}, {9.0f, 16.0f, 2.0f, 5.0f});
  t.sqrt();

  EXPECT_FLOAT_EQ(t(0), 3.0f);
  EXPECT_FLOAT_EQ(t(1), 4.0f);
  EXPECT_NEAR(t(2), 1.4142f, 0.0001f);
  EXPECT_NEAR(t(3), 2.2361f, 0.0001f);
}

TEST(TensorTest, AbsComputesAbsoluteValues) {
  Tensor<int> t({2, 2}, {1, -2, -3, 4});
  t.abs();

  EXPECT_EQ(t(0, 0), 1);
  EXPECT_EQ(t(0, 1), 2);
  EXPECT_EQ(t(1, 0), 3);
  EXPECT_EQ(t(1, 1), 4);
}

TEST(TensorTest, PermuteReordersDimensions) {
  Tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
  t.permute({1, 0});

  ASSERT_EQ(t.shape().axis_dim(0), 3);
  ASSERT_EQ(t.shape().axis_dim(1), 2);
  EXPECT_EQ(t(0, 0), 1);
  EXPECT_EQ(t(2, 1), 6);
}

TEST(TensorTest, MinMaxNormalization) {
  Tensor<float> t({3}, {10.0f, 20.0f, 30.0f});
  t.normalize(0, NormalizationType::MIN_MAX);

  EXPECT_FLOAT_EQ(t(0), 0.0f);
  EXPECT_FLOAT_EQ(t(1), 0.5f);
  EXPECT_FLOAT_EQ(t(2), 1.0f);
}

TEST(TensorTest, ZScoreNormalization) {

  txeo::Tensor<double> tens7({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  tens7.normalize(0, txeo::NormalizationType::Z_SCORE);
  txeo::Tensor<double> resp7({3, 3}, {-1, -1, -1, 0, 0, 0, 1, 1, 1});
  EXPECT_TRUE(tens7 == resp7);
}

TEST(TensorTest, VectorDotProduct) {
  Tensor<int> a({3}, {1, 2, 3});
  Tensor<int> b({3}, {4, 5, 6});

  int result = a.dot(b);
  EXPECT_EQ(result, 1 * 4 + 2 * 5 + 3 * 6);
}

TEST(TensorTest, MatrixDotProduct) {
  Tensor<int> a({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor<int> b({3, 2}, {7, 8, 9, 10, 11, 12});

  EXPECT_EQ(a.dot(b), 1 * 7 + 2 * 8 + 3 * 9 + 4 * 10 + 5 * 11 + 6 * 12);
}

TEST(TensorTest, InvalidPermutationThrows) {
  Tensor<int> t({2, 3});
  EXPECT_THROW(t.permute({2, 0}), TensorFuncError);
}

TEST(TensorTest, DotProductDimensionMismatch) {
  Tensor<int> a({3}, {1, 2, 3});
  Tensor<int> b({4}, {1, 2, 3, 4});
  EXPECT_THROW(a.dot(b), TensorOpError);
}

} // namespace txeo