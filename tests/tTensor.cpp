#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "txeo/Tensor.h"
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
  EXPECT_EQ(copy(1, 1), 4); // Deep copy verification
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
  EXPECT_EQ(t2(0, 1), 2); // Deep copy verification
}

TEST(TensorTest, ElementAccess) {
  Tensor<int> t({{1, 2}, {3, 4}});

  // Valid access
  EXPECT_EQ(t(0, 0), 1);
  EXPECT_EQ(t(1, 1), 4);

  // Modification
  t(1, 0) = 5;
  EXPECT_EQ(t(1, 0), 5);

  // Const access
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

  // Valid reshape
  t.reshape({4});
  EXPECT_EQ(t.order(), 1);
  EXPECT_EQ(t.dim(), 4);
  EXPECT_EQ(t(3), 4);

  // Invalid reshape
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

// TEST(TensorTest, SelfAssignment) {
//   Tensor<int> tensor(txeo::TensorShape({2}), {4, 5});
//   tensor = std::move(tensor); // Should handle gracefully

//   EXPECT_EQ(tensor.dim(), 2);
//   EXPECT_EQ(tensor(1), 5);
// }

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

} // namespace txeo