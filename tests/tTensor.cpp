#include "txeo/Tensor.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace txeo {

TEST(TensorTest, ShapeConstructor) {
  TensorShape shape({2, 3, 4});
  Tensor<int> t(shape);
  EXPECT_EQ(t.dim(), 24);
  EXPECT_EQ(t.order(), 3);
  EXPECT_EQ(t.shape().axes_dims(), std::vector<int64_t>({2, 3, 4}));
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
  EXPECT_EQ(original.dim(), 0); // NOLINT
}

TEST(TensorTest, AssignmentOperator) {
  Tensor<int> t1({{1, 2}, {3, 4}});
  Tensor<int> t2;
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
  Tensor<int> t({4, 3}, 0);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 3; ++j)
      t(i, j) = i * 3 + j;

  auto slice = t.slice(1, 3);
  EXPECT_EQ(slice.order(), 2);
  EXPECT_EQ(slice.shape().axes_dims(), std::vector<int64_t>({2, 3}));
  EXPECT_EQ(slice(0, 0), 3);
  EXPECT_EQ(slice(1, 2), 8);
}

TEST(TensorTest, FillAndAssignment) {
  Tensor<int> t({2, 2});
  t.fill(42);
  EXPECT_EQ(t(0, 0), 42);
  EXPECT_EQ(t(1, 1), 42);

  t = 7;
  EXPECT_EQ(t(0, 1), 7);
}

TEST(TensorTest, RandomInitialization) {
  Tensor<double> t({1000});
  t.fill_with_uniform_random(0.0, 1.0, 42);

  double min = *std::min_element(t.data(), t.data() + t.dim());
  double max = *std::max_element(t.data(), t.data() + t.dim());

  EXPECT_GE(min, 0.0);
  EXPECT_LE(max, 1.0);
}

TEST(TensorTest, Shuffle) {
  Tensor<int> t({100});
  for (int i = 0; i < 100; ++i)
    t(i) = i;

  auto original = t.clone();
  t.shuffle();

  EXPECT_NE(std::equal(t.data(), t.data() + t.dim(), original.data()), true);
  std::sort(t.data(), t.data() + t.dim());
  EXPECT_TRUE(std::equal(t.data(), t.data() + t.dim(), original.data()));
}

TEST(TensorTest, Squeeze) {
  Tensor<int> t({1, 3, 1, 4});
  t.squeeze();
  EXPECT_EQ(t.order(), 2);
  EXPECT_EQ(t.shape().axes_dims(), std::vector<int64_t>({3, 4}));
}

TEST(TensorTest, Clone) {
  Tensor<int> original({2, 2}, 5);
  auto clone = original.clone();

  original(0, 0) = 10;
  EXPECT_EQ(clone(0, 0), 5);
  EXPECT_EQ(clone.dim(), 4);
}

TEST(TensorTest, EqualityOperators) {
  Tensor<int> t1({2, 2}, 5);
  Tensor<int> t2({2, 2}, 5);
  Tensor<int> t3({2, 2}, 6);

  EXPECT_TRUE(t1 == t2);
  EXPECT_FALSE(t1 == t3);
  EXPECT_TRUE(t1 != t3);
}

TEST(TensorTest, ScalarTensor) {
  Tensor<int> t;
  t = 42;
  EXPECT_EQ(t.dim(), 1);
  EXPECT_EQ(t(), 42);
  EXPECT_THROW(t.at(0), TensorError);
}

TEST(TensorTest, MemoryOperations) {
  Tensor<double> t({1000});
  size_t expected_size = 1000 * sizeof(double);
  EXPECT_GE(t.memory_size(), expected_size);
}
} // namespace txeo