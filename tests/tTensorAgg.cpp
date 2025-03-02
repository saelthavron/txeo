#include "txeo/Tensor.h"
#include "txeo/TensorAgg.h"
#include <cmath>
#include <gtest/gtest.h>

namespace txeo {

TEST(TensorAggTest, ReduceSum) {

  Tensor<int> tensor1D({5}, {1, 2, 3, 4, 5});
  auto result1D = TensorAgg<int>::reduce_sum(tensor1D, {0});
  EXPECT_EQ(result1D.shape(), TensorShape({}));
  EXPECT_EQ(result1D(), 15);

  Tensor<int> tensor2D({{1, 2, 3}, {4, 5, 6}});
  auto result2D_axis0 = TensorAgg<int>::reduce_sum(tensor2D, {0});
  EXPECT_EQ(result2D_axis0.shape(), TensorShape({3}));
  EXPECT_EQ(result2D_axis0(0), 5);
  EXPECT_EQ(result2D_axis0(1), 7);
  EXPECT_EQ(result2D_axis0(2), 9);

  auto result2D_axis1 = TensorAgg<int>::reduce_sum(tensor2D, {1});
  EXPECT_EQ(result2D_axis1.shape(), TensorShape({2}));
  EXPECT_EQ(result2D_axis1(0), 6);
  EXPECT_EQ(result2D_axis1(1), 15);

  Tensor<int> tensor3D({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
  auto result3D = TensorAgg<int>::reduce_sum(tensor3D, {0, 1});
  EXPECT_EQ(result3D.shape(), TensorShape({2}));
  EXPECT_EQ(result3D(0), 16);
  EXPECT_EQ(result3D(1), 20);
}

TEST(TensorAggTest, ReduceMean) {

  Tensor<int> tensor1D({5}, {1, 2, 3, 4, 5});
  auto result1D = TensorAgg<int>::reduce_mean(tensor1D, {0});
  EXPECT_EQ(result1D.shape(), TensorShape({}));
  EXPECT_EQ(result1D(), 3);

  Tensor<int> tensor2D({{1, 2, 3}, {4, 5, 6}});
  auto result2D_axis0 = TensorAgg<int>::reduce_mean(tensor2D, {0});
  EXPECT_EQ(result2D_axis0.shape(), TensorShape({3}));
  EXPECT_EQ(result2D_axis0(0), 2);
  EXPECT_EQ(result2D_axis0(1), 3);
  EXPECT_EQ(result2D_axis0(2), 4);
}

TEST(TensorAggTest, ReduceMax) {

  Tensor<int> tensor1D({5}, {1, 2, 3, 4, 5});
  auto result1D = TensorAgg<int>::reduce_max(tensor1D, {0});
  EXPECT_EQ(result1D.shape(), TensorShape({}));
  EXPECT_EQ(result1D(), 5);

  Tensor<int> tensor2D({{1, 2, 3}, {4, 5, 6}});
  auto result2D_axis1 = TensorAgg<int>::reduce_max(tensor2D, {1});
  EXPECT_EQ(result2D_axis1.shape(), TensorShape({2}));
  EXPECT_EQ(result2D_axis1(0), 3);
  EXPECT_EQ(result2D_axis1(1), 6);
}

TEST(TensorAggTest, ReduceMin) {

  Tensor<int> tensor1D({5}, {1, 2, 3, 4, 5});
  auto result1D = TensorAgg<int>::reduce_min(tensor1D, {0});
  EXPECT_EQ(result1D.shape(), TensorShape({}));
  EXPECT_EQ(result1D(), 1);

  Tensor<int> tensor2D({{1, 2, 3}, {4, 5, 6}});
  auto result2D_axis0 = TensorAgg<int>::reduce_min(tensor2D, {0});
  EXPECT_EQ(result2D_axis0.shape(), TensorShape({3}));
  EXPECT_EQ(result2D_axis0(0), 1);
  EXPECT_EQ(result2D_axis0(1), 2);
  EXPECT_EQ(result2D_axis0(2), 3);
}

TEST(TensorAggTest, ArgMax) {

  Tensor<int> tensor1D({5}, {1, 2, 3, 4, 5});
  auto result1D = TensorAgg<int>::arg_max(tensor1D, 0);
  EXPECT_EQ(result1D.shape(), TensorShape({}));
  EXPECT_EQ(result1D(), 4);

  Tensor<int> tensor2D({{1, 2, 3}, {4, 5, 6}});
  auto result2D_axis1 = TensorAgg<int>::arg_max(tensor2D, 1);

  std::cout << "Tensor After T: \n";
  std::cout << result2D_axis1(1) << std::endl;
  EXPECT_EQ(result2D_axis1.shape(), TensorShape({2}));
  EXPECT_EQ(result2D_axis1(0), 2);
  EXPECT_EQ(result2D_axis1(1), 2);
}

TEST(TensorAggTest, ArgMin) {

  Tensor<int> tensor1D({5}, {1, 2, 3, 4, 5});
  auto result1D = TensorAgg<int>::arg_min(tensor1D, 0);
  EXPECT_EQ(result1D.shape(), TensorShape({}));
  EXPECT_EQ(result1D(), 0);

  Tensor<int> tensor2D({{1, 2, 3}, {4, 5, 6}});
  auto result2D_axis0 = TensorAgg<int>::arg_min(tensor2D, 0);
  EXPECT_EQ(result2D_axis0.shape(), TensorShape({3}));
  EXPECT_EQ(result2D_axis0(0), 0);
  EXPECT_EQ(result2D_axis0(1), 0);
  EXPECT_EQ(result2D_axis0(2), 0);
}

TEST(TensorAggTest, Abs) {
  Tensor<int> tensor({4}, {-1, 2, -3, 4});
  auto result = TensorAgg<int>::abs(tensor);
  EXPECT_EQ(result.shape(), TensorShape({4}));
  EXPECT_EQ(result(0), 1);
  EXPECT_EQ(result(1), 2);
  EXPECT_EQ(result(2), 3);
  EXPECT_EQ(result(3), 4);
}

TEST(TensorAggTest, Variance) {
  Tensor<double> tensor({5}, {1, 2, 3, 4, 5});
  auto result = TensorAgg<double>::variance(tensor);
  EXPECT_NEAR(result, 2.0, 1e-5);
}

TEST(TensorAggTest, StandardDeviation) {
  Tensor<double> tensor({5}, {1, 2, 3, 4, 5});
  auto result = TensorAgg<double>::standard_deviation(tensor);
  EXPECT_NEAR(result, std::sqrt(2.0), 1e-5);
}

TEST(TensorAggTest, ReduceProd) {
  Tensor<int> tensor1D({3}, {2, 3, 4});
  auto result1D = TensorAgg<int>::reduce_prod(tensor1D, {0});
  EXPECT_EQ(result1D.shape(), TensorShape({}));
  EXPECT_EQ(result1D(), 24);

  Tensor<int> tensor2D({{2, 3}, {4, 5}});
  auto result2D_axis0 = TensorAgg<int>::reduce_prod(tensor2D, {0});
  EXPECT_EQ(result2D_axis0.shape(), TensorShape({2}));
  EXPECT_EQ(result2D_axis0(0), 8);
  EXPECT_EQ(result2D_axis0(1), 15);

  auto result2D_axis1 = TensorAgg<int>::reduce_prod(tensor2D, {1});
  EXPECT_EQ(result2D_axis1.shape(), TensorShape({2}));
  EXPECT_EQ(result2D_axis1(0), 6);
  EXPECT_EQ(result2D_axis1(1), 20);
}

TEST(TensorAggTest, ReduceEuclideanNorm) {
  Tensor<double> tensor1D({3}, {3.0, 4.0, 0.0});
  auto result1D = TensorAgg<double>::reduce_euclidean_norm(tensor1D, {0});
  EXPECT_EQ(result1D.shape(), TensorShape({}));
  EXPECT_NEAR(result1D(), 5.0, 1e-5);

  Tensor<double> tensor2D({{3.0, 4.0}, {0.0, 5.0}});
  auto result2D_axis0 = TensorAgg<double>::reduce_euclidean_norm(tensor2D, {0});
  EXPECT_EQ(result2D_axis0.shape(), TensorShape({2}));
  EXPECT_NEAR(result2D_axis0(0), 3.0, 1e-5);
  EXPECT_NEAR(result2D_axis0(1), 6.40312, 1e-5);
}

TEST(TensorAggTest, ReduceAll) {
  Tensor<bool> tensor1D({3}, {true, true, false});
  auto result1D = TensorAgg<bool>::reduce_all(tensor1D, {0});
  EXPECT_EQ(result1D.shape(), TensorShape({}));
  EXPECT_EQ(result1D(), false);

  Tensor<bool> tensor2D({{true, true}, {false, true}});
  auto result2D_axis1 = TensorAgg<bool>::reduce_all(tensor2D, {1});
  EXPECT_EQ(result2D_axis1.shape(), TensorShape({2}));
  EXPECT_EQ(result2D_axis1(0), true);
  EXPECT_EQ(result2D_axis1(1), false);
}

TEST(TensorAggTest, ReduceAny) {
  Tensor<bool> tensor1D({3}, {false, false, true});
  auto result1D = TensorAgg<bool>::reduce_any(tensor1D, {0});
  EXPECT_EQ(result1D.shape(), TensorShape({}));
  EXPECT_EQ(result1D(), true);

  Tensor<bool> tensor2D({{false, false}, {true, false}});
  auto result2D_axis0 = TensorAgg<bool>::reduce_any(tensor2D, {0});
  EXPECT_EQ(result2D_axis0.shape(), TensorShape({2}));
  EXPECT_EQ(result2D_axis0(0), true);
  EXPECT_EQ(result2D_axis0(1), false);
}

TEST(TensorAggTest, CumulativeSum) {
  Tensor<int> tensor1D({4}, {1, 2, 3, 4});
  auto result1D = TensorAgg<int>::cumulative_sum(tensor1D, 0);
  EXPECT_EQ(result1D.shape(), TensorShape({4}));
  EXPECT_EQ(result1D(0), 1);
  EXPECT_EQ(result1D(1), 3);
  EXPECT_EQ(result1D(2), 6);
  EXPECT_EQ(result1D(3), 10);

  Tensor<int> tensor2D({{1, 2}, {3, 4}});
  auto result2D_axis1 = TensorAgg<int>::cumulative_sum(tensor2D, 1);
  EXPECT_EQ(result2D_axis1.shape(), TensorShape({2, 2}));
  EXPECT_EQ(result2D_axis1(0, 0), 1);
  EXPECT_EQ(result2D_axis1(0, 1), 3);
  EXPECT_EQ(result2D_axis1(1, 0), 3);
  EXPECT_EQ(result2D_axis1(1, 1), 7);
}

TEST(TensorAggTest, CumulativeProd) {
  Tensor<int> tensor1D({4}, {1, 2, 3, 4});
  auto result1D = TensorAgg<int>::cumulative_prod(tensor1D, 0);
  EXPECT_EQ(result1D.shape(), TensorShape({4}));
  EXPECT_EQ(result1D(0), 1);
  EXPECT_EQ(result1D(1), 2);
  EXPECT_EQ(result1D(2), 6);
  EXPECT_EQ(result1D(3), 24);

  Tensor<int> tensor2D({{1, 2}, {3, 4}});
  auto result2D_axis0 = TensorAgg<int>::cumulative_prod(tensor2D, 0);
  EXPECT_EQ(result2D_axis0.shape(), TensorShape({2, 2}));
  EXPECT_EQ(result2D_axis0(0, 0), 1);
  EXPECT_EQ(result2D_axis0(0, 1), 2);
  EXPECT_EQ(result2D_axis0(1, 0), 3);
  EXPECT_EQ(result2D_axis0(1, 1), 8);
}

} // namespace txeo