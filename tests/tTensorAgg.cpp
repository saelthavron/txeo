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

} // namespace txeo