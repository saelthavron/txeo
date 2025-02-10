
#include "txeo/detail/utils.h"
#include "gtest/gtest.h"
#include <tensorflow/core/framework/types.pb.h>

namespace tf = tensorflow;

TEST(UtilsTest, TestTypes) {
  EXPECT_EQ(txeo::detail::get_tf_dtype<short>(), tf::DT_INT16);
  EXPECT_EQ(txeo::detail::get_tf_dtype<int>(), tf::DT_INT32);
  EXPECT_TRUE(txeo::detail::get_tf_dtype<long>() == tf::DT_INT32 ||
              txeo::detail::get_tf_dtype<long>() == tf::DT_INT64);
  EXPECT_TRUE(txeo::detail::get_tf_dtype<long long>() == tf::DT_INT64);
  EXPECT_EQ(txeo::detail::get_tf_dtype<float>(), tf::DT_FLOAT);
  EXPECT_EQ(txeo::detail::get_tf_dtype<double>(), tf::DT_DOUBLE);
  EXPECT_EQ(txeo::detail::get_tf_dtype<bool>(), tf::DT_BOOL);

  EXPECT_EQ(txeo::detail::get_tf_dtype<txeo::detail::cpp_type<tf::DT_INT16>>(), tf::DT_INT16);
  EXPECT_EQ(txeo::detail::get_tf_dtype<txeo::detail::cpp_type<tf::DT_INT32>>(), tf::DT_INT32);
  EXPECT_TRUE(txeo::detail::get_tf_dtype<txeo::detail::cpp_type<tf::DT_INT64>>() == tf::DT_INT64);
  EXPECT_EQ(txeo::detail::get_tf_dtype<txeo::detail::cpp_type<tf::DT_FLOAT>>(), tf::DT_FLOAT);
  EXPECT_EQ(txeo::detail::get_tf_dtype<txeo::detail::cpp_type<tf::DT_DOUBLE>>(), tf::DT_DOUBLE);
  EXPECT_EQ(txeo::detail::get_tf_dtype<txeo::detail::cpp_type<tf::DT_BOOL>>(), tf::DT_BOOL);
}

TEST(UtilsTest, ToInt64Valid) {
  size_t val = 42;
  EXPECT_EQ(txeo::detail::to_int64(val), 42);
}

TEST(UtilsTest, ToInt64Overflow) {
  size_t val = static_cast<size_t>(std::numeric_limits<int64_t>::max()) + 1;
  EXPECT_THROW(txeo::detail::to_int64(val), std::overflow_error);
}

TEST(UtilsTest, ToSizeTValid) {
  int64_t val = 42;
  EXPECT_EQ(txeo::detail::to_size_t(val), 42);
}

TEST(UtilsTest, ToSizeTNegative) {
  int64_t val = -1;
  EXPECT_THROW(txeo::detail::to_size_t(val), std::overflow_error);
}

TEST(UtilsTest, ToSizeTVector) {
  std::vector<int64_t> input = {1, 2, 3};
  std::vector<size_t> expected = {1, 2, 3};
  EXPECT_EQ(txeo::detail::to_size_t(input), expected);
}

TEST(UtilsTest, ToInt64Vector) {
  std::vector<size_t> input = {1, 2, 3};
  std::vector<int64_t> expected = {1, 2, 3};
  EXPECT_EQ(txeo::detail::to_int64(input), expected);
}

TEST(UtilsTest, ToIntValid) {
  size_t val = 42;
  EXPECT_EQ(txeo::detail::to_int(val), 42);
}

TEST(UtilsTest, ToIntOverflow) {
  size_t val = static_cast<size_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_THROW(txeo::detail::to_int(val), std::overflow_error);
}

TEST(UtilsTest, ToInt64ToIntValid) {
  int64_t val = 42;
  EXPECT_EQ(txeo::detail::to_int(val), 42);
}

TEST(UtilsTest, ToInt64ToIntOverflow) {
  int64_t val = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_THROW(txeo::detail::to_int(val), std::overflow_error);
}

TEST(UtilsTest, ToTxeoTensorShape) {
  tf::TensorShape tf_shape({2, 3, 4});
  txeo::TensorShape txeo_shape = txeo::detail::to_txeo_tensor_shape(tf_shape);
  EXPECT_EQ(txeo::detail::to_size_t(txeo_shape.axes_dims()), std::vector<size_t>({2, 3, 4}));
}

TEST(UtilsTest, CalcStride) {
  tf::TensorShape shape({3, 4, 5});
  std::vector<size_t> expected = {20, 5};
  EXPECT_EQ(txeo::detail::calc_stride(shape), expected);
}

TEST(UtilsTest, CalcStrideSingleDim) {
  tf::TensorShape shape({5});
  EXPECT_TRUE(txeo::detail::calc_stride(shape).empty());
}
