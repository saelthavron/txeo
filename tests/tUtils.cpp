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
