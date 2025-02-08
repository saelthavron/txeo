#ifndef TXEO_UTILS_H
#define TXEO_UTILS_H

#pragma once

#include <cstddef>
#include <cstdint>
#include <tensorflow/core/framework/tensor_shape.h>

#include "tensorflow/core/framework/types.h"
#include "txeo/TensorShape.h"

namespace tf = tensorflow;

namespace txeo::detail {

/**
 * @brief Gets the correspondent TensorFlow type from the cpp type
 *
 * @tparam T cpp type
 * @return tf::DataType TensorFlow type
 */
template <typename T>
constexpr tf::DataType get_tf_dtype() {
  static_assert(tf::DataTypeToEnum<T>::value != tf::DT_INVALID,
                "Unsupported C++ type for TensorFlow mapping");
  return tf::DataTypeToEnum<T>::value;
}

/**
 * @brief Specifies the cpp type from the TensorFlow type
 *
 * @tparam T TensorFlow type
 */
template <tf::DataType T>
using cpp_type = typename tf::EnumToDataType<T>::Type;

int64_t to_int64(const size_t &val);

int to_int(const size_t &val);

int to_int(const int64_t &val);

txeo::TensorShape to_txeo_tensor_shape(const tf::TensorShape &shape);

std::vector<int64_t> calc_stride(const tf::TensorShape &shape);

} // namespace txeo::detail

#endif