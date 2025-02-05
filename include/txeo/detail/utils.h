#ifndef TXEO_UTILS_H
#define TXEO_UTILS_H
#include <cstddef>
#include <memory>
#pragma once

#include "tensorflow/core/framework/types.h"

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

namespace tensor {

inline void update_shape(auto &tf_tensor, auto &txeo_shape) {
  txeo_shape->remove_all_axes();
  for (auto &item : tf_tensor->shape().dim_sizes())
    txeo_shape->push_axis_back(item);
}

inline void check_indexes(const auto &txeo_shape, std::vector<size_t> indexes) {
  for (size_t i{0}; i < indexes.size(); ++i) {
    if (txeo_shape->axis_dim(i) >= (int64_t)indexes[i])
      throw TensorError("Axis " + std::to_string(i) + " not in the range [0," +
                        std::to_string(txeo_shape->axis_dim(i) - 1) + "]");
  }
}

} // namespace tensor
} // namespace txeo::detail

#endif