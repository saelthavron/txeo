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

inline void check_indexes(const auto &txeo_shape, std::vector<size_t> indexes) {
  for (size_t i{0}; i < indexes.size(); ++i) {
    if (txeo_shape.axis_dim(i) >= (int64_t)indexes[i])
      throw TensorError("Axis " + std::to_string(i) + " not in the range [0," +
                        std::to_string(txeo_shape.axis_dim(i) - 1) + "]");
  }
}

size_t calc_flat_index(const std::vector<size_t> &indexes, const tf::TensorShape *sizes) {
  size_t accum_sizes{1};
  size_t resp{indexes.back()};

  const size_t *idx_ptr = indexes.data();

  for (size_t i = indexes.size() - 1; i > 0; --i) {
    accum_sizes *= sizes->dim_size(i);
    resp += idx_ptr[i - 1] * accum_sizes;
  }

  return resp;
}

} // namespace tensor
} // namespace txeo::detail

#endif