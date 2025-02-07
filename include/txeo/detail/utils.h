#ifndef TXEO_UTILS_H
#define TXEO_UTILS_H

#pragma once

#include <cstddef>

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

inline int64_t to_int64(const size_t &val) {
  if (val > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    throw std::overflow_error("size_t value exceeds int64_t maximum");

  return static_cast<int64_t>(val);
}

inline int to_int(const size_t &val) {
  if (val > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("size_t value exceeds int maximum");

  return static_cast<int>(val);
}

namespace tensor {

inline void check_indexes(const auto &txeo_shape, std::vector<size_t> indexes) {
  for (size_t i{0}; i < indexes.size(); ++i) {
    if (txeo_shape.axis_dim(i) >= txeo::detail::to_int64(indexes[i]))
      throw TensorError("Axis " + std::to_string(i) + " not in the range [0," +
                        std::to_string(txeo_shape.axis_dim(i) - 1) + "]");
  }
}

size_t calc_flat_index(const std::vector<size_t> &indexes, const tf::TensorShape *sizes) {
  size_t accum_sizes{1};
  size_t resp{indexes.back()};

  const size_t *idx_ptr = indexes.data();

  for (size_t i = indexes.size() - 1; i > 0; --i) {
    accum_sizes *= sizes->dim_size(txeo::detail::to_int(i));
    resp += idx_ptr[i - 1] * accum_sizes;
  }

  return resp;
}

size_t calc_flat_index2(const size_t *indexes, const size_t &size, const tf::TensorShape *shape) {
  size_t accum_sizes{1};
  size_t resp{indexes[size - 1]};

  for (size_t i = size - 1; i > 0; --i) {
    accum_sizes *= shape->dim_size(txeo::detail::to_int(i));
    resp += indexes[i - 1] * accum_sizes;
  }

  return resp;
}

} // namespace tensor
} // namespace txeo::detail

#endif