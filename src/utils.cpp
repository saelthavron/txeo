#include "txeo/detail/utils.h"

#include <cstddef>
#include <cstdint>
#include <tensorflow/core/framework/tensor_shape.h>

namespace tf = tensorflow;

namespace txeo::detail {

int64_t to_int64(const size_t &val) {
  if (val > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    throw std::overflow_error("size_t value exceeds int64_t maximum");

  return static_cast<int64_t>(val);
}

int to_int(const size_t &val) {
  if (val > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("size_t value exceeds int maximum");

  return static_cast<int>(val);
}

int to_int(const int64_t &val) {
  if (val > static_cast<int64_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("int64_t value exceeds int maximum");

  return static_cast<int>(val);
}

txeo::TensorShape to_txeo_tensor_shape(const tf::TensorShape &shape) {
  std::vector<int64_t> aux;
  auto dim_sizes = shape.dim_sizes();
  std::ranges::copy(std::begin(dim_sizes), std::end(dim_sizes), std::back_inserter(aux));

  txeo::TensorShape res{aux};

  return res;
}

std::vector<int64_t> calc_stride(const tf::TensorShape &shape) {
  if (shape.dims() <= 1)
    return std::vector<int64_t>{};
  std::vector<int64_t> resp(shape.dims() - 1);
  int64_t accum_sizes{1};
  for (int64_t i = shape.dims() - 1; i > 0; --i) {
    accum_sizes *= shape.dim_size(txeo::detail::to_int(i));
    resp[i - 1] = accum_sizes;
  }
  return resp;
}

} // namespace txeo::detail
