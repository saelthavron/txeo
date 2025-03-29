#include "txeo/detail/utils.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/tensor_shape.pb.h>

namespace tf = tensorflow;

namespace txeo::detail {

int64_t to_int64(const size_t &val) {
  if (val > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    throw std::overflow_error("size_t value exceeds int64_t maximum");

  return static_cast<int64_t>(val);
}

size_t to_size_t(const int64_t &val) {
  if (val < 0)
    throw std::overflow_error("int64_t has a negative value");

  return static_cast<size_t>(val);
}

std::vector<size_t> to_size_t(const std::vector<int64_t> &vec) {
  std::vector<size_t> aux;
  for (auto &item : vec)
    aux.emplace_back(txeo::detail::to_size_t(item));
  return aux;
}

std::vector<int64_t> to_int64(const std::vector<size_t> &vec) {
  std::vector<int64_t> aux;
  for (auto &item : vec)
    aux.emplace_back(txeo::detail::to_int64(item));
  return aux;
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

std::string format(const double &a, int precision) {
  std::stringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(precision + 1);
  stream << a;

  return stream.str();
}

txeo::TensorShape to_txeo_tensor_shape(const tf::TensorShape &shape) {
  std::vector<size_t> aux;
  auto dim_sizes = shape.dim_sizes();
  std::ranges::copy(dim_sizes, std::back_inserter(aux));

  txeo::TensorShape res(aux);

  return res;
}

txeo::TensorShape proto_to_txeo_tensor_shape(const tf::TensorShapeProto &shape) {
  std::vector<size_t> aux;
  for (const auto &item : shape.dim()) {
    if (item.size() < 0)
      aux.emplace_back(0);
    else
      aux.emplace_back(static_cast<size_t>(item.size()));
  }

  txeo::TensorShape res(aux);

  return res;
}

std::vector<size_t> calc_stride(const tf::TensorShape &shape) {
  if (shape.dims() <= 1)
    return std::vector<size_t>{};
  std::vector<size_t> resp(shape.dims() - 1);
  size_t accum_sizes{1};
  for (size_t i = shape.dims() - 1; i > 0; --i) {
    accum_sizes *= shape.dim_size(txeo::detail::to_int(i));
    resp[i - 1] = accum_sizes;
  }
  return resp;
}

bool is_numeric(const std::string &word) {
  std::istringstream word_stream{word};
  double val{};
  char c{};

  return (word_stream >> val) && !(word_stream >> c);
}

} // namespace txeo::detail
