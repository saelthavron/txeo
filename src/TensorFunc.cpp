#include "txeo/TensorFunc.h"
#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/TensorOp.h"
#include "txeo/detail/TensorHelper.h"
#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <tensorflow/cc/ops/array_ops.h>
#include <utility>

namespace txeo {

template <typename T>
txeo::Tensor<T> TensorFunc<T>::power_elem(const txeo::Tensor<T> &tensor, const T &exponent) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  txeo::Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = static_cast<T>(std::pow(tensor.data()[i], exponent));

  return resp;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::power_elem_by(txeo::Tensor<T> &tensor, const T &exponent) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] = static_cast<T>(std::pow(tensor.data()[i], exponent));

  return tensor;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::square(const txeo::Tensor<T> &tensor) {
  auto resp = TensorOp<T>::hadamard_prod(tensor, tensor);
  return resp;
}

template <typename T>
txeo::Tensor<T> &TensorFunc<T>::square_by(txeo::Tensor<T> &tensor) {
  TensorOp<T>::hadamard_prod_by(tensor, tensor);
  return tensor;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::sqrt(const txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  txeo::Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = static_cast<T>(std::sqrt(tensor.data()[i]));

  return resp;
}

template <typename T>
txeo::Tensor<T> &TensorFunc<T>::sqrt_by(txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] = static_cast<T>(std::sqrt(tensor.data()[i]));

  return tensor;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::abs(const txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, [](const tf::Scope &scope, tf::Input input) -> tf::Output {
          return tf::ops::Abs(scope, input);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorFuncError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
txeo::Tensor<T> &TensorFunc<T>::abs_by(txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] = static_cast<T>(std::abs(tensor.data()[i]));

  return tensor;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::permute(const txeo::Tensor<T> &tensor,
                                       const std::vector<size_t> &axes) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  if (tensor.order() != txeo::detail::to_int(axes.size()))
    throw txeo::TensorFuncError("Tensor order and number of axes are different.");

  for (auto &item : axes)
    if (item >= txeo::detail::to_size_t(tensor.order()))
      throw txeo::TensorFuncError("Inconsistent axes.");

  tf::Tensor perm(tf::DT_INT64, tf::TensorShape({txeo::detail::to_int64(axes.size())}));
  auto perm_flat = perm.flat<int64_t>();
  for (int64_t i{0}; i < txeo::detail::to_int64(axes.size()); ++i)
    perm_flat(i) = txeo::detail::to_int64((axes[i]));

  return txeo::detail::TensorHelper::ope_tensors<T>(
      *tensor._impl->tf_tensor, perm, [](const tf::Scope &scope, tf::Input left, tf::Input right) {
        return tf::ops::Transpose(scope, left, right);
      });
}

template <typename T>
txeo::Tensor<T> &TensorFunc<T>::permute_by(txeo::Tensor<T> &tensor,
                                           const std::vector<size_t> &axes) {
  tensor = std::move(TensorFunc<T>::permute(tensor, axes));
  return tensor;
}

template <typename T>
txeo::Matrix<T> TensorFunc<T>::transpose(const txeo::Matrix<T> &matrix) {
  return txeo::Matrix(TensorFunc<T>::permute(matrix, {1, 0}));
}

template <typename T>
txeo::Matrix<T> &TensorFunc<T>::transpose_by(txeo::Matrix<T> &matrix) {
  matrix = std::move(TensorFunc<T>::transpose(matrix));
  return matrix;
}

// Type specialization to avoid calling abs for unsigned types
template <>
txeo::Tensor<bool> &TensorFunc<bool>::abs_by(txeo::Tensor<bool> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");
  return tensor;
}

// Type specialization to avoid calling abs for unsigned types
template <>
txeo::Tensor<size_t> &TensorFunc<size_t>::abs_by(txeo::Tensor<size_t> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");
  return tensor;
}

template <typename T>
void TensorFunc<T>::min_max_normalize(const std::vector<T> &values,
                                      const std::vector<T *> &adresses) {
  auto [min_it, max_it] = std::ranges::minmax_element(values);
  auto dif = *max_it - *min_it;
  if (txeo::detail::is_zero(dif))
    return;

  auto min = *min_it;
  for (size_t i{0}; i < adresses.size(); ++i)
    *(adresses[i]) = (*(adresses[i]) - min) / dif;
}

template <typename T>
void TensorFunc<T>::z_score_normalize(const std::vector<T> &values,
                                      const std::vector<T *> &adresses) {
  if (values.size() == 1)
    return;

  T mean = 0.0;
  for (const auto &item : values)
    mean += item;
  mean /= values.size();

  T variance_num = 0.0;
  for (const auto &item : values) {
    auto dif = (item - mean);
    variance_num += dif * dif;
  }
  auto std_dev = std::sqrt(variance_num / (values.size() - 1.));

  if (txeo::detail::is_zero(std_dev))
    return;

  for (size_t i{0}; i < adresses.size(); ++i)
    *(adresses[i]) = (*(adresses[i]) - mean) / std_dev;
}

template <typename T>
void TensorFunc<T>::axis_func(
    txeo::Tensor<T> &tensor, size_t axis,
    std::function<void(const std::vector<T> &, const std::vector<T *> &)> func) {

  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  if (axis >= txeo::detail::to_size_t(tensor.order()))
    throw txeo::TensorFuncError("Inconsistent axis.");

  int64_t accum_step = 1;
  for (size_t i{axis + 1}; i < txeo::detail::to_size_t(tensor.order()); ++i)
    accum_step *= tensor.shape().axis_dim(i);
  auto axis_dim = txeo::detail::to_size_t(tensor.shape().axis_dim(axis));

  size_t p{0}, s{0};
  auto data = tensor.data();
  std::priority_queue<size_t, std::vector<size_t>, std::greater<>> accum_indexes;
  std::vector<T> values;
  std::vector<T *> addresses;

  while (p < tensor.dim()) {
    s = p;
    values.emplace_back(data[p]);
    addresses.push_back(data + p);
    for (size_t i{0}; i < axis_dim - 1; ++i) {
      s += accum_step;
      accum_indexes.emplace(s);
      values.emplace_back(data[s]);
      addresses.push_back(data + s);
    }
    func(values, addresses);
    values.clear();
    addresses.clear();
    ++p;
    while (!accum_indexes.empty() && p == accum_indexes.top()) {
      accum_indexes.pop();
      ++p;
    }
  }
}

template <typename T>
txeo::Tensor<T> &TensorFunc<T>::normalize_by(txeo::Tensor<T> &tensor, size_t axis,
                                             txeo::NormalizationType type) {
  if (type == txeo::NormalizationType::MIN_MAX)
    axis_func(tensor, axis,
              [](const std::vector<T> &values, const std::vector<T *> &addresses) -> void {
                min_max_normalize(values, addresses);
              });
  else
    axis_func(tensor, axis,
              [](const std::vector<T> &values, const std::vector<T *> &addresses) -> void {
                z_score_normalize(values, addresses);
              });

  return tensor;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::normalize(const txeo::Tensor<T> &tensor, size_t axis,
                                         txeo::NormalizationType type) {
  txeo::Tensor<T> resp{tensor};
  TensorFunc<T>::normalize_by(resp, axis, type);

  return resp;
}

template <typename T>
void TensorFunc<T>::min_max_normalize(txeo::Tensor<T> &tensor) {
  auto [min_it, max_it] = std::ranges::minmax_element(tensor);
  auto dif = *max_it - *min_it;

  if (txeo::detail::is_zero(dif))
    return;

  auto min = *min_it;
  for (auto &element : tensor)
    element = (element - min) / dif;
}

template <typename T>
void TensorFunc<T>::z_score_normalize(txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 1)
    return;

  T mean = 0.0;
  for (auto &element : tensor)
    mean += element;
  mean /= tensor.dim();

  T variance_num = 0.0;
  for (const auto &element : tensor) {
    auto dif = (element - mean);
    variance_num += dif * dif;
  }
  auto std_dev = std::sqrt(variance_num / (tensor.dim() - 1.));
  if (txeo::detail::is_zero(std_dev))
    return;

  for (auto &element : tensor)
    element = (element - mean) / std_dev;
}

template <typename T>
txeo::Tensor<T> &TensorFunc<T>::normalize_by(txeo::Tensor<T> &tensor,
                                             txeo::NormalizationType type) {

  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  if (type == txeo::NormalizationType::MIN_MAX)
    min_max_normalize(tensor);
  else
    z_score_normalize(tensor);

  return tensor;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::normalize(const txeo::Tensor<T> &tensor,
                                         txeo::NormalizationType type) {
  txeo::Tensor<T> resp{tensor};
  TensorFunc<T>::normalize_by(resp, type);

  return resp;
}

template <typename T>
txeo::Matrix<T> TensorFunc<T>::get_gram_matrix(const txeo::Matrix<T> &matrix) {
  auto resp = TensorFunc<T>::transpose(matrix);

  return txeo::TensorOp<T>::product(resp, matrix);
}

template class TensorFunc<size_t>;
template class TensorFunc<short>;
template class TensorFunc<int>;
template class TensorFunc<bool>;
template class TensorFunc<long>;
template class TensorFunc<long long>;
template class TensorFunc<float>;
template class TensorFunc<double>;

} // namespace txeo