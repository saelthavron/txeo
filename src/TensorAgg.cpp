#include "txeo/TensorAgg.h"
#include "txeo/Tensor.h"
#include "txeo/TensorFunc.h"
#include "txeo/detail/TensorHelper.h"
#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <queue>
#include <stdexcept>
#include <string>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

namespace txeo {

namespace tf = tensorflow;

template <typename T>
void TensorAgg<T>::verify_parameters(const Tensor<T> &tensor, const std::vector<size_t> &axes) {
  if (tensor.dim() == 0)
    throw TensorAggError("Tensor has dimension zero.");

  for (auto &item : axes)
    if (item >= detail::to_size_t(tensor.order()))
      throw TensorAggError("Inconsistent axes.");
}

template <typename T>
Tensor<T> TensorAgg<T>::accumulate(const Tensor<T> &tensor, size_t axis,
                                   std::function<T(std::vector<T> &)> acc_fun) {

  if (tensor.dim() == 0)
    throw TensorAggError("Tensor has dimension zero.");

  if (axis >= detail::to_size_t(tensor.order()))
    throw TensorAggError("Inconsistent axis.");

  int64_t accum_step = 1;
  for (size_t i{axis + 1}; i < detail::to_size_t(tensor.order()); ++i)
    accum_step *= tensor.shape().axis_dim(i);
  auto axis_dim = detail::to_size_t(tensor.shape().axis_dim(axis));

  size_t p{0}, s{0};
  auto data = tensor.data();
  std::priority_queue<size_t, std::vector<size_t>, std::greater<>> accum_indexes;
  std::vector<T> values;
  std::vector<T> vec_resp;

  while (p < tensor.dim()) {
    s = p;
    values.emplace_back(data[p]);
    for (size_t i{0}; i < axis_dim - 1; ++i) {
      s += accum_step;
      accum_indexes.emplace(s);
      values.emplace_back(data[s]);
    }
    vec_resp.emplace_back(acc_fun(values));
    values.clear();
    ++p;
    while (!accum_indexes.empty() && p == accum_indexes.top()) {
      accum_indexes.pop();
      ++p;
    }
  }

  auto dims = tensor.shape().axes_dims();
  std::vector<size_t> shape_resp;

  for (size_t i{0}; i < detail::to_size_t(tensor.order()); ++i)
    if (i != axis)
      shape_resp.emplace_back(dims[i]);

  Tensor<T> resp(shape_resp, vec_resp);

  return resp;
}

template <typename T>
Tensor<size_t> TensorAgg<T>::count(const Tensor<T> &tensor, size_t axis,
                                   std::function<size_t(std::vector<T> &)> count_fun) {

  if (tensor.dim() == 0)
    throw TensorAggError("Tensor has dimension zero.");

  if (axis >= detail::to_size_t(tensor.order()))
    throw TensorAggError("Inconsistent axis.");

  int64_t accum_step = 1;
  for (size_t i{axis + 1}; i < detail::to_size_t(tensor.order()); ++i)
    accum_step *= tensor.shape().axis_dim(i);
  auto axis_dim = detail::to_size_t(tensor.shape().axis_dim(axis));

  size_t p{0}, s{0};
  auto data = tensor.data();
  std::priority_queue<size_t, std::vector<size_t>, std::greater<>> accum_indexes;
  std::vector<T> values;
  std::vector<size_t> vec_resp;

  while (p < tensor.dim()) {
    s = p;
    values.emplace_back(data[p]);
    for (size_t i{0}; i < axis_dim - 1; ++i) {
      s += accum_step;
      accum_indexes.emplace(s);
      values.emplace_back(data[s]);
    }
    vec_resp.emplace_back(count_fun(values));
    values.clear();
    ++p;
    while (!accum_indexes.empty() && p == accum_indexes.top()) {
      accum_indexes.pop();
      ++p;
    }
  }

  auto dims = tensor.shape().axes_dims();
  std::vector<size_t> shape_resp;

  for (size_t i{0}; i < detail::to_size_t(tensor.order()); ++i)
    if (i != axis)
      shape_resp.emplace_back(dims[i]);

  Tensor<size_t> resp(shape_resp, vec_resp);

  return resp;
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_sum(const Tensor<T> &tensor, const std::vector<size_t> &axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::ReduceSum(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_prod(const Tensor<T> &tensor, const std::vector<size_t> &axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::ReduceProd(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_mean(const Tensor<T> &tensor, const std::vector<size_t> &axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::Mean(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_max(const Tensor<T> &tensor, const std::vector<size_t> &axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::Max(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_min(const Tensor<T> &tensor, const std::vector<size_t> &axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::Min(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_euclidean_norm(const Tensor<T> &tensor,
                                              const std::vector<size_t> &axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::EuclideanNorm(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_maximum_norm(const Tensor<T> &tensor, size_t axis) {
  return TensorAgg<T>::accumulate(
      tensor, axis, [](std::vector<T> &values) { return TensorAgg<T>::maximum_norm(values); });
}

template <typename T>
Tensor<T> TensorAgg<T>::cumulative_prod(const Tensor<T> &tensor, size_t axis) {
  if (tensor.dim() == 0)
    throw TensorAggError("Tensor has dimension zero.");

  if (axis >= detail::to_size_t(tensor.order()))
    throw TensorAggError("Inconsistent axis.");

  try {
    auto resp = detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axis,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::Cumprod(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<T> TensorAgg<T>::cumulative_sum(const Tensor<T> &tensor, size_t axis) {
  try {
    auto resp = detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axis,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::Cumsum(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<size_t> TensorAgg<T>::arg_max(const Tensor<T> &tensor, size_t axis) {
  if (tensor.dim() == 0)
    throw TensorAggError("Tensor has dimension zero.");

  if (axis >= detail::to_size_t(tensor.order()))
    throw TensorAggError("Inconsistent axis.");

  try {
    auto resp = detail::TensorHelper::reduce_tensor_to_indexes<T>(
        *tensor._impl->tf_tensor, detail::to_int64(axis),
        [](const tf::Scope &scope, tf::Input input, int64_t axis) -> tf::Output {
          return tf::ops::ArgMax(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<size_t> TensorAgg<T>::arg_min(const Tensor<T> &tensor, size_t axis) {
  if (tensor.dim() == 0)
    throw TensorAggError("Tensor has dimension zero.");

  if (axis >= detail::to_size_t(tensor.order()))
    throw TensorAggError("Inconsistent axis.");

  try {
    auto resp = detail::TensorHelper::reduce_tensor_to_indexes<T>(
        *tensor._impl->tf_tensor, detail::to_int64(axis),
        [](const tf::Scope &scope, tf::Input input, int64_t axis) -> tf::Output {
          return tf::ops::ArgMin(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<size_t> TensorAgg<T>::count_non_zero(const Tensor<T> &tensor, size_t axis) {

  return TensorAgg<T>::count(tensor, axis, [](std::vector<T> &values) -> size_t {
    return TensorAgg<T>::count_non_zero(values);
  });
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_all(const Tensor<T> &tensor, const std::vector<size_t> &axes)
  requires(std::convertible_to<T, bool>)
{
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::ReduceAll(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_any(const Tensor<T> &tensor, const std::vector<size_t> &axes)
  requires(std::convertible_to<T, bool>)
{
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::ReduceAny(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_variance(const Tensor<T> &tensor, size_t axis) {
  return TensorAgg<T>::accumulate(
      tensor, axis, [](std::vector<T> &values) -> T { return TensorAgg<T>::variance(values); });
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_standard_deviation(const Tensor<T> &tensor, size_t axis) {
  auto aux = TensorAgg<T>::reduce_variance(tensor, axis);
  return TensorFunc<T>::sqrt_by(aux);
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_median(const Tensor<T> &tensor, size_t axis) {
  return TensorAgg<T>::accumulate(
      tensor, axis, [](std::vector<T> &values) -> T { return TensorAgg<T>::median(values); });
}

template <typename T>
Tensor<T> TensorAgg<T>::reduce_geometric_mean(const Tensor<T> &tensor, size_t axis) {
  return TensorAgg<T>::accumulate(tensor, axis, [](std::vector<T> &values) -> T {
    return TensorAgg<T>::geometric_mean(values);
  });
}

template <typename T>
T TensorAgg<T>::sum_all(const Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw TensorAggError("Tensor has dimension zero.");

  T resp = 0.0;
  for (size_t i{0}; i < tensor.dim(); ++i)
    resp += tensor.data()[i];

  return resp;
}

template <typename T>
T TensorAgg<T>::median(std::vector<T> &values) {
  if (values.size() == 1)
    return values[0];

  std::sort(std::begin(values), std::end(values));

  auto index = values.size() / 2;
  if (values.size() % 2 != 0)
    return values[index];

  return (values[index] + values[index - 1]) / 2;
}

template <typename T>
T TensorAgg<T>::geometric_mean(std::vector<T> &values) {
  if (values.size() == 1)
    return values[0];

  T mean = 1.0;
  for (const auto &item : values)
    mean *= item;

  return std::pow(mean, 1.0 / values.size());
}

template <typename T>
T TensorAgg<T>::variance(std::vector<T> &values) {
  if (values.size() == 1)
    return values[0];

  T mean = 0.0;
  for (const auto &item : values)
    mean += item;
  mean /= values.size();

  T resp = 0.0;
  for (const auto &item : values) {
    auto dif = (item - mean);
    resp += dif * dif;
  }

  return resp / (values.size() - 1.);
}

template <typename T>
T TensorAgg<T>::maximum_norm(std::vector<T> &values) {
  auto resp = std::ranges::max_element(
      values, [](const T &left, const T &right) { return std::abs(left) < std::abs(right); });

  return *resp;
}

// Type specialization to avoid calling abs for unsigned types
template <>
bool TensorAgg<bool>::maximum_norm(std::vector<bool> &values) {
  auto resp = std::ranges::max_element(values);

  return *resp;
}

// Type specialization to avoid calling abs for unsigned types
template <>
size_t TensorAgg<size_t>::maximum_norm(std::vector<size_t> &values) {
  auto resp = std::ranges::max_element(values);
  return *resp;
}

template <typename T>
size_t TensorAgg<T>::count_non_zero(std::vector<T> &values) {
  return std::ranges::count_if(values,
                               [](const T &item) -> bool { return !detail::is_zero(item); });
}

template class TensorAgg<short>;
template class TensorAgg<int>;
template class TensorAgg<bool>;
template class TensorAgg<long>;
template class TensorAgg<long long>;
template class TensorAgg<float>;
template class TensorAgg<double>;
template class TensorAgg<size_t>;

} // namespace txeo