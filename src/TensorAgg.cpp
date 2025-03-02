#include "txeo/TensorAgg.h"
#include "txeo/Tensor.h"
#include "txeo/detail/TensorHelper.h"
#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/utils.h"

#include <cstddef>
#include <cstdint>
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
void TensorAgg<T>::verify_parameters(const txeo::Tensor<T> &tensor, std::vector<size_t> axes) {
  if (tensor.dim() == 0)
    throw txeo::TensorAggError("Tensor has dimension zero.");

  for (auto &item : axes)
    if (item >= txeo::detail::to_size_t(tensor.order()))
      throw txeo::TensorAggError("Inconsistent axes.");
}

template <typename T>
txeo::Tensor<T> TensorAgg<T>::reduce_sum(const txeo::Tensor<T> &tensor, std::vector<size_t> axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::ReduceSum(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
txeo::Tensor<T> TensorAgg<T>::reduce_prod(const txeo::Tensor<T> &tensor, std::vector<size_t> axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::ReduceProd(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
txeo::Tensor<T> TensorAgg<T>::reduce_mean(const txeo::Tensor<T> &tensor, std::vector<size_t> axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::Mean(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
txeo::Tensor<T> TensorAgg<T>::reduce_max(const txeo::Tensor<T> &tensor, std::vector<size_t> axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::Max(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
txeo::Tensor<T> TensorAgg<T>::reduce_min(const txeo::Tensor<T> &tensor, std::vector<size_t> axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::Min(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
inline txeo::Tensor<T> TensorAgg<T>::reduce_euclidean_norm(const txeo::Tensor<T> &tensor,
                                                           std::vector<size_t> axes) {
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::EuclideanNorm(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
txeo::Tensor<T> TensorAgg<T>::cumulative_prod(const txeo::Tensor<T> &tensor, size_t axis) {
  if (tensor.dim() == 0)
    throw txeo::TensorAggError("Tensor has dimension zero.");

  if (axis >= txeo::detail::to_size_t(tensor.order()))
    throw txeo::TensorAggError("Inconsistent axis.");

  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axis,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::Cumprod(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
inline txeo::Tensor<T> TensorAgg<T>::cumulative_sum(const txeo::Tensor<T> &tensor, size_t axis) {
  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axis,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::Cumsum(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<size_t> TensorAgg<T>::arg_max(const txeo::Tensor<T> &tensor, size_t axis) {
  if (tensor.dim() == 0)
    throw txeo::TensorAggError("Tensor has dimension zero.");

  if (axis >= txeo::detail::to_size_t(tensor.order()))
    throw txeo::TensorAggError("Inconsistent axis.");

  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor_to_indexes<T>(
        *tensor._impl->tf_tensor, txeo::detail::to_int64(axis),
        [](const tf::Scope &scope, tf::Input input, int64_t axis) -> tf::Output {
          return tf::ops::ArgMax(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
Tensor<size_t> TensorAgg<T>::arg_min(const txeo::Tensor<T> &tensor, size_t axis) {
  if (tensor.dim() == 0)
    throw txeo::TensorAggError("Tensor has dimension zero.");

  if (axis >= txeo::detail::to_size_t(tensor.order()))
    throw txeo::TensorAggError("Inconsistent axis.");

  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor_to_indexes<T>(
        *tensor._impl->tf_tensor, txeo::detail::to_int64(axis),
        [](const tf::Scope &scope, tf::Input input, int64_t axis) -> tf::Output {
          return tf::ops::ArgMin(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
txeo::Tensor<T> TensorAgg<T>::reduce_all(const txeo::Tensor<T> &tensor, std::vector<size_t> axes)
  requires(std::convertible_to<T, bool>)
{
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::ReduceAll(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
txeo::Tensor<T> TensorAgg<T>::reduce_any(const txeo::Tensor<T> &tensor, std::vector<size_t> axes)
  requires(std::convertible_to<T, bool>)
{
  TensorAgg<T>::verify_parameters(tensor, axes);
  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, axes,
        [](const tf::Scope &scope, tf::Input input, tf::Input axis) -> tf::Output {
          return tf::ops::ReduceAny(scope, input, axis);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
txeo::Tensor<T> TensorAgg<T>::abs(const txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorAggError("Tensor has dimension zero.");

  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, [](const tf::Scope &scope, tf::Input input) -> tf::Output {
          return tf::ops::Abs(scope, input);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorAggError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
T TensorAgg<T>::variance(const txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorAggError("Tensor has dimension zero.");

  std::vector<size_t> axes;
  for (size_t i{0}; i < txeo::detail::to_size_t(tensor.order()); ++i)
    axes.emplace_back(i);

  auto mean = TensorAgg<T>::reduce_mean(tensor, axes);
  txeo::Tensor<T> sq_dif{(tensor - mean()).square()};
  auto mean_sqdif = TensorAgg<T>::reduce_mean(sq_dif, axes);

  return mean_sqdif();
}

template <typename T>
T TensorAgg<T>::standard_deviation(const txeo::Tensor<T> &tensor) {
  return std::sqrt(txeo::TensorAgg<T>::variance(tensor));
}

template <typename T>
inline T TensorAgg<T>::sum_all(const txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorAggError("Tensor has dimension zero.");

  T resp = 0.0;
  for (size_t i{0}; i < tensor.dim(); ++i)
    resp += tensor.data()[i];

  return resp;
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