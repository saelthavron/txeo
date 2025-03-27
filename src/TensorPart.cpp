#include "txeo/TensorPart.h"
#include "txeo/TensorShape.h"
#include "txeo/detail/TensorHelper.h"
#include "txeo/detail/utils.h"

#include <algorithm>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

namespace tf = tensorflow;

namespace txeo {

template <typename T>
std::vector<Tensor<T>> TensorPart<T>::unstack(const Tensor<T> &tensor, size_t axis) {
  if (axis >= detail::to_size_t(tensor.order()))
    throw TensorError("Axis inconsistent with the orderof this tensor!");

  auto shp = tensor.shape();

  auto root = tf::Scope::NewRootScope();

  auto aux = tf::ops::Unstack(root, *tensor._impl->tf_tensor, shp.axis_dim(axis),
                              tf::ops::Unstack::Attrs().Axis(detail::to_int64(axis)));

  tf::ClientSession session(root);
  std::vector<tf::Tensor> outputs;
  auto status = session.Run({aux.output}, &outputs);
  if (!status.ok())
    throw TensorPartError("This tensor could not be unstacked: " + status.ToString());

  std::vector<Tensor<T>> resp;
  for (auto &item : outputs)
    resp.emplace_back(detail::TensorHelper::to_txeo_tensor<T>(std::move(item)));

  return resp;
}

template <typename T>
Tensor<T> TensorPart<T>::slice(const Tensor<T> &tensor, size_t first_axis_begin,
                               size_t first_axis_end) {
  if (first_axis_end < first_axis_begin)
    throw TensorError("The end index can not be less than the initial index!");
  if (detail::to_int64(first_axis_end) > tensor._impl->txeo_shape.axis_dim(0))
    throw TensorPartError(
        "The end index can not be greater than or equal to the dimension of first axis!");

  auto t_slice = tensor._impl->tf_tensor->Slice(first_axis_begin, first_axis_end);
  Tensor<T> resp{detail::to_txeo_tensor_shape(t_slice.shape())};
  if (!resp._impl->tf_tensor->CopyFrom(t_slice, t_slice.shape()))
    throw TensorPartError("This tensor could not be sliced!");

  return resp;
}

template <typename T>
Tensor<T> TensorPart<T>::increase_dimension(const Tensor<T> &tensor, size_t axis, T value) {
  if (axis >= detail::to_size_t(tensor.order()))
    throw TensorError("Axis inconsistent with the order of this tensor!");

  TensorShape shp{tensor.shape()};
  auto axis_int = detail::to_int(axis);
  auto old_dim = shp.axis_dim(axis_int);
  shp.set_dim(axis_int, old_dim + 1);

  auto vec_shape = shp.axes_dims();
  int64_t step = 1;
  for (size_t i{axis + 1}; i < vec_shape.size(); ++i)
    step *= vec_shape[i];

  Tensor<T> resp(shp);

  auto tensor_flat = tensor.data();
  auto resp_flat = resp.data();

  auto insert_pos = step * detail::to_size_t(old_dim);
  size_t i{0}, j{0};
  resp_flat[i] = tensor_flat[j];
  auto resp_dim = resp.dim();
  while (++i < resp.dim()) {
    if (++j % insert_pos == 0) {
      if ((i + step) <= resp_dim)
        for (int64_t k{0}; k < step; ++k) {
          resp_flat[i] = value;
          i++;
        }
      if (i >= resp_dim)
        break;
    }
    resp_flat[i] = tensor_flat[j];
  }

  return resp;
}

template <typename T>
Tensor<T> &TensorPart<T>::increase_dimension_by(Tensor<T> &tensor, size_t axis, T value) {
  tensor = std::move(TensorPart<T>::increase_dimension(tensor, axis, value));

  return tensor;
}

template <typename T>
Matrix<T> TensorPart<T>::sub_matrix_cols(const Matrix<T> &matrix, const std::vector<size_t> &cols) {
  if (cols.empty())
    throw MatrixError("Column indexes vector cannot be empty.");
  for (auto &item : cols)
    if (item >= matrix.col_size())
      throw MatrixError("Inconsistent column indexes");

  Matrix<T> resp{matrix.row_size(), cols.size()};
  for (size_t i{0}; i < matrix.row_size(); ++i)
    for (size_t j{0}; j < cols.size(); ++j)
      resp(i, j) = matrix(i, cols[j]);
  return resp;
}

template <typename T>
txeo::Matrix<T> TensorPart<T>::sub_matrix_cols_exclude(const txeo::Matrix<T> &matrix,
                                                       const std::vector<size_t> &cols) {
  if (cols.empty())
    throw MatrixError("Column indexes vector cannot be empty.");
  for (auto &item : cols)
    if (item >= matrix.col_size())
      throw MatrixError("Inconsistent column indexes");

  std::vector<size_t> in_cols;
  for (size_t i{0}; i < matrix.col_size(); ++i) {
    if (std::ranges::find(cols, i) == std::cend(cols))
      in_cols.emplace_back(i);
  }

  return TensorPart<T>::sub_matrix_cols(matrix, in_cols);
}

template <typename T>
Matrix<T> TensorPart<T>::sub_matrix_rows(const Matrix<T> &matrix, const std::vector<size_t> &rows) {
  if (rows.empty())
    throw MatrixError("Row indexes cannot be empty.");
  for (auto &item : rows)
    if (item >= matrix.row_size())
      throw MatrixError("Inconsistent row indexes");

  Matrix<T> resp{rows.size(), matrix.col_size()};
  for (size_t i{0}; i < rows.size(); ++i)
    for (size_t j{0}; j < matrix.col_size(); ++j)
      resp(i, j) = matrix(rows[i], j);

  return resp;
}

template class TensorPart<short>;
template class TensorPart<int>;
template class TensorPart<bool>;
template class TensorPart<long>;
template class TensorPart<long long>;
template class TensorPart<float>;
template class TensorPart<double>;
template class TensorPart<size_t>;

} // namespace txeo