#include "txeo/DataTable.h"

#include <utility>

namespace txeo {

template <typename T>
DataTable<T>::DataTable(const txeo::Matrix<T> &data, std::vector<size_t> x_cols,
                        std::vector<size_t> y_cols, size_t eval_percent, size_t test_percent) {
  if (data.dim() == 0)
    throw DataTableError("Tensor has zero dimension.");

  if (eval_percent >= 100 || eval_percent == 0)
    throw DataTableError("Inconsistent evaluation percentage.");

  if (test_percent >= 100 || test_percent == 0)
    throw DataTableError("Inconsistent test percentage.");

  size_t eval_size = (static_cast<double>(eval_percent) / 100.0) * data.shape().axis_dim(0);
  if (eval_size == 0)
    throw DataTableError("Inconsistent evaluation percentage.");

  size_t test_size = (static_cast<double>(test_percent) / 100.0) * data.shape().axis_dim(0);
  if (test_size == 0)
    throw DataTableError("Inconsistent test percentage.");

  if (eval_size + test_size >= data.shape().axis_dim(0))
    throw DataTableError("Inconsistent combination of test and eval percentages.");

  size_t train_size = data.shape().axis_dim(0) - eval_size - test_size;

  Matrix<T> train{data.slice(0, train_size)};
  Matrix<T> eval{data.slice(train_size, train_size + eval_size)};
  Matrix<T> test{data.slice(train_size + eval_size, train_size + eval_size + test_size)};

  _x_train = std::move(TensorPart<T>::sub_matrix_cols(train, x_cols));
  _y_train = std::move(TensorPart<T>::sub_matrix_cols(train, y_cols));
  _x_eval = std::move(TensorPart<T>::sub_matrix_cols(eval, x_cols));
  _y_eval = std::move(TensorPart<T>::sub_matrix_cols(eval, y_cols));
  _x_test = std::move(TensorPart<T>::sub_matrix_cols(test, x_cols));
  _y_test = std::move(TensorPart<T>::sub_matrix_cols(test, y_cols));
  _has_test = _has_eval = true;
}

template <typename T>
DataTable<T>::DataTable(const txeo::Matrix<T> &data, std::vector<size_t> y_cols,
                        size_t eval_percent, size_t test_percent) {

  if (data.dim() == 0)
    throw DataTableError("Tensor has zero dimension.");

  if (eval_percent >= 100 || eval_percent == 0)
    throw DataTableError("Inconsistent evaluation percentage.");

  if (test_percent >= 100 || test_percent == 0)
    throw DataTableError("Inconsistent test percentage.");

  size_t eval_size = (static_cast<double>(eval_percent) / 100.0) * data.shape().axis_dim(0);
  if (eval_size == 0)
    throw DataTableError("Inconsistent evaluation percentage.");

  size_t test_size = (static_cast<double>(test_percent) / 100.0) * data.shape().axis_dim(0);
  if (test_size == 0)
    throw DataTableError("Inconsistent test percentage.");

  if (eval_size + test_size >= data.shape().axis_dim(0))
    throw DataTableError("Inconsistent combination of test and eval percentages.");

  size_t train_size = data.shape().axis_dim(0) - eval_size - test_size;

  Matrix<T> train{data.slice(0, train_size)};
  Matrix<T> eval{data.slice(train_size, train_size + eval_size)};
  Matrix<T> test{data.slice(train_size + eval_size, train_size + eval_size + test_size)};

  _x_train = std::move(TensorPart<T>::sub_matrix_cols_exclude(train, y_cols));
  _y_train = std::move(TensorPart<T>::sub_matrix_cols(train, y_cols));
  _x_eval = std::move(TensorPart<T>::sub_matrix_cols_exclude(eval, y_cols));
  _y_eval = std::move(TensorPart<T>::sub_matrix_cols(eval, y_cols));
  _x_test = std::move(TensorPart<T>::sub_matrix_cols_exclude(test, y_cols));
  _y_test = std::move(TensorPart<T>::sub_matrix_cols(test, y_cols));
  _has_test = _has_eval = true;
}

template <typename T>
DataTable<T>::DataTable(const txeo::Matrix<T> &data, std::vector<size_t> x_cols,
                        std::vector<size_t> y_cols, size_t eval_percent) {

  if (data.dim() == 0)
    throw DataTableError("Tensor has zero dimension.");

  if (eval_percent >= 100 || eval_percent == 0)
    throw DataTableError("Inconsistent evaluation percentage.");

  size_t eval_size = (static_cast<double>(eval_percent) / 100.0) * data.shape().axis_dim(0);
  if (eval_size == 0)
    throw DataTableError("Inconsistent evaluation percentage.");

  size_t train_size = data.shape().axis_dim(0) - eval_size;

  Matrix<T> train{data.slice(0, train_size)};
  Matrix<T> eval{data.slice(train_size, train_size + eval_size)};

  _x_train = std::move(TensorPart<T>::sub_matrix_cols(train, x_cols));
  _y_train = std::move(TensorPart<T>::sub_matrix_cols(train, y_cols));
  _x_eval = std::move(TensorPart<T>::sub_matrix_cols(eval, x_cols));
  _y_eval = std::move(TensorPart<T>::sub_matrix_cols(eval, y_cols));
  _has_eval = true;
  _has_test = false;
}

template <typename T>
DataTable<T>::DataTable(const txeo::Matrix<T> &data, std::vector<size_t> y_cols,
                        size_t eval_percent) {
  if (data.dim() == 0)
    throw DataTableError("Tensor has zero dimension.");

  if (eval_percent >= 100 || eval_percent == 0)
    throw DataTableError("Inconsistent evaluation percentage.");

  size_t eval_size = (static_cast<double>(eval_percent) / 100.0) * data.shape().axis_dim(0);
  if (eval_size == 0)
    throw DataTableError("Inconsistent evaluation percentage.");

  size_t train_size = data.shape().axis_dim(0) - eval_size;

  Matrix<T> train{data.slice(0, train_size)};
  Matrix<T> eval{data.slice(train_size, train_size + eval_size)};

  _x_train = std::move(TensorPart<T>::sub_matrix_cols_exclude(train, y_cols));
  _y_train = std::move(TensorPart<T>::sub_matrix_cols(train, y_cols));
  _x_eval = std::move(TensorPart<T>::sub_matrix_cols_exclude(eval, y_cols));
  _y_eval = std::move(TensorPart<T>::sub_matrix_cols(eval, y_cols));

  _has_eval = true;
  _has_test = false;
}

template <typename T>
DataTable<T>::DataTable(const txeo::Matrix<T> &data, std::vector<size_t> x_cols,
                        std::vector<size_t> y_cols) {

  if (data.dim() == 0)
    throw DataTableError("Tensor has zero dimension.");

  _x_train = std::move(TensorPart<T>::sub_matrix_cols(data, x_cols));
  _y_train = std::move(TensorPart<T>::sub_matrix_cols(data, y_cols));
  _has_eval = _has_test = false;
}

template <typename T>
DataTable<T>::DataTable(const txeo::Matrix<T> &data, std::vector<size_t> y_cols) {

  if (data.dim() == 0)
    throw DataTableError("Tensor has zero dimension.");

  _x_train = std::move(TensorPart<T>::sub_matrix_cols_exclude(data, y_cols));
  _y_train = std::move(TensorPart<T>::sub_matrix_cols(data, y_cols));
  _has_eval = _has_test = false;
}

template <typename T>
std::optional<const txeo::Matrix<T> &> DataTable<T>::x_eval() const {
  return _has_eval ? _x_eval : std::nullopt;
}

template <typename T>
std::optional<const txeo::Matrix<T> &> DataTable<T>::y_eval() const {
  return _has_eval ? _y_eval : std::nullopt;
}

template <typename T>
std::optional<const txeo::Matrix<T> &> DataTable<T>::x_test() const {
  return _has_test ? _x_test : std::nullopt;
}

template <typename T>
std::optional<const txeo::Matrix<T> &> DataTable<T>::y_test() const {
  return _has_test ? _y_test : std::nullopt;
}

} // namespace txeo