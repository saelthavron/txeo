#include "txeo/DataTable.h"
#include "txeo/TensorPart.h"

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

  if (eval_size + test_size >= detail::to_size_t(data.shape().axis_dim(0)))
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

  if (eval_size + test_size >= detail::to_size_t(data.shape().axis_dim(0)))
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
DataTable<T>::DataTable(const txeo::Matrix<T> &x_train, const txeo::Matrix<T> &y_train,
                        const txeo::Matrix<T> &x_eval, const txeo::Matrix<T> &y_eval,
                        const txeo::Matrix<T> &x_test, const txeo::Matrix<T> &y_test) {
  if (x_train.dim() == 0 || y_train.dim() == 0 || x_eval.dim() == 0 || y_eval.dim() == 0)
    throw DataTableError("One of the tensors has zero dimension.");

  if (x_train.shape().axis_dim(0) != y_train.shape().axis_dim(0) ||
      x_eval.shape().axis_dim(0) != y_eval.shape().axis_dim(0) ||
      x_test.shape().axis_dim(0) != y_test.shape().axis_dim(0))
    throw DataTableError("Training or Validation or Test pair of tensors are incompatible.");

  _x_train = std::move(x_train);
  _y_train = std::move(y_train);
  _x_eval = std::move(x_eval);
  _y_eval = std::move(y_eval);
  _x_test = std::move(x_test);
  _y_test = std::move(y_test);
  _has_test = _has_eval = true;
}

template <typename T>
DataTable<T>::DataTable(const txeo::Matrix<T> &x_train, const txeo::Matrix<T> &y_train,
                        const txeo::Matrix<T> &x_eval, const txeo::Matrix<T> &y_eval) {
  if (x_train.dim() == 0 || y_train.dim() == 0 || x_eval.dim() == 0 || y_eval.dim() == 0)
    throw DataTableError("One of the tensors has zero dimension.");

  if (x_train.row_size() != y_train.row_size() || x_eval.row_size() != y_eval.row_size())
    throw DataTableError("Training or Validation of tensors are incompatible.");

  _x_train = std::move(x_train);
  _y_train = std::move(y_train);
  _x_eval = std::move(x_eval);
  _y_eval = std::move(y_eval);
  _has_eval = true;
  _has_test = false;
}

template <typename T>
DataTable<T>::DataTable(const txeo::Matrix<T> &x_train, const txeo::Matrix<T> &y_train) {
  if (x_train.dim() == 0 || y_train.dim() == 0)
    throw DataTableError("One of the tensors has zero dimension.");

  if (x_train.row_size() != y_train.row_size())
    throw DataTableError("Training pair of tensors are incompatible.");

  _x_train = std::move(x_train);
  _y_train = std::move(y_train);
  _has_eval = false;
  _has_test = false;
}

template <typename T>
const txeo::Matrix<T> *DataTable<T>::x_eval() const {
  return _has_eval ? &_x_eval : nullptr;
}

template <typename T>
const txeo::Matrix<T> *DataTable<T>::y_eval() const {
  return _has_eval ? &_y_eval : nullptr;
}

template <typename T>
const txeo::Matrix<T> *DataTable<T>::x_test() const {
  return _has_test ? &_x_test : nullptr;
}

template <typename T>
const txeo::Matrix<T> *DataTable<T>::y_test() const {
  return _has_test ? &_y_test : nullptr;
}

template <typename T>
size_t DataTable<T>::row_size() const {
  return _x_train.row_size();
}

template <typename T>
size_t DataTable<T>::x_dim() const {
  return _x_train.col_size();
}

template <typename T>
size_t DataTable<T>::y_dim() const {
  return _y_train.col_size();
}

template class DataTable<size_t>;
template class DataTable<short>;
template class DataTable<int>;
template class DataTable<bool>;
template class DataTable<long>;
template class DataTable<long long>;
template class DataTable<float>;
template class DataTable<double>;

} // namespace txeo