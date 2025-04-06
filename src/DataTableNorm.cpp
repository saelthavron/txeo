#include "txeo/DataTableNorm.h"
#include "txeo/TensorFunc.h"

namespace txeo {

template <typename T>
DataTableNorm<T>::DataTableNorm(const txeo::DataTable<T> &data, txeo::NormalizationType type)
    : _type{type}, _data_table(&data),
      _funcs(txeo::TensorFunc<T>::make_normalize_functions(data.x_train(), 0, _type)) {
}

template <typename T>
void DataTableNorm<T>::set_data_table(const txeo::DataTable<T> &data) {
  _data_table = &data;
  _funcs = txeo::TensorFunc<T>::make_normalize_functions(data.x_train(), 0, _type);
}

template <typename T>
txeo::Matrix<T> DataTableNorm<T>::normalize(txeo::Matrix<T> &&x) const {
  if (x.col_size() != _funcs.size())
    throw txeo::DataTableNormError("Inconsistent feature matrix.");

  txeo::Matrix<T> resp{std::move(x)};

  for (size_t j{0}; j < x.col_size(); ++j)
    for (size_t i{0}; i < x.row_size(); ++i)
      resp(i) = _funcs[j](x(i));

  return resp;
}

template <typename T>
txeo::Matrix<T> DataTableNorm<T>::x_train_normalized() {
  return this->normalize(_data_table->x_train());
}

template <typename T>
txeo::Matrix<T> DataTableNorm<T>::x_eval_normalized() {
  if (!_data_table->has_eval())
    throw txeo::DataTableNormError("No evaluation data was defined.");

  return this->normalize(*_data_table->x_eval());
}

template <typename T>
txeo::Matrix<T> DataTableNorm<T>::x_test_normalized() {
  if (!_data_table->has_test())
    throw txeo::DataTableNormError("No test data was defined.");

  return this->normalize(*_data_table->x_test());
}

template class DataTableNorm<size_t>;
template class DataTableNorm<short>;
template class DataTableNorm<int>;
template class DataTableNorm<bool>;
template class DataTableNorm<long>;
template class DataTableNorm<long long>;
template class DataTableNorm<float>;
template class DataTableNorm<double>;

} // namespace txeo