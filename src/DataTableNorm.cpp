#include "txeo/DataTableNorm.h"

namespace txeo {

template <typename T>
DataTableNorm<T>::DataTableNorm(const txeo::DataTable<T> &data, txeo::NormalizationType type)
    : _type{type} {
}

} // namespace txeo