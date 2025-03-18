
#include "txeo/TensorShape.h"
#include "txeo/detail/TensorShapePrivate.h"
#include "txeo/detail/utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/errors.h>
#include <utility>
#include <vector>

namespace txeo {

namespace tf = tensorflow;

template <typename P>
void TensorShape::create_from_vector(P &&shape) {
  auto aux = std::forward<P>(shape);

  _impl->tf_shape = std::make_unique<tf::TensorShape>(aux);
  _impl->stride = txeo::detail::calc_stride(*_impl->tf_shape);
}

TensorShape::TensorShape() : _impl{std::make_unique<Impl>()} {
  this->create_from_vector(std::vector<int64_t>{});
}

TensorShape::TensorShape(const std::vector<size_t> &shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_vector(txeo::detail::to_int64(shape));
}

TensorShape::TensorShape(std::vector<size_t> &&shape) : _impl{std::make_unique<Impl>()} {
  auto shp = std::move(shape);
  this->create_from_vector(txeo::detail::to_int64(shp));
}

TensorShape::TensorShape(int number_of_axes, size_t dim) : _impl{std::make_unique<Impl>()} {
  if (number_of_axes < 0)
    throw TensorShapeError("Negative number_of_axes or dimension is not allowed.");
  _impl->tf_shape = std::make_unique<tf::TensorShape>(
      std::vector<int64_t>(number_of_axes, txeo::detail::to_int64(dim)));
  _impl->stride = txeo::detail::calc_stride(*_impl->tf_shape);
}

TensorShape::TensorShape(const TensorShape &shape) : _impl{std::make_unique<Impl>()} {
  _impl->tf_shape = shape._impl->tf_shape != nullptr
                        ? std::make_unique<tf::TensorShape>(*shape._impl->tf_shape)
                        : nullptr;
  _impl->stride = shape._impl->stride;
  _impl->ext_tf_shape = shape._impl->ext_tf_shape;
}

TensorShape::TensorShape(TensorShape &&shape) noexcept : _impl{std::make_unique<Impl>()} {
  if (this != &shape) {
    _impl->tf_shape = std::move(shape._impl->tf_shape);
    _impl->stride = std::move(shape._impl->stride);
    _impl->ext_tf_shape = std::move(shape._impl->ext_tf_shape);
  }
}

TensorShape &TensorShape::operator=(const TensorShape &shape) {
  _impl->tf_shape = shape._impl->tf_shape != nullptr
                        ? std::make_unique<tf::TensorShape>(*shape._impl->tf_shape)
                        : nullptr;
  _impl->stride = shape._impl->stride;
  _impl->ext_tf_shape = shape._impl->ext_tf_shape;

  return *this;
}

TensorShape &TensorShape::operator=(TensorShape &&shape) noexcept {
  if (this != &shape) {
    _impl->tf_shape = std::move(shape._impl->tf_shape);
    _impl->stride = std::move(shape._impl->stride);
    _impl->ext_tf_shape = std::move(shape._impl->ext_tf_shape);
  }

  return *this;
}

// Defined here after "Impl" implementation in order to avoid incompleteness
TensorShape::~TensorShape() = default;

int TensorShape::number_of_axes() const noexcept {
  return _impl->tf_shape != nullptr ? _impl->tf_shape->dims() : _impl->ext_tf_shape->dims();
}

size_t TensorShape::calculate_capacity() const noexcept {
  return _impl->tf_shape != nullptr ? _impl->tf_shape->num_elements()
                                    : _impl->ext_tf_shape->num_elements();
}

TensorShape TensorShape::clone() const {
  TensorShape resp(*this);
  return resp;
}

int64_t TensorShape::axis_dim(int axis) const {
  if (axis < 0 || axis >= this->number_of_axes())
    throw TensorShapeError("Axis " + std::to_string(axis) + " not in the range [0," +
                           std::to_string(this->number_of_axes() - 1) + "]");
  return _impl->tf_shape != nullptr ? _impl->tf_shape->dim_size(axis)
                                    : _impl->ext_tf_shape->dim_size(axis);
}

const std::vector<size_t> &TensorShape::stride() const {
  return _impl->stride;
}

std::vector<int64_t> TensorShape::axes_dims() const noexcept {
  const auto &aux =
      _impl->tf_shape != nullptr ? _impl->tf_shape->dim_sizes() : _impl->ext_tf_shape->dim_sizes();
  std::vector<int64_t> res;
  std::ranges::copy(aux, std::back_inserter(res));

  return res;
}

bool TensorShape::is_fully_defined() const noexcept {
  return _impl->tf_shape != nullptr ? _impl->tf_shape->IsFullyDefined()
                                    : _impl->ext_tf_shape->IsFullyDefined();
}

void TensorShape::push_axis_back(size_t dim) {
  _impl->tf_shape->AddDim(txeo::detail::to_int64(dim));
  _impl->stride = txeo::detail::calc_stride(*_impl->tf_shape);
}

void TensorShape::insert_axis(int axis, size_t dim) {
  if (axis < 0 || axis >= this->number_of_axes())
    throw TensorShapeError("Axis " + std::to_string(axis) + " not in the range [0," +
                           std::to_string(this->number_of_axes() - 1) + "]");

  _impl->tf_shape->InsertDim(axis, txeo::detail::to_int64(dim));
  _impl->stride = txeo::detail::calc_stride(*_impl->tf_shape);
}

void TensorShape::remove_axis(int axis) {
  if (axis < 0 || axis >= this->number_of_axes())
    throw TensorShapeError("Axis " + std::to_string(axis) + " not in the range [0," +
                           std::to_string(this->number_of_axes() - 1) + "]");

  _impl->tf_shape->RemoveDim(axis);
  _impl->stride = txeo::detail::calc_stride(*_impl->tf_shape);
}

void TensorShape::remove_all_axes() {
  _impl->tf_shape->Clear();
}

void TensorShape::set_dim(int axis, size_t dim) {
  if (axis < 0 || axis >= this->number_of_axes())
    throw TensorShapeError("Axis " + std::to_string(axis) + " not in the range [0," +
                           std::to_string(this->number_of_axes() - 1) + "]");

  _impl->tf_shape->set_dim(axis, txeo::detail::to_int64(dim));
  _impl->stride = txeo::detail::calc_stride(*_impl->tf_shape);
}

bool TensorShape::operator==(const TensorShape &shape) const {
  if (_impl->tf_shape != nullptr) {
    if (shape._impl->tf_shape != nullptr)
      return *this->_impl->tf_shape == *shape._impl->tf_shape;
    return *this->_impl->tf_shape == *shape._impl->ext_tf_shape;
  }
  if (shape._impl->tf_shape != nullptr)
    return *this->_impl->ext_tf_shape == *shape._impl->tf_shape;
  return *this->_impl->ext_tf_shape == *shape._impl->ext_tf_shape;
}

bool TensorShape::operator!=(const TensorShape &shape) const {
  if (_impl->tf_shape != nullptr) {
    if (shape._impl->tf_shape != nullptr)
      return *this->_impl->tf_shape != *shape._impl->tf_shape;
    return *this->_impl->tf_shape != *shape._impl->ext_tf_shape;
  }
  if (shape._impl->tf_shape != nullptr)
    return *this->_impl->ext_tf_shape != *shape._impl->tf_shape;
  return *this->_impl->ext_tf_shape != *shape._impl->ext_tf_shape;
}

std::ostream &operator<<(std::ostream &os, const TensorShape &shape) {
  if (shape._impl->tf_shape != nullptr)
    os << *shape._impl->tf_shape;
  else
    os << *shape._impl->ext_tf_shape;
  return os;
}

} // namespace txeo
