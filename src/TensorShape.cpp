#include "txeo/TensorShape.h"

#include <memory>
#include <string>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/errors.h>

// namespace txeo {

namespace tf = tensorflow;

struct TensorShape::Impl {
    std::unique_ptr<tf::TensorShape> tf_shape{nullptr};
};

TensorShape::TensorShape(std::vector<int64_t> shape) : _impl{std::make_unique<Impl>()} {
  _impl->tf_shape = std::make_unique<tf::TensorShape>(shape);
}

TensorShape::TensorShape(int order, int64_t dim) : _impl{std::make_unique<Impl>()} {
  if (dim < 0 || order < 0)
    throw TensorShapeError("Negative order or dimension is not allowed.");
  _impl->tf_shape = std::make_unique<tf::TensorShape>(std::vector(order, dim));
}

TensorShape::TensorShape(const TensorShape &shape) : _impl{std::make_unique<Impl>()} {
  if (this != &shape)
    _impl->tf_shape = std::make_unique<tf::TensorShape>(*shape._impl->tf_shape);
}

TensorShape::TensorShape(TensorShape &&shape) noexcept : _impl{std::make_unique<Impl>()} {
  if (this != &shape)
    _impl->tf_shape = std::make_unique<tf::TensorShape>(std::move(*shape._impl->tf_shape));
}

TensorShape &TensorShape::operator=(const TensorShape &shape) {
  *_impl->tf_shape = *shape._impl->tf_shape;

  return *this;
}

TensorShape &TensorShape::operator=(TensorShape &&shape) noexcept {
  *_impl->tf_shape = std::move(*shape._impl->tf_shape);

  return *this;
}

int TensorShape::order() const noexcept {
  return _impl->tf_shape->dims();
}

int64_t TensorShape::dim() const noexcept {
  return _impl->tf_shape->num_elements();
}

int64_t TensorShape::axis_dim(int axis) const {
  if (axis < 0 || axis >= this->order())
    throw TensorShapeError("Axis " + std::to_string(axis) + " not in the range [0," +
                           std::to_string(this->order() - 1) + "]");
  auto res = _impl->tf_shape->dim_size(axis);
  return res;
}

std::vector<int64_t> TensorShape::axes_dims() const noexcept {
  std::vector<int64_t> res;
  for (int i{0}; i < this->order(); ++i)
    res.emplace_back(_impl->tf_shape->dim_size(i));

  return res;
}

bool TensorShape::is_fully_defined() const noexcept {
  return _impl->tf_shape->IsFullyDefined();
}

void TensorShape::push_dim_back(int64_t dim) {
  if (dim < 0)
    throw TensorShapeError("Negative dimension is not allowed.");
  _impl->tf_shape->AddDim(dim);
}

bool TensorShape::operator==(const TensorShape &shape) const {
  return this->_impl->tf_shape == shape._impl->tf_shape;
}

bool TensorShape::operator!=(const TensorShape &shape) const {
  return this->_impl->tf_shape != shape._impl->tf_shape;
}

std::ostream &operator<<(std::ostream &os, const TensorShape &shape) {
  os << shape._impl->tf_shape->DebugString();

  return os;
}

//} // namespace txeo
