#include "txeo/TensorShape.h"

#include <cstdint>
#include <memory>
#include <string>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/errors.h>
#include <vector>

namespace txeo {

namespace tf = tensorflow;

struct TensorShape::Impl {
    std::unique_ptr<tf::TensorShape> tf_shape{nullptr};
};

TensorShape::TensorShape(std::vector<int64_t> shape) : _impl{std::make_unique<Impl>()} {
  for (auto item : shape) {
    if (item < 0)
      throw TensorShapeError("Negative dimension is not allowed.");
  }

  _impl->tf_shape = std::make_unique<tf::TensorShape>(shape);
}

TensorShape::TensorShape(int number_of_axes, int64_t dim) : _impl{std::make_unique<Impl>()} {
  if (dim < 0 || number_of_axes < 0)
    throw TensorShapeError("Negative number_of_axes or dimension is not allowed.");
  _impl->tf_shape = std::make_unique<tf::TensorShape>(std::vector<int64_t>(number_of_axes, dim));
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

TensorShape::~TensorShape() { // Defined here (not in the header) to avoid Impl incompleteness
}

int TensorShape::number_of_axes() const noexcept {
  return _impl->tf_shape->dims();
}

int64_t TensorShape::number_of_elements() const noexcept {
  return _impl->tf_shape->num_elements();
}

int64_t TensorShape::axis_dim(int axis) const {
  if (axis < 0 || axis >= this->number_of_axes())
    throw TensorShapeError("Axis " + std::to_string(axis) + " not in the range [0," +
                           std::to_string(this->number_of_axes() - 1) + "]");
  auto res = _impl->tf_shape->dim_size(axis);
  return res;
}

std::vector<int64_t> TensorShape::axes_dims() const noexcept {
  std::vector<int64_t> res;
  for (int i{0}; i < this->number_of_axes(); ++i)
    res.emplace_back(_impl->tf_shape->dim_size(i));

  return res;
}

bool TensorShape::is_fully_defined() const noexcept {
  return _impl->tf_shape->IsFullyDefined();
}

void TensorShape::push_axis_back(int64_t dim) {
  if (dim < 0)
    throw TensorShapeError("Negative dimension is not allowed.");
  std::cout << "NMumero de elementos: " << this->number_of_elements() << std::endl;

  _impl->tf_shape->AddDim(dim);
}

void TensorShape::insert_axis(int axis, int64_t dim) {
  if (axis < 0 || axis >= this->number_of_axes())
    throw TensorShapeError("Axis " + std::to_string(axis) + " not in the range [0," +
                           std::to_string(this->number_of_axes() - 1) + "]");
  if (dim < 0)
    throw TensorShapeError("Negative dimension is not allowed.");

  _impl->tf_shape->InsertDim(axis, dim);
}

void TensorShape::remove_axis(int axis) {
  if (axis < 0 || axis >= this->number_of_axes())
    throw TensorShapeError("Axis " + std::to_string(axis) + " not in the range [0," +
                           std::to_string(this->number_of_axes() - 1) + "]");

  _impl->tf_shape->RemoveDim(axis);
}

void TensorShape::set_dim(int axis, int64_t dim) {
  if (axis < 0 || axis >= this->number_of_axes())
    throw TensorShapeError("Axis " + std::to_string(axis) + " not in the range [0," +
                           std::to_string(this->number_of_axes() - 1) + "]");

  _impl->tf_shape->set_dim(axis, dim);
}

bool TensorShape::operator==(const TensorShape &shape) const {
  return *this->_impl->tf_shape == *shape._impl->tf_shape;
}

bool TensorShape::operator!=(const TensorShape &shape) const {
  return *this->_impl->tf_shape != *shape._impl->tf_shape;
}

std::ostream &operator<<(std::ostream &os, const TensorShape &shape) {
  os << shape._impl->tf_shape->DebugString();

  return os;
}

} // namespace txeo
