#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H
#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace txeo {

/**
 * @brief The shape of a tensor is an ordered set of dimensions of mathematical vector spaces.
 * @details Each position of the tensor shape is an "axis", labeled starting from zero, and the
 * value in this position is a "dimension". An empty tensor shape is the shape of a scalar value. In
 * some some conditions, a negative dimension represents an undefined dimension, but this
 * attribution is reserved to TensorFlow C++.
 *
 */
class TensorShape {
  private:
    struct Impl;
    std::unique_ptr<Impl> _impl{nullptr};

  public:
    explicit TensorShape() = delete;

    /**
     * @brief Constructs a tensor shape with axes having the same dimension
     *
     * @param number_of_axes The number of axes
     * @param dim The dimension in each axis
     */
    explicit TensorShape(int number_of_axes, int64_t dim);

    /**
     * @brief Constructs a tensor shape from a std::vector
     *
     * @param shape vector of dimensions
     */
    explicit TensorShape(std::vector<int64_t> shape);

    TensorShape(const TensorShape &shape);
    TensorShape(TensorShape &&shape) noexcept;
    TensorShape &operator=(const TensorShape &shape);
    TensorShape &operator=(TensorShape &&shape) noexcept;
    ~TensorShape();

    /**
     * @brief Returns the size of the tensor shape
     *
     * @return int
     */
    [[nodiscard]] int number_of_axes() const noexcept;

    /**
     * @brief Synonym for @ref TensorShape::number_of_axes()
     *
     * @return int
     */
    [[nodiscard]] int size() const noexcept { return this->number_of_axes(); };

    /**
     * @brief
     *
     * @return int64_t
     */
    [[nodiscard]] int64_t number_of_elements() const noexcept;
    [[nodiscard]] int64_t axis_dim(int axis) const;
    [[nodiscard]] std::vector<int64_t> axes_dims() const noexcept;
    [[nodiscard]] bool is_fully_defined() const noexcept;

    void push_axis_back(int64_t dim);
    void insert_axis(int axis, int64_t dim);
    void remove_axis(int axis);
    void set_dim(int axis, int64_t dim);

    bool operator==(const TensorShape &shape) const;
    bool operator!=(const TensorShape &shape) const;

    friend std::ostream &operator<<(std::ostream &os, const TensorShape &shape);
};

class TensorShapeError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif