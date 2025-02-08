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
 * @brief The shape of a tensor is an ordered collection of dimensions of mathematical vector
 * spaces.
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

    template <typename T>
    friend class Tensor;

    template <typename P>
    void create_from_vector(P &&shape);

    explicit TensorShape();

  public:
    TensorShape(const TensorShape &shape);
    TensorShape(TensorShape &&shape) noexcept;
    ~TensorShape();

    TensorShape &operator=(const TensorShape &shape);
    TensorShape &operator=(TensorShape &&shape) noexcept;
    bool operator==(const TensorShape &shape) const;
    bool operator!=(const TensorShape &shape) const;

    friend std::ostream &operator<<(std::ostream &os, const TensorShape &shape);

    /**
     * @brief Constructs a tensor shape with axes having the same dimension
     *
     * @param number_of_axes Specifies the number of axes
     * @param dim Specifies the dimension in each axis
     */
    explicit TensorShape(int number_of_axes, int64_t dim);

    /**
     * @brief Constructs a tensor shape from a std::vector
     *
     * @param shape vector of dimensions
     */
    explicit TensorShape(const std::vector<int64_t> &shape);

    /**
     * @brief Constructs a tensor shape from a std::vector
     *
     * @param shape vector of dimensions
     */
    explicit TensorShape(std::vector<int64_t> &&shape);

    /**
     * @brief Returns the size of the tensor shape
     *
     * @return int
     */
    [[nodiscard]] int number_of_axes() const noexcept;

    /**
     * @brief Synonym for @ref txeo::TensorShape::number_of_axes()
     *
     * @return int
     */
    [[nodiscard]] int size() const noexcept { return this->number_of_axes(); };

    /**
     * @brief Returns the dimension of the specified axis
     *
     * @param axis Specifies axis
     * @return int64_t Dimension
     */
    [[nodiscard]] int64_t axis_dim(int axis) const;

    [[nodiscard]] const std::vector<int64_t> &stride() const;

    /**
     * @brief Returns the collection of tensor shape dimensions
     *
     * @return std::vector<int64_t>
     */
    [[nodiscard]] std::vector<int64_t> axes_dims() const noexcept;

    /**
     * @brief Indicates whether the tensor shape has any negative(undefined) dimensions or not
     *
     * @return true Does not have a negative dimension
     * @return false Has a negative dimension
     */
    [[nodiscard]] bool is_fully_defined() const noexcept;

    /**
     * @brief Inserts a dimension after the last axis
     *
     * @param dim Specified dimension
     */
    void push_axis_back(int64_t dim);

    /**
     * @brief Inserts a dimension at the specified axis
     *
     * @param axis Specified axis
     * @param dim Specified dimension
     */
    void insert_axis(int axis, int64_t dim);

    /**
     * @brief Removes a specified axis
     *
     * @param axis Specified axis
     */
    void remove_axis(int axis);

    /**
     * @brief Removes all axes
     *
     */
    void remove_all_axes();

    /**
     * @brief Sets a dimension in a specified axis
     *
     * @param axis Specified axis
     * @param dim Specified dimension
     */
    void set_dim(int axis, int64_t dim);

    /**
     * @brief Calculates the number of available tensor elements specified by the tensor shape
     *
     * @return int64_t
     */
    [[nodiscard]] int64_t calculate_capacity() const noexcept;
};

/**
 * @brief Exception class related to errors in @ref txeo::TensorShape
 *
 */
class TensorShapeError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif