#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H
#pragma once

#include <cstddef>
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

    template <typename T>
    friend class Predictor;

    template <typename T>
    friend class TensorAgg;

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
     * @param number_of_axes Number of axes of the tensor
     * @param dim  dimension in each axis
     *
     * @throw txeo::TensorShapeError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::TensorShape shape(3, 5); // A tensor shape with 3 dimensions, each of size 5
     *
     *     std::cout << "TensorShape created: " << shape << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit TensorShape(int number_of_axes, size_t dim);

    /**
     * @brief Constructs a tensor shape from a std::vector
     *
     * @param shape vector of dimensions
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     std::vector<size_t> dims = {3, 4, 5};
     *     txeo::TensorShape shape(dims); // Creates a 3D shape with dimensions 3x4x5
     *
     *     std::cout << "TensorShape created: " << shape << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit TensorShape(const std::vector<size_t> &shape);

    /**
     * @brief Constructs a tensor shape from a std::vector
     *
     * @param shape vector of dimensions
     * * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::TensorShape shape({3, 4, 5}); // Creates a 3D shape with dimensions 3x4x5
     *
     *     std::cout << "TensorShape created: " << shape << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit TensorShape(std::vector<size_t> &&shape);

    /**
     * @brief Constructs a tensor shape from an initializer list
     *
     * @param shape vector of dimensions
     * * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::TensorShape shape({3, 4, 5}); // Creates a 3D shape with dimensions 3x4x5
     *
     *     std::cout << "TensorShape created: " << shape << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit TensorShape(const std::initializer_list<size_t> &shape)
        : TensorShape(std::vector<size_t>(shape)) {}

    /**
     * @brief Returns the size of the tensor shape
     *
     * @return int Number of axes
     */
    [[nodiscard]] int number_of_axes() const noexcept;

    /**
     * @brief Synonym for @ref txeo::TensorShape::number_of_axes()
     *
     * @return int Size of this shape
     */
    [[nodiscard]] int size() const noexcept { return this->number_of_axes(); };

    /**
     * @brief Returns the dimension of the specified axis
     *
     * @param axis Axis whose dimension is to be returned
     * @return int64_t Dimension
     *
     * @throw txeo::TensorShapeError
     *
     */
    [[nodiscard]] int64_t axis_dim(int axis) const;

    /**
     * @brief Returns the stride of each dimension in the tensor.
     *
     * @details The stride represents the step size needed to move along each dimension of the
     * tensor in memory layout. It is useful for operations requiring efficient indexing.
     *
     * @return The stride for each dimension.
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor({3, 4, 5}); // Create a 3x4x5 tensor
     *     const std::vector<size_t>& strides = tensor.stride();
     *
     *     std::cout << "Tensor strides: ";
     *     for (size_t s : strides) {
     *         std::cout << s << " ";
     *     }
     *     std::cout << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    [[nodiscard]] const std::vector<size_t> &stride() const;

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
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::TensorShape shape({2, 3}); // Initial shape (2,3)
     *     shape.push_axis_back(4); // Adds a new axis, resulting in shape (2,3,4)
     *
     *     std::cout << "Updated TensorShape: " << shape << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    void push_axis_back(size_t dim);

    /**
     * @brief Inserts a dimension at the specified axis
     *
     * @param axis Specified axis
     * @param dim Specified dimension
     *
     * @throw txeo::TensorShapeError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::TensorShape shape({2, 3}); // Initial shape (2,3)
     *     shape.insert_axis(1, 4); // Inserts a new axis at index 1, resulting in shape (2,4,3)
     *
     *     std::cout << "Updated TensorShape: " << shape << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    void insert_axis(int axis, size_t dim);

    /**
     * @brief Removes a specified axis
     *
     * @param axis Specified axis
     *
     * @throw txeo::TensorShapeError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::TensorShape shape({2, 3, 4}); // Initial shape (2,3,4)
     *     shape.remove_axis(1); // Removes the axis at index 1, resulting in shape (2,4)
     *
     *     std::cout << "Updated TensorShape: " << shape << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    void remove_axis(int axis);

    /**
     * @brief Removes all axes from this shape, resulting an empty shape
     *
     */
    void remove_all_axes();

    /**
     * @brief Sets a dimension in a specified axis
     *
     * @param axis Axis whose dimension will be changed
     * @param dim New dimension
     *
     * @throw txeo::TensorShapeError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::TensorShape shape({2, 3, 4}); // Initial shape (2,3,4)
     *     shape.set_dim(1, 5); // Changes the second axis size from 3 to 5
     *
     *     std::cout << "Updated TensorShape: " << shape << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    void set_dim(int axis, size_t dim);

    /**
     * @brief Calculates the number of available tensor elements specified by the tensor shape
     *
     * @return int64_t Number of tensor elements
     */
    [[nodiscard]] size_t calculate_capacity() const noexcept;

    /**
     * @brief Returns a clone of this tensor
     *
     * @return TensorShape Clone of this tensor
     */
    [[nodiscard]] TensorShape clone() const;
};

/**
 * @brief Exceptions concerning @ref txeo::TensorShape
 *
 */
class TensorShapeError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif