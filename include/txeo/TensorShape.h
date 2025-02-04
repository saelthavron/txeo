#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H
#include <cstdint>
#include <stdexcept>
#pragma once

#include <memory>
#include <ostream>
#include <vector>

// namespace txeo {

class TensorShapeError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class TensorShape {
  private:
    struct Impl;
    std::unique_ptr<Impl> _impl{nullptr};

  public:
    explicit TensorShape() = delete;
    explicit TensorShape(int order, int64_t dim);
    explicit TensorShape(std::vector<int64_t> shape);

    TensorShape(const TensorShape &shape);
    TensorShape(TensorShape &&shape) noexcept;
    TensorShape &operator=(const TensorShape &shape);
    TensorShape &operator=(TensorShape &&shape) noexcept;
    ~TensorShape() = default;

    [[nodiscard]] int order() const noexcept;
    [[nodiscard]] int64_t dim() const noexcept;
    [[nodiscard]] int64_t number_of_elements() const noexcept { return this->dim(); };
    [[nodiscard]] int64_t axis_dim(int axis) const;
    [[nodiscard]] std::vector<int64_t> axes_dims() const noexcept;
    [[nodiscard]] bool is_fully_defined() const noexcept;
    void push_dim_back(int64_t dim);

    bool operator==(const TensorShape &shape) const;
    bool operator!=(const TensorShape &shape) const;

    /*

    1. InsertDim(int index, int64_t size)	Inserts a dimension at a specific index.	void
    ✅ Yes
    2. RemoveDim(int index)	Removes a dimension at a specific index.	void	✅ Yes
    3. set_dim(int index, int64_t size)	Changes the size of an existing dimension.	void	✅
    Yes
    5. unknown_rank()	Checks if the rank is unknown.	bool	✅ Yes

    */

    friend std::ostream &operator<<(std::ostream &os, const TensorShape &shape);
};

//} // namespace txeo

#endif