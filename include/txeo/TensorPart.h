#ifndef TENSORPART_H
#define TENSORPART_H
#pragma once

#include "txeo/Tensor.h"

template <typename T>
class TensorPart {
  public:
    TensorPart() = delete;
    TensorPart(const TensorPart &) = delete;
    TensorPart(TensorPart &&) = delete;
    TensorPart &operator=(const TensorPart &) = delete;
    TensorPart &operator=(TensorPart &&) = delete;
    ~TensorPart();

    static std::vector<txeo::Tensor<T>> unstack(const txeo::Tensor<T> &tensor, size_t axis);
};

#endif