#ifndef TENSOR_ITERATOR_CONST_H
#define TENSOR_ITERATOR_CONST_H
#pragma once

#include <cstddef>

template <typename T>
class TensorIteratorConst {
  private:
    T *_elements;
    std::ptrdiff_t _step;

  public:
    TensorIteratorConst(T *elements, std::ptrdiff_t step = 1) : _elements(elements), _step(step) {}

    T &operator*() const { return *_elements; }
    T *operator->() const { return _elements; }

    TensorIteratorConst &operator++() {
      _elements += _step;
      return (*this);
    }

    TensorIteratorConst &operator++(int) {
      auto aux = *this;
      ++(*this);
      return aux;
    }

    TensorIteratorConst &operator--() {
      _elements -= _step;
      return *this;
    }

    TensorIteratorConst &operator--(int) {
      auto aux = *this;
      --(*this);
      return *this;
    }

    TensorIteratorConst &operator+=(const std::ptrdiff_t &val) {
      _elements += val * _step;
      return *this;
    }

    TensorIteratorConst &operator-=(const std::ptrdiff_t &val) {
      _elements -= val * _step;
      return *this;
    }

    friend TensorIteratorConst &operator+(TensorIteratorConst &iterator,
                                          const std::ptrdiff_t &val) {
      iterator += val;
      return iterator;
    }

    friend TensorIteratorConst &operator+(const std::ptrdiff_t &val,
                                          TensorIteratorConst &iterator) {
      iterator += val;
      return iterator;
    }

    friend TensorIteratorConst &operator-(TensorIteratorConst &iterator,
                                          const std::ptrdiff_t &val) {
      iterator -= val;
      return iterator;
    }

    std::ptrdiff_t operator-(const TensorIteratorConst &other) const {
      return (_elements - other._elements) / _step;
    }

    bool operator==(const TensorIteratorConst &other) const { return _elements == other._elements; }
    bool operator!=(const TensorIteratorConst &other) const { return _elements != other._elements; }
    bool operator<(const TensorIteratorConst &other) const { return _elements < other._elements; }
    bool operator>(const TensorIteratorConst &other) const { return _elements > other._elements; }
    bool operator<=(const TensorIteratorConst &other) const { return _elements <= other._elements; }
    bool operator>=(const TensorIteratorConst &other) const { return _elements >= other._elements; }
};

#endif