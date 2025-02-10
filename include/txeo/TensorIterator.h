#ifndef TENSOR_ITERATOR_H
#define TENSOR_ITERATOR_H
#pragma once

#include <cstddef>

template <typename T>
class TensorIterator {
  private:
    T *_elements;
    std::ptrdiff_t _step;

  public:
    TensorIterator(T *elements, std::ptrdiff_t step = 1) : _elements(elements), _step(step) {}

    T &operator*() const { return *_elements; }
    T *operator->() const { return _elements; }

    TensorIterator &operator++() {
      _elements += _step;
      return (*this);
    }

    TensorIterator &operator++(int) {
      auto aux = *this;
      ++(*this);
      return aux;
    }

    TensorIterator &operator--() {
      _elements -= _step;
      return *this;
    }

    TensorIterator &operator--(int) {
      auto aux = *this;
      --(*this);
      return *this;
    }

    TensorIterator &operator+=(const std::ptrdiff_t &val) {
      _elements += val * _step;
      return *this;
    }

    TensorIterator &operator-=(const std::ptrdiff_t &val) {
      _elements -= val * _step;
      return *this;
    }

    friend TensorIterator &operator+(TensorIterator &iterator, const std::ptrdiff_t &val) {
      iterator += val;
      return iterator;
    }

    friend TensorIterator &operator+(const std::ptrdiff_t &val, TensorIterator &iterator) {
      iterator += val;
      return iterator;
    }

    friend TensorIterator &operator-(TensorIterator &iterator, const std::ptrdiff_t &val) {
      iterator -= val;
      return iterator;
    }

    std::ptrdiff_t operator-(const TensorIterator &other) const {
      return (_elements - other._elements) / _step;
    }

    bool operator==(const TensorIterator &other) const { return _elements == other._elements; }
    bool operator!=(const TensorIterator &other) const { return _elements != other._elements; }
    bool operator<(const TensorIterator &other) const { return _elements < other._elements; }
    bool operator>(const TensorIterator &other) const { return _elements > other._elements; }
    bool operator<=(const TensorIterator &other) const { return _elements <= other._elements; }
    bool operator>=(const TensorIterator &other) const { return _elements >= other._elements; }
};

#endif