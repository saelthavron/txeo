#ifndef TENSOR_ITERATOR_H
#define TENSOR_ITERATOR_H
#pragma once

#include <cstddef>
#include <iterator>

namespace txeo {

template <typename T>
class TensorIterator {
  private:
    T *_elements;
    std::ptrdiff_t _step;

  public:
    using value_type = T;
    using pointer = T *;
    using reference = T &;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    TensorIterator(T *elements, std::ptrdiff_t step = 1) : _elements(elements), _step(step) {}

    reference operator*() const { return *_elements; }
    pointer operator->() const { return _elements; }

    TensorIterator &operator++() {
      _elements += _step;
      return *this;
    }

    TensorIterator operator++(int) {
      auto aux = *this;
      ++(*this);
      return aux;
    }

    TensorIterator &operator--() {
      _elements -= _step;
      return *this;
    }

    TensorIterator operator--(int) {
      auto aux = *this;
      --(*this);
      return aux;
    }

    TensorIterator &operator+=(const difference_type &val) {
      _elements += val * _step;
      return *this;
    }

    TensorIterator &operator-=(const difference_type &val) {
      _elements -= val * _step;
      return *this;
    }

    friend TensorIterator operator+(TensorIterator iterator, const difference_type &val) {
      iterator += val;
      return iterator;
    }

    friend TensorIterator operator+(const difference_type &val, TensorIterator iterator) {
      iterator += val;
      return iterator;
    }

    friend TensorIterator operator-(TensorIterator iterator, const difference_type &val) {
      iterator -= val;
      return iterator;
    }

    difference_type operator-(const TensorIterator &other) const {
      return (_elements - other._elements) / _step;
    }

    bool operator==(const TensorIterator &other) const { return _elements == other._elements; }
    bool operator!=(const TensorIterator &other) const { return _elements != other._elements; }
    bool operator<(const TensorIterator &other) const { return _elements < other._elements; }
    bool operator>(const TensorIterator &other) const { return _elements > other._elements; }
    bool operator<=(const TensorIterator &other) const { return _elements <= other._elements; }
    bool operator>=(const TensorIterator &other) const { return _elements >= other._elements; }
};

} // namespace txeo

#endif