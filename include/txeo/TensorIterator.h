#ifndef TENSOR_ITERATOR_H
#define TENSOR_ITERATOR_H
#pragma once

#include <cstddef>
#include <iterator>

namespace txeo {

template <typename T>
class TensorIterator {
  public:
    using value_type = T;
    using pointer = T *;
    using reference = T &;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    TensorIterator(T *elements, std::ptrdiff_t step = 1) : _elements(elements), _step(step) {}

    inline reference operator*() const { return *_elements; }
    inline pointer operator->() const { return _elements; }

    inline TensorIterator &operator++() {
      _elements += _step;
      return *this;
    }

    inline TensorIterator operator++(int) {
      auto aux = *this;
      ++(*this);
      return aux;
    }

    inline TensorIterator &operator--() {
      _elements -= _step;
      return *this;
    }

    inline TensorIterator operator--(int) {
      auto aux = *this;
      --(*this);
      return aux;
    }

    inline TensorIterator &operator+=(const difference_type &val) {
      _elements += val * _step;
      return *this;
    }

    inline TensorIterator &operator-=(const difference_type &val) {
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

    inline difference_type operator-(const TensorIterator &other) const {
      return (_elements - other._elements) / _step;
    }

    inline bool operator==(const TensorIterator &other) const {
      return _elements == other._elements;
    }
    inline bool operator!=(const TensorIterator &other) const {
      return _elements != other._elements;
    }
    inline bool operator<(const TensorIterator &other) const { return _elements < other._elements; }
    inline bool operator>(const TensorIterator &other) const { return _elements > other._elements; }
    inline bool operator<=(const TensorIterator &other) const {
      return _elements <= other._elements;
    }
    inline bool operator>=(const TensorIterator &other) const {
      return _elements >= other._elements;
    }

  private:
    T *_elements;
    std::ptrdiff_t _step;
};

} // namespace txeo

#endif