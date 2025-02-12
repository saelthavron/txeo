#include "txeo/TensorIO.h"
#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include "txeo/detail/utils.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

namespace txeo {

template <typename T>
txeo::Tensor<T> TensorIO::read_text_file(bool has_header) const {
  std::string line;
  std::string word;
  size_t n_rows{0};
  size_t n_cols{0};
  size_t aux{0};

  std::ifstream rf{_path};
  if (rf.is_open()) {
    while (std::getline(rf, line)) {
      if (std::count(line.begin(), line.end(), _separator) == 0) {
        rf.close();
        throw txeo::TensorIOError("Separator not found!");
      };
      std::stringstream line_stream{line};
      while (std::getline(line_stream, word, _separator))
        ++aux;
      if (n_cols != 0 && aux != n_cols) {
        rf.close();
        throw txeo::TensorIOError("Inconsistent number of columns!");
      }
      n_cols = aux;
      aux = 0;
      ++n_rows;
    }
    if (n_rows == 0) {
      rf.close();
      throw txeo::TensorIOError("File can not be empty!");
    }
    if (has_header)
      --n_rows;
  } else
    throw txeo::TensorIOError("Could not open file!");

  txeo::Tensor<T> resp{txeo::TensorShape({n_rows, n_cols})};
  rf.clear();
  rf.seekg(0);
  auto iterator = std::begin(resp);
  if (has_header)
    std::getline(rf, line);
  while (std::getline(rf, line)) {
    std::stringstream line_stream{line};
    while (std::getline(line_stream, word, _separator)) {
      try {
        *iterator = static_cast<T>(std::stod(word));
      } catch (...) {
        rf.close();
        throw txeo::TensorIOError("Invalid element!");
      }
      ++iterator;
    }
  }
  rf.close();
  return resp;
}

template <typename T>
void TensorIO::write_text_file(const txeo::Tensor<T> &tensor) const {
  if (tensor.order() != 2)
    throw txeo::TensorIOError("Tensor is not a matrix!");
  std::ofstream wf{_path, std::ios::out};
  if (wf.is_open()) {
    size_t n_rows = txeo::detail::to_size_t(tensor.shape().axis_dim(0));
    size_t n_cols = txeo::detail::to_size_t(tensor.shape().axis_dim(1));
    auto iterator = std::cbegin(tensor);
    size_t aux_r{0};
    while (aux_r < n_rows) {
      size_t aux_c{0};
      while (aux_c < n_cols) {
        wf << std::to_string(*iterator);
        ++iterator;
        if (++aux_c < n_cols)
          wf << _separator;
      }
      if (++aux_r < n_rows)
        wf << "\n";
    }
    wf.close();
  } else
    throw txeo::TensorIOError("Could not open file!");
}

template <typename T>
  requires(std::is_floating_point_v<T>)
void TensorIO::write_text_file(const txeo::Tensor<T> &tensor, size_t precision) const {
  if (precision <= 1)
    throw txeo::TensorIOError("Precision must be greater than 1!");
  --precision;
  if (tensor.order() != 2)
    throw txeo::TensorIOError("Tensor is not a matrix!");
  std::ofstream wf{_path, std::ios::out};
  if (wf.is_open()) {
    size_t n_rows = txeo::detail::to_size_t(tensor.shape().axis_dim(0));
    size_t n_cols = txeo::detail::to_size_t(tensor.shape().axis_dim(1));
    auto iterator = std::cbegin(tensor);
    size_t aux_r{0};
    while (aux_r < n_rows) {
      size_t aux_c{0};
      while (aux_c < n_cols) {
        wf << txeo::detail::format(*iterator, precision);
        ++iterator;
        if (++aux_c < n_cols)
          wf << _separator;
      }
      if (++aux_r < n_rows)
        wf << "\n";
    }
    wf.close();
  } else
    throw txeo::TensorIOError("Could not open file!");
}

template txeo::Tensor<short> TensorIO::read_text_file<short>(bool has_header) const;
template txeo::Tensor<int> TensorIO::read_text_file<int>(bool has_header) const;
template txeo::Tensor<bool> TensorIO::read_text_file<bool>(bool has_header) const;
template txeo::Tensor<long> TensorIO::read_text_file<long>(bool has_header) const;
template txeo::Tensor<long long> TensorIO::read_text_file<long long>(bool has_header) const;
template txeo::Tensor<float> TensorIO::read_text_file<float>(bool has_header) const;
template txeo::Tensor<double> TensorIO::read_text_file<double>(bool has_header) const;

template void TensorIO::write_text_file(const txeo::Tensor<short> &tensor) const;
template void TensorIO::write_text_file(const txeo::Tensor<int> &tensor) const;
template void TensorIO::write_text_file(const txeo::Tensor<bool> &tensor) const;
template void TensorIO::write_text_file(const txeo::Tensor<long> &tensor) const;
template void TensorIO::write_text_file(const txeo::Tensor<long long> &tensor) const;
template void TensorIO::write_text_file(const txeo::Tensor<float> &tensor) const;
template void TensorIO::write_text_file(const txeo::Tensor<double> &tensor) const;

template void TensorIO::write_text_file(const txeo::Tensor<float> &tensor, size_t precision) const;
template void TensorIO::write_text_file(const txeo::Tensor<double> &tensor, size_t precision) const;

} // namespace txeo
