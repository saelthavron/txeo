#ifndef TENSORIO_H
#define TENSORIO_H
#pragma once

#include "txeo/Tensor.h"

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace txeo {

/**
 * @brief This class is deprecated. Please use class txeo::MatrixIO
 *
 */

class TensorIO {
  public:
    explicit TensorIO(const std::filesystem::path &path, char separator = ',')
        : _path(std::move(path)), _separator(separator) {};

    template <typename T>
    [[deprecated("Use class txeo::MatrixIO.")]]
    txeo::Tensor<T> read_text_file(bool has_header = false) const;

    template <typename T>
    [[deprecated("Use class txeo::MatrixIO.")]]
    void write_text_file(const txeo::Tensor<T> &tensor) const;

    template <typename T>
      requires(std::is_floating_point_v<T>)
    [[deprecated("Use class txeo::MatrixIO.")]]
    void write_text_file(const txeo::Tensor<T> &tensor, size_t precision) const;

    template <typename T>
    [[deprecated("Use class txeo::MatrixIO.")]]
    static txeo::Tensor<T> read_textfile(const std::filesystem::path &path, char separator = ',',
                                         bool has_header = false) {
      txeo::TensorIO io{path, separator};
      Tensor<T> resp{io.read_text_file<T>(has_header)};
      return resp;
    };

    template <typename T>
    [[deprecated("Use class txeo::MatrixIO.")]]
    static void write_textfile(const txeo::Tensor<T> &tensor, const std::filesystem::path &path,
                               char separator = ',') {
      txeo::TensorIO io{path, separator};
      io.write_text_file(tensor);
    }

    template <typename T>
      requires(std::is_floating_point_v<T>)
    [[deprecated("Use class txeo::MatrixIO.")]]
    static void write_textfile(const txeo::Tensor<T> &tensor, size_t precision,
                               const std::filesystem::path &path, char separator = ',') {
      txeo::TensorIO io{path, separator};
      io.write_text_file(tensor, precision);
    };

  private:
    std::filesystem::path _path;
    char _separator;
};

/**
 * @brief Exceptions concerning @ref txeo::TensorIO
 *
 */
class TensorIOError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif