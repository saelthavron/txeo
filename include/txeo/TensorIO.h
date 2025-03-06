#ifndef TENSORIO_H
#define TENSORIO_H
#pragma once

#include "txeo/Tensor.h"

#include <cstddef>
#include <filesystem>
#include <type_traits>

namespace txeo {

/**
 * @brief A class to read file data to a second order tensor and to write file data to a second
 * order tensor
 *
 */
class TensorIO {
  public:
    /**
     * @brief Constructs Tensor IO object
     *
     * @param path Path to the file
     * @param separator Character delimiting each element in a row
     */
    explicit TensorIO(const std::filesystem::path &path, char separator = ',')
        : _path(std::move(path)), _separator(separator) {};

    /**
     * @brief Returns a second order tensor with elements read from a text file
     *
     * @tparam T Data type of the tensor elements
     * @param has_header Whether the first line contains column headers
     * @return Tensor<T> Created tensor with data from the file
     *
     * @throws TensorIOError
     *
     * @par Example (Instance usage):
     * @code
     * txeo::TensorIO io("data.csv");
     * auto tensor = io.read_text_file<float>(true); // Read with header
     * std::cout << "Tensor shape: " << tensor.shape();
     * @endcode
     *
     */
    template <typename T>
    txeo::Tensor<T> read_text_file(bool has_header = false) const;

    /**
     * @brief Writes a second tensor to a text file
     *
     * @tparam T Data type of the tensor elements
     * @param tensor Tensor to write to file
     *
     * @throws TensorIOError
     *
     * @par Example (Instance usage):
     * @code
     * txeo::Tensor<int> data(txeo::TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
     * txeo::TensorIO io("output.csv");
     * io.write_text_file(data); // Writes as CSV
     * @endcode
     */
    template <typename T>
    void write_text_file(const txeo::Tensor<T> &tensor) const;

    /**
     * @brief Writes a floating-point second order tensor with specified precision to a text file
     *
     * @tparam T Floating-point type (float/double)
     * @param tensor Tensor to write
     * @param precision Number of decimal places to write
     *
     * @throws TensorIOError
     *
     * @par Example (Precision control):
     * @code
     * txeo::Tensor<double> values(txeo::TensorShape({1, 3}), {1.2345, 2.3456, 3.4567});
     * txeo::TensorIO io("results.csv");
     * io.write_text_file(values, 2); // Writes 1.23,2.35,3.46
     * @endcode
     */
    template <typename T>
      requires(std::is_floating_point_v<T>)
    void write_text_file(const txeo::Tensor<T> &tensor, size_t precision) const;

    /**
     * @brief Returns a second order tensor with elements read from a text file
     *
     * @tparam T Data type of tensor elements
     * @param path File path to read from
     * @param separator Column separator character
     * @param has_header Whether to skip first line as header
     * @return Tensor<T> Created tensor
     *
     * @par Example (One-time read):
     * @code
     * auto data = txeo::TensorIO::read_textfile<float>(
     *     "input.tsv", '\t', true
     * );
     * @endcode
     */
    template <typename T>
    static txeo::Tensor<T> read_textfile(const std::filesystem::path &path, char separator = ',',
                                         bool has_header = false) {
      txeo::TensorIO io{path, separator};
      Tensor<T> resp{io.read_text_file<T>(has_header)};
      return resp;
    };

    /**
     * @brief Writes a second order tensor to a text file
     *
     * @tparam T Data type of tensor elements
     * @param tensor Tensor to write
     * @param path Output file path
     * @param separator Column separator
     *
     * @par Example (One-time write):
     * @code
     * txeo::Tensor<int> matrix(txeo::TensorShape({3, 2}), {1, 2, 3, 4, 5, 6});
     * txeo::TensorIO::write_textfile(matrix, "matrix.csv");
     * @endcode
     */
    template <typename T>
    static void write_textfile(const txeo::Tensor<T> &tensor, const std::filesystem::path &path,
                               char separator = ',') {
      txeo::TensorIO io{path, separator};
      io.write_text_file(tensor);
    }

    /**
     * @brief Writes a floating-point second order tensor with specified precision to a text file
     *
     * @tparam T Floating-point type (float/double)
     * @param tensor Tensor to write
     * @param precision Decimal places to display
     * @param path Output file path
     * @param separator Column delimiter
     *
     * @par Example (Scientific notation):
     * @code
     * txeo::Tensor<double> results(txeo::TensorShape({2, 2}), {0.000123, 4567.8, 9.1, 234.567});
     * txeo::TensorIO::write_textfile(results, 3, "science.csv");
     * // Writes: 0.000,4567.800,9.100,234.567
     * @endcode
     */
    template <typename T>
      requires(std::is_floating_point_v<T>)
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