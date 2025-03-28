#ifndef MATRIXIO_H
#define MATRIXIO_H
#pragma once

#include "txeo/Matrix.h"

#include <cstddef>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace txeo {

/**
 * @brief A class to read file data to matrix and to write file data to a matrix
 *
 */
class MatrixIO {
  public:
    /**
     * @brief Constructs MatrixIO object
     *
     * @param path Path to the file
     * @param separator Character delimiting each element in a row
     */
    explicit MatrixIO(const std::filesystem::path &path, char separator = ',')
        : _path(std::move(path)), _separator(separator) {};

    [[nodiscard]] std::filesystem::path path() const { return _path; }

    [[nodiscard]] char separator() const { return _separator; }

    /**
     * @brief Returns a matrix with elements read from a text file
     *
     * @tparam T Data type of the matrix elements
     * @param has_header Whether the first line contains column headers
     * @return Matrix<T> Created matrix with data from the file
     *
     * @throws MatrixIOError
     *
     * @par Example (Instance usage):
     * @code
     * txeo::MatrixIO io("data.csv");
     * auto matrix = io.read_text_file<float>(true); // Read with header
     * std::cout << "Matrix shape: " << matrix.shape();
     * @endcode
     *
     */
    template <typename T>
    txeo::Matrix<T> read_text_file(bool has_header = false) const;

    /**
     * @brief Writes a matrix to a text file
     *
     * @tparam T Data type of the matrix elements
     * @param matrix Matrix to write to file
     *
     * @throws MatrixIOError
     *
     * @par Example (Instance usage):
     * @code
     * txeo::Matrix<int> data(txeo::TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
     * txeo::MatrixIO io("output.csv");
     * io.write_text_file(data); // Writes as CSV
     * @endcode
     */
    template <typename T>
    void write_text_file(const txeo::Matrix<T> &matrix) const;

    /**
     * @brief Writes a floating-point matrix with specified precision to a text file
     *
     * @tparam T Floating-point type (float/double)
     * @param matrix Matrix to write
     * @param precision Number of decimal places to write
     *
     * @throws MatrixIOError
     *
     * @par Example (Precision control):
     * @code
     * txeo::Matrix<double> values(txeo::TensorShape({1, 3}), {1.2345, 2.3456, 3.4567});
     * txeo::MatrixIO io("results.csv");
     * io.write_text_file(values, 2); // Writes 1.23,2.35,3.46
     * @endcode
     */
    template <typename T>
      requires(std::is_floating_point_v<T>)
    void write_text_file(const txeo::Matrix<T> &matrix, size_t precision) const;

    /**
     * @brief Returns a matrix with elements read from a text file
     *
     * @tparam T Data type of matrix elements
     * @param path File path to read from
     * @param separator Column separator character
     * @param has_header Whether to skip first line as header
     * @return Matrix<T> Created matrix
     *
     * @par Example (One-time read):
     * @code
     * auto data = txeo::MatrixIO::read_textfile<float>(
     *     "input.tsv", '\t', true
     * );
     * @endcode
     */
    template <typename T>
    static txeo::Matrix<T> read_textfile(const std::filesystem::path &path, char separator = ',',
                                         bool has_header = false) {
      txeo::MatrixIO io{path, separator};
      Matrix<T> resp{io.read_text_file<T>(has_header)};
      return resp;
    };

    /**
     * @brief Writes a matrix to a text file
     *
     * @tparam T Data type of matrix elements
     * @param matrix Matrix to write
     * @param path Output file path
     * @param separator Column separator
     *
     * @par Example (One-time write):
     * @code
     * txeo::Matrix<int> matrix(txeo::TensorShape({3, 2}), {1, 2, 3, 4, 5, 6});
     * txeo::MatrixIO::write_textfile(matrix, "matrix.csv");
     * @endcode
     */
    template <typename T>
    static void write_textfile(const txeo::Matrix<T> &matrix, const std::filesystem::path &path,
                               char separator = ',') {
      txeo::MatrixIO io{path, separator};
      io.write_text_file(matrix);
    }

    /**
     * @brief Writes a floating-point matrix with specified precision to a text file
     *
     * @tparam T Floating-point type (float/double)
     * @param matrix Matrix to write
     * @param precision Decimal places to display
     * @param path Output file path
     * @param separator Column delimiter
     *
     * @par Example (Scientific notation):
     * @code
     * txeo::Matrix<double> results(txeo::TensorShape({2, 2}), {0.000123, 4567.8, 9.1, 234.567});
     * txeo::MatrixIO::write_textfile(results, 3, "science.csv");
     * // Writes: 0.000,4567.800,9.100,234.567
     * @endcode
     */
    template <typename T>
      requires(std::is_floating_point_v<T>)
    static void write_textfile(const txeo::Matrix<T> &matrix, size_t precision,
                               const std::filesystem::path &path, char separator = ',') {
      txeo::MatrixIO io{path, separator};
      io.write_text_file(matrix, precision);
    };

    /**
     * @brief Performs one-hot encoding in all non-numeric columns in a text file and writes the
     * result to a target file.
     *
     * @param source_path The path to the source text file containing the input data.
     * @param separator The delimiter used in the input file (e.g., ',' for CSV).
     * @param has_header A flag indicating whether the input file has a header row.
     * @param target_path The path to the target text file where the encoded data will be written.
     * @return A `MatrixIO` object representing the encoded data file pointing to target_path.
     *
     * @throws txeo::MatrixIOError If the source and target paths are the same, if the file cannot
     * be opened, or if the input data is inconsistent (e.g., different types in the same column).
     *
     * **Example Usage:**
     * @code
     * #include "MatrixIO.h"
     * #include <filesystem>
     *
     * int main() {
     *     std::filesystem::path source_path = "input.csv";
     *     std::filesystem::path target_path = "output.csv";
     *     char separator = ',';
     *     bool has_header = true;
     *
     *     try
     *     {
     *          auto matrix_io = MatrixIO::one_hot_encode_text_file(source_path, separator,
     * has_header, target_path);
     *          std::cout << "One-hot encoding completed successfully. Output
     * written to: " << target_path << std::endl;
     *      } catch (const txeo::MatrixIOError &e) { std::cerr
     * << "Error: " << e.what() << std::endl;
     *     }
     *
     *     return 0;
     * }
     * @endcode
     */
    static txeo::MatrixIO one_hot_encode_text_file(const std::filesystem::path &source_path,
                                                   char separator, bool has_header,
                                                   const std::filesystem::path &target_path);

  private:
    std::filesystem::path _path;
    char _separator;

    static std::map<size_t, std::unordered_set<std::string>>
    build_lookups_map(const std::filesystem::path &source_path, char separator, bool has_header);

    static std::string
    build_target_header(const std::filesystem::path &source_path, char separator, bool has_header,
                        const std::map<size_t, std::unordered_set<std::string>> &lookups_map);
};

/**
 * @brief Exceptions concerning @ref txeo::MatrixIO
 *
 */
class MatrixIOError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif