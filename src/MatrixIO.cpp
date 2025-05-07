#include "txeo/MatrixIO.h"
#include "txeo/detail/utils.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

namespace txeo {

template <typename T>
Matrix<T> MatrixIO::read_text_file(bool has_header) const {
  std::string line;
  std::string word;
  size_t n_rows{0};
  size_t n_cols{0};
  size_t aux{0};

  std::ifstream rf{_path};
  if (rf.is_open()) {
    _logger->info("Reading text file...");
    while (std::getline(rf, line)) {
      if (line.empty())
        continue;

      std::stringstream line_stream{line};
      while (std::getline(line_stream, word, _separator))
        ++aux;
      if (n_cols != 0 && aux != n_cols) {
        rf.close();
        throw MatrixIOError("Inconsistent number of columns!");
      }
      n_cols = aux;
      aux = 0;
      ++n_rows;
    }
    if (n_rows == 0) {
      rf.close();
      throw MatrixIOError("File can not be empty!");
    }
    if (has_header)
      --n_rows;
  } else
    throw MatrixIOError("Could not open file!");

  Matrix<T> resp{n_rows, n_cols};
  rf.clear();
  rf.seekg(0);
  auto resp_ite = std::begin(resp);
  if (has_header)
    std::getline(rf, line);
  size_t line_number{0};
  while (std::getline(rf, line)) {
    ++line_number;
    if (line.empty())
      continue;
    std::stringstream line_stream{line};
    while (std::getline(line_stream, word, _separator)) {
      try {
        *resp_ite = static_cast<T>(std::stod(word));
      } catch (...) {
        rf.close();
        throw MatrixIOError("Invalid element at line " + std::to_string(line_number));
      }
      ++resp_ite;
    }
  }
  rf.close();
  return resp;
}

template <typename T>
void MatrixIO::write_text_file(const Matrix<T> &tensor) const {
  if (tensor.order() != 2)
    throw MatrixIOError("Tensor is not a matrix!");

  std::ofstream wf{_path, std::ios::out};
  if (wf.is_open()) {
    _logger->info("Writing text file...");
    size_t n_rows = detail::to_size_t(tensor.shape().axis_dim(0));
    size_t n_cols = detail::to_size_t(tensor.shape().axis_dim(1));
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
    throw MatrixIOError("Could not open file!");
}

template <typename T>
  requires(std::is_floating_point_v<T>)
void MatrixIO::write_text_file(const Matrix<T> &tensor, size_t precision) const {
  if (precision <= 1)
    throw MatrixIOError("Precision must be greater than 1!");
  auto prec = precision - 1;
  if (tensor.order() != 2)
    throw MatrixIOError("Tensor is not a matrix!");
  std::ofstream wf{_path, std::ios::out};
  if (wf.is_open()) {
    _logger->info("Writing text file...");
    size_t n_rows = detail::to_size_t(tensor.shape().axis_dim(0));
    size_t n_cols = detail::to_size_t(tensor.shape().axis_dim(1));
    auto iterator = std::cbegin(tensor);
    size_t aux_r{0};
    while (aux_r < n_rows) {
      size_t aux_c{0};
      while (aux_c < n_cols) {
        wf << detail::format(*iterator, prec);
        ++iterator;
        if (++aux_c < n_cols)
          wf << _separator;
      }
      if (++aux_r < n_rows)
        wf << "\n";
    }
    wf.close();
  } else
    throw MatrixIOError("Could not open file!");
}

std::map<size_t, std::unordered_set<std::string>>
MatrixIO::build_lookups_map(const std::filesystem::path &source_path, char separator,
                            bool has_header) {
  std::ifstream rf{source_path};

  if (!rf.is_open())
    throw MatrixIOError("Could not open file to read!");

  std::string line;
  std::string word;
  size_t n_cols{0};
  size_t col{0};
  bool first_line{true};

  std::map<size_t, std::unordered_set<std::string>> resp;

  if (has_header)
    std::getline(rf, line);
  while (std::getline(rf, line)) {
    if (line.empty())
      continue;
    if (std::count(line.begin(), line.end(), separator) == 0) {
      rf.close();
      throw MatrixIOError("Separator not found!");
    };
    std::stringstream line_stream{line};
    while (std::getline(line_stream, word, separator)) {
      if (!detail::is_numeric(word)) {
        auto item = resp.find(col);
        if (item != std::end(resp)) {
          auto &lookups = item->second;
          auto item_lookup = lookups.find(word);
          if (item_lookup == std::end(lookups)) {
            lookups.emplace(word);
          }
        } else {
          if (!first_line) {
            rf.close();
            throw MatrixIOError("Different types in the same column!");
          }
          std::unordered_set<std::string> lookup;
          lookup.emplace(word);
          resp.emplace(col, lookup);
        }
      }
      ++col;
    }
    if (n_cols != 0 && col != n_cols) {
      rf.close();
      throw MatrixIOError("Inconsistent number of columns!");
    }
    n_cols = col;
    col = 0;
  }
  rf.close();
  return resp;
}

std::string MatrixIO::build_target_header(
    const std::filesystem::path &source_path, char separator, bool has_header,
    const std::map<size_t, std::unordered_set<std::string>> &lookups_map) {
  std::ifstream rf{source_path};

  if (!rf.is_open())
    throw MatrixIOError("Could not open file to read!");

  std::string line;
  std::string word;
  size_t col{0};

  std::getline(rf, line);
  col = 0;
  std::string resp{""};
  std::stringstream line_stream{line};
  while (std::getline(line_stream, word, separator)) {
    if (!has_header)
      word = "col_" + std::to_string(col);
    if (!resp.empty())
      resp += separator;
    auto item = lookups_map.find(col);
    if (item != std::end(lookups_map)) {
      auto &lookup_set = lookups_map.find(col)->second;
      for (auto &item : lookup_set) {
        if (resp.back() != ',')
          resp += separator;
        resp += word + "_" + item;
      }
    } else
      resp += word;
    ++col;
  }

  rf.close();
  return resp;
}

txeo::MatrixIO MatrixIO::one_hot_encode_text_file(const std::filesystem::path &source_path,
                                                  char separator, bool has_header,
                                                  const std::filesystem::path &target_path,
                                                  txeo::Logger &logger) {
  if (source_path == target_path)
    throw MatrixIOError("Source and target paths cannot be equal!");

  std::ifstream rf{source_path};
  std::ofstream wf{target_path};

  if (!rf.is_open())
    throw MatrixIOError("Could not open file to read!");

  if (!wf.is_open())
    throw MatrixIOError("Could not open file to write!");

  auto lookups_map = MatrixIO::build_lookups_map(source_path, separator, has_header);

  auto target_header =
      MatrixIO::build_target_header(source_path, separator, has_header, lookups_map);

  logger.info("Building one-hot-encoded file...");

  wf << target_header << "\n";

  std::string line;
  std::string word;
  size_t col{0};
  if (has_header)
    std::getline(rf, line);

  std::string target_line{""};
  std::stringstream line_stream{line};
  while (std::getline(rf, line)) {
    if (line.empty())
      continue;
    std::stringstream line_stream{line};
    col = 0;
    target_line = "";
    while (std::getline(line_stream, word, separator)) {
      if (!target_line.empty())
        target_line += separator;
      if (!detail::is_numeric(word)) {
        auto &lookup_set = lookups_map.find(col)->second;
        for (auto &item : lookup_set) {
          if (target_line.back() != ',')
            target_line += separator;
          if (item == word)
            target_line += "1";
          else
            target_line += "0";
        }
      } else
        target_line += word;
      ++col;
    }
    wf << target_line << "\n";
  }
  wf.close();
  rf.close();

  MatrixIO resp{target_path, separator};

  return resp;
}

template Matrix<short> MatrixIO::read_text_file<short>(bool has_header) const;
template Matrix<int> MatrixIO::read_text_file<int>(bool has_header) const;
template Matrix<bool> MatrixIO::read_text_file<bool>(bool has_header) const;
template Matrix<long> MatrixIO::read_text_file<long>(bool has_header) const;
template Matrix<long long> MatrixIO::read_text_file<long long>(bool has_header) const;
template Matrix<float> MatrixIO::read_text_file<float>(bool has_header) const;
template Matrix<double> MatrixIO::read_text_file<double>(bool has_header) const;
template Matrix<size_t> MatrixIO::read_text_file<size_t>(bool has_header) const;

template void MatrixIO::write_text_file(const Matrix<short> &tensor) const;
template void MatrixIO::write_text_file(const Matrix<int> &tensor) const;
template void MatrixIO::write_text_file(const Matrix<bool> &tensor) const;
template void MatrixIO::write_text_file(const Matrix<long> &tensor) const;
template void MatrixIO::write_text_file(const Matrix<long long> &tensor) const;
template void MatrixIO::write_text_file(const Matrix<float> &tensor) const;
template void MatrixIO::write_text_file(const Matrix<double> &tensor) const;
template void MatrixIO::write_text_file(const Matrix<size_t> &tensor) const;

template void MatrixIO::write_text_file(const Matrix<float> &tensor, size_t precision) const;
template void MatrixIO::write_text_file(const Matrix<double> &tensor, size_t precision) const;

} // namespace txeo
