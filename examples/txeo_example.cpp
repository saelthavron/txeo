#include "txeo/MatrixIO.h"
#include "txeo/Tensor.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorIO.h"
#include "txeo/Vector.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_set>

template <typename T>
void min_max_normalize(std::vector<T> &values) {
  auto [min, max] = std::ranges::minmax_element(values);
  auto dif = *max - *min;
  for (size_t i{0}; i < values.size(); ++i)
    values[i] = (values[i] - *min) / dif;
}

template <typename T>
void z_score_normalize(std::vector<T> &values) {
  T mean = 0.0;
  for (const auto &item : values)
    mean += item;
  mean /= values.size();

  T variance_num = 0.0;
  for (const auto &item : values) {
    auto dif = (item - mean);
    variance_num += dif * dif;
  }
  auto std_dev = std::sqrt(variance_num / (values.size() - 1.));

  for (size_t i{0}; i < values.size(); ++i)
    values[i] = (values[i] - mean) / std_dev;
}

int main() {

  // txeo::Vector<double> vec({1., 2., 3., 4., 5., 6., 7., 8., 9.});
  // vec.normalize(txeo::NormalizationType::MIN_MAX);

  txeo::TensorShape shape({2, 3, 4});
  txeo::Tensor<int> t(shape);

  std::cout << t.dim() << std::endl;

  //  txeo::TensorFunc<double>::normalize_by(tens, txeo::NormalizationType::MIN_MAX);

  // txeo::TensorFunc<double>::normalize_by(tens, 0, txeo::NormalizationType::Z_SCORE);

  // std::cout << tens << std::endl;

  // bool has_header = true;

  // std::filesystem::path _input_path{"/home/roberto/my_works/personal/txeo/examples/input.txt"};
  // std::ifstream rf{_input_path};
  // std::filesystem::path _output_path{"/home/roberto/my_works/personal/txeo/examples/output.txt"};
  // std::ofstream wf{_output_path};

  // if (!rf.is_open())
  //   throw txeo::MatrixIOError("Could not open file to read!");

  // if (!wf.is_open())
  //   throw txeo::MatrixIOError("Could not open file to write!");

  // char _separator = ',';
  // std::string line;
  // std::string word;
  // size_t n_cols{0};
  // size_t col{0};
  // bool first_line{true};

  // std::map<size_t, std::unordered_set<std::string>> lookups_map;

  // if (has_header)
  //   std::getline(rf, line);
  // while (std::getline(rf, line)) {
  //   if (std::count(line.begin(), line.end(), _separator) == 0) {
  //     rf.close();
  //     throw txeo::MatrixIOError("Separator not found!");
  //   };
  //   std::stringstream line_stream{line};
  //   while (std::getline(line_stream, word, _separator)) {
  //     if (!is_numeric(word)) {
  //       auto item = lookups_map.find(col);
  //       if (item != std::end(lookups_map)) {
  //         auto &lookups = item->second;
  //         auto item_lookup = lookups.find(word);
  //         if (item_lookup == std::end(lookups)) {
  //           lookups.emplace(word);
  //         }
  //       } else {
  //         if (!first_line) {
  //           rf.close();
  //           throw txeo::MatrixIOError("Different types in the same column!");
  //         }
  //         std::unordered_set<std::string> lookup;
  //         lookup.emplace(word);
  //         lookups_map.emplace(col, lookup);
  //       }
  //     }
  //     ++col;
  //   }
  //   if (n_cols != 0 && col != n_cols) {
  //     rf.close();
  //     throw txeo::MatrixIOError("Inconsistent number of columns!");
  //   }
  //   n_cols = col;
  //   col = 0;
  // }

  // rf.clear();
  // rf.seekg(0);

  // std::getline(rf, line);
  // col = 0;
  // std::string output_line{""};
  // std::stringstream line_stream{line};
  // while (std::getline(line_stream, word, _separator)) {
  //   if (!has_header)
  //     word = "col_" + std::to_string(col);
  //   if (!output_line.empty())
  //     output_line += _separator;
  //   auto item = lookups_map.find(col);
  //   if (item != std::end(lookups_map)) {
  //     auto &lookup_set = lookups_map.find(col)->second;
  //     for (auto &item : lookup_set) {
  //       if (output_line.back() != ',')
  //         output_line += _separator;
  //       output_line += word + "_" + item;
  //     }
  //   } else
  //     output_line += word;
  //   ++col;
  // }
  // wf << output_line << "\n";
  // if (!has_header) {
  //   rf.clear();
  //   rf.seekg(0);
  // }
  // while (std::getline(rf, line)) {
  //   std::stringstream line_stream{line};
  //   col = 0;
  //   output_line = "";
  //   while (std::getline(line_stream, word, _separator)) {
  //     if (!output_line.empty())
  //       output_line += _separator;
  //     if (!is_numeric(word)) {
  //       auto &lookup_set = lookups_map.find(col)->second;
  //       for (auto &item : lookup_set) {
  //         if (output_line.back() != ',')
  //           output_line += _separator;
  //         if (item == word)
  //           output_line += "1";
  //         else
  //           output_line += "0";
  //       }
  //     } else {
  //       output_line += word;
  //     }
  //     ++col;
  //   }
  //   wf << output_line << "\n";
  // }
  // wf.close();
  // rf.close();
  // return 0;
}

// int main() {

//   string model_path{"path/to/saved_model"};
//   string input_path{"path/to/input_tensor.txt"};
//   string output_path{"path/to/output_tensor.txt"};

//   Predictor<float> predictor{model_path};
//   auto input = TensorIO::read_textfile<float>(input_path);
//   auto output = predictor.predict(input);
//   TensorIO::write_textfile(output, output_path);

//   return 0;
// }