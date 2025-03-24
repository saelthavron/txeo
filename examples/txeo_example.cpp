
#include "txeo/Loss.h"
#include "txeo/Matrix.h"
#include "txeo/MatrixIO.h"
#include "txeo/OlsGDTrainer.h"
#include "txeo/Tensor.h"
#include "txeo/TensorAgg.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorIO.h"
#include "txeo/TensorOp.h"
#include "txeo/TensorPart.h"
#include "txeo/types.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

namespace tf = tensorflow;
using namespace tensorflow::ops;

void MyLinearRegression() {

  std::vector<float> X_data = {1, 2, 3, 4};
  std::vector<float> Y_data = {2, 4, 6, 8};

  size_t p = 4;
  size_t n = 1;
  size_t m = 1;
  size_t epochs = 50;
  float learning_rate = 1.;
  bool enable_barzelay_borwein = true;
  float epsilon = 0.0001;
  auto loss_func = txeo::LossFunc::MSE;

  txeo::Matrix<float> X(p, n + 1);

  for (size_t i{0}; i < p; ++i) {
    for (size_t j{0}; j < n; ++j)
      X(i, j) = X_data[i];
    X(i, n) = 1.;
  }

  txeo::Matrix<float> Y(m, p, Y_data);

  auto X_t = txeo::TensorFunc<float>::transpose(X);
  auto Z = txeo::TensorOp<float>::product(X_t, X);
  auto K = txeo::TensorOp<float>::product(Y, X);

  auto norm_X = txeo::TensorAgg<float>::reduce_euclidean_norm(X, {0, 1})();
  auto norm_Y = txeo::TensorAgg<float>::reduce_euclidean_norm(Y, {0, 1})();

  txeo::Matrix<float> B_prev{m, n + 1, norm_Y / norm_X};
  if (enable_barzelay_borwein)
    learning_rate = 1.0f / (norm_X * norm_X);

  auto B = B_prev - ((txeo::TensorOp<float>::product(B_prev, Z) - K) * learning_rate);
  auto L = B - B_prev;
  auto Y_t = txeo::TensorFunc<float>::transpose(Y);
  txeo::Loss<float> loss{Y_t, loss_func};

  for (size_t e{0}; e < epochs; ++e) {
    auto B_t = txeo::TensorFunc<float>::transpose(B);
    auto loss_value = loss.get_loss(txeo::TensorOp<float>::product(X, B_t));
    std::cout << "Epoch " << e << ", Loss: " << loss_value << ", Learning Rate: " << learning_rate
              << std::endl;
    if (loss_value < epsilon)
      break;

    B_prev = B;
    if (enable_barzelay_borwein) {
      auto LZ = txeo::TensorOp<float>::product(L, Z);
      auto numerator = std::abs(txeo::TensorOp<float>::dot(L, LZ));
      auto denominator = txeo::TensorOp<float>::dot(LZ, LZ);
      learning_rate = numerator / denominator;
    };
    B -= (txeo::TensorOp<float>::product(B, Z) - K) * learning_rate;
    L = B - B_prev;
  }
  std::cout << "Weight: " << B << std::endl;
}

int main() {

  //  MyLinearRegression();

  // auto train_data =
  //     txeo::MatrixIO::one_hot_encode_text_file("/home/roberto/Downloads/housing.csv", ',', true,
  //                                              "/home/roberto/Downloads/housing_one_hot.csv");

  // txeo::MatrixIO train_data{"/home/roberto/Downloads/housing_one_hot.csv"};
  // auto x_train = train_data.read_text_file<double>(true).sub_matrix_cols({2, 3, 4, 5, 6, 7});
  // auto y_train = train_data.read_text_file<double>(true).sub_matrix_cols({8});

  // x_train.normalize_columns(txeo::NormalizationType::MIN_MAX);

  txeo::MatrixIO x_train_data{"/home/roberto/Downloads/regression_x_train.csv"};
  txeo::MatrixIO y_train_data{"/home/roberto/Downloads/regression_y_train.csv"};
  auto x_train = x_train_data.read_text_file<double>(true);
  auto y_train = y_train_data.read_text_file<double>(true);

  std::cout << "X: " << x_train.shape() << std::endl;
  std::cout << "Y: " << y_train.shape() << std::endl;
  std::cout << "min Y:" << txeo::TensorAgg<double>::reduce_min(y_train, {0})() << std::endl;

  txeo::OlsGDTrainer<double> ols{x_train, y_train};

  ols.enable_variable_lr();
  // //  ols.set_learning_rate(0.000001);
  ols.fit(50, txeo::LossFunc::MAE);

  std::cout << ols.weight_bias() << std::endl;

  // txeo::Vector<double> vec({1., 2., 3., 4., 5., 6., 7., 8., 9.});
  // vec.normalize(txeo::NormalizationType::MIN_MAX);

  // txeo::TensorShape shape({2, 3, 4});
  // txeo::Tensor<int> t(shape);

  // std::cout << t.dim() << std::endl;

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