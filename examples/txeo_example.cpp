
#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/TensorAgg.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorOp.h"
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
  float learning_rate = 0.01;
  bool enable_barzelay_borwein = true;
  float epsilon = 0.0001;

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

  txeo::Matrix<float> B{B_prev - (txeo::TensorOp<float>::product(B_prev, Z) - K) * learning_rate};
  txeo::Matrix<float> L{B - B_prev};

  for (size_t e{0}; e < epochs; ++e) {
    auto B_t = txeo::TensorFunc<float>::transpose(B);
    auto Y_t = txeo::TensorFunc<float>::transpose(Y);
    auto loss = txeo::TensorFunc<float>::square(txeo::TensorOp<float>::product(X, B_t) - Y_t);
    loss = txeo::TensorAgg<float>::reduce_mean(loss, {0, 1});
    std::cout << "Epoch " << e << ", Loss: " << loss() << std::endl;
    if (loss() < epsilon)
      break;

    B_prev = B;
    if (enable_barzelay_borwein) {
      txeo::Matrix<float> LZ{txeo::TensorOp<float>::product(L, Z)};
      auto numerator = std::abs(txeo::TensorOp<float>::dot(L, LZ));
      auto denominator = txeo::TensorOp<float>::dot(LZ, LZ);
      learning_rate = numerator / denominator;
    };
    B -= (txeo::TensorOp<float>::product(B, Z) - K) * learning_rate;
    L = txeo::Matrix<float>::to_matrix(B - B_prev);
  }
  std::cout << "Weight: " << B << std::endl;
}

void LinearRegressionExample() {
  tf::Scope root = tf::Scope::NewRootScope();

  std::vector<float> X_data = {1, 2, 3, 4};
  std::vector<float> Y_data = {2, 4, 6, 8};

  tf::Tensor X_tensor(tf::DT_FLOAT, {4, 2});
  tf::Tensor Y_tensor(tf::DT_FLOAT, {1, 4});

  auto X_tensor_mapped = X_tensor.matrix<float>();
  auto Y_tensor_mapped = Y_tensor.matrix<float>();

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 1; ++j)
      X_tensor_mapped(i, j) = X_data[i];
    Y_tensor_mapped(0, i) = Y_data[i];
    X_tensor_mapped(i, 1) = 1.0;
  }

  auto denominator = tf::ops::Mul(root, X_tensor, X_tensor);
  auto denominator_reduced = tf::ops::Sum(root, denominator, {0, 1});
  auto numerator = tf::ops::Mul(root, Y_tensor, Y_tensor);
  auto numerator_reduced = tf::ops::Sum(root, numerator, {0, 1});

  tf::ClientSession session(root);
  std::vector<tensorflow::Tensor> outputs;
  auto status = session.Run({denominator_reduced, numerator_reduced}, &outputs);
  if (!status.ok())
    std::cout << "Error calculating weight and bias: " << status.ToString();

  auto guess = outputs[1].scalar<float>()(0) / outputs[0].scalar<float>()(0);
  // Initialize weight W
  tf::Tensor W(tf::DT_FLOAT, {1, 2});
  auto W_flat = W.flat<float>();
  for (int i = 0; i < 2; ++i)
    W_flat(i) = std::sqrt(guess);

  auto Z = tf::ops::MatMul(root, tf::ops::Transpose(root, X_tensor, {1, 0}), X_tensor);

  auto WZ = tf::ops::MatMul(root, W, Z);
  auto YX = tf::ops::MatMul(root, Y_tensor, X_tensor);
  auto WZ_YX = tf::ops::Sub(root, WZ, YX);
  auto lr_Z_YX = tf::ops::Mul(root, tf::ops::Const(root, 0.001), WZ_YX);

  auto XW = tf::ops::MatMul(root, X_tensor, W);
  auto Y_pred = tf::ops::MatMul(root, W, X_tensor);

  auto error = tf::ops::Sub(root, Y_pred, Y_tensor);
  auto loss = tf::ops::Mean(root, tf::ops::Square(root, error), {0});

  auto W_update = tf::ops::Sub(root, W, lr_Z_YX);

  for (int epoch = 0; epoch < 50; ++epoch) {
    std::vector<tf::Tensor> outputs;
    auto status = session.Run({W_update}, &outputs);
    if (!status.ok())
      std::cout << "Error calculating weight and bias: " << status.ToString();
    W = outputs[0];

    if (epoch % 100 == 0) {
      std::vector<tf::Tensor> loss_output;
      status = session.Run({loss}, &loss_output);
      if (!status.ok())
        std::cout << "Error calculating loss model: " << status.ToString();
      std::cout << "Epoch " << epoch << ", Loss: " << loss_output[0].scalar<float>()() << std::endl;
    }
  }
  std::cout << "W: " << W << std::endl;
}

int main() {

  // MyLinearRegression();

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