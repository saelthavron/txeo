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
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

namespace tf = tensorflow;
using namespace tensorflow::ops;

void LinearRegressionExample() {
  // 1. Create a TF scope
  tf::Scope root = tf::Scope::NewRootScope();

  // 2. Training data (replace with actual data loading)
  std::vector<float> X_data = {1, 2, 3, 4};
  std::vector<float> Y_data = {2, 4, 6, 8};

  tf::Tensor X_tensor(tf::DT_FLOAT, {4, 1});
  tf::Tensor Y_tensor(tf::DT_FLOAT, {4, 1});

  auto X_tensor_mapped = X_tensor.matrix<float>();
  auto Y_tensor_mapped = Y_tensor.matrix<float>();

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 1; ++j)
      X_tensor_mapped(i, j) = X_data[i];
    Y_tensor_mapped(i, 0) = Y_data[i];
  }

  // Initialize weights W and bias b
  tf::Tensor W(tf::DT_FLOAT, {1, 1});
  tf::Tensor b(tf::DT_FLOAT, {4, 1});
  auto W_flat = W.flat<float>();
  auto b_flat = b.flat<float>();
  for (int i = 0; i < 4; ++i) {
    W_flat(i) = 0.0f;
    b_flat(i) = 0.0f;
  }

  auto XW = tf::ops::MatMul(root, X_tensor, W); // 4,1 . 1,1
  auto Y_pred = tf::ops::Add(root, XW, b);      // 4,1 + 4,1

  auto error = tf::ops::Sub(root, Y_pred, Y_tensor);                  // 4,1 - 4,1
  auto loss = tf::ops::Mean(root, tf::ops::Square(root, error), {0}); // 1,1

  auto dW = tf::ops::MatMul(root, tf::ops::Transpose(root, X_tensor, {1, 0}), error); // 1,4 . 4,1
  auto db = tf::ops::Mean(root, error, {0});                                          // 4, 1

  auto W_update =
      tf::ops::Sub(root, W, tf::ops::Mul(root, tf::ops::Const(root, 0.001), dW)); // 1,1 - 1,1
  auto b_update =
      tf::ops::Sub(root, b, tf::ops::Mul(root, tf::ops::Const(root, 0.001), db)); // 4,1 - 4,1

  tf::ClientSession session(root);
  for (int epoch = 0; epoch < 50; ++epoch) {
    std::vector<tf::Tensor> outputs;
    auto status = session.Run({W_update, b_update}, &outputs);
    if (!status.ok())
      std::cout << "Error calculating weight and bias: " << status.ToString();
    W = outputs[0];
    b = outputs[1];

    if (epoch % 100 == 0) {
      std::vector<tf::Tensor> loss_output;
      status = session.Run({loss}, &loss_output);
      if (!status.ok())
        std::cout << "Error calculating loss model: " << status.ToString();
      std::cout << "Epoch " << epoch << ", Loss: " << loss_output[0].scalar<float>()() << std::endl;
    }
  }
  std::cout << "W: " << W << std::endl;
  std::cout << "b: " << b << std::endl;
}

int main() {

  LinearRegressionExample();

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