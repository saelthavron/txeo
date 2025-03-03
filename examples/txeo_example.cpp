#include "txeo/Tensor.h"
#include "txeo/TensorIO.h"
#include <cstdint>
#include <iostream>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
// #include "tensorflow/core/framework/types.pb.h"

#include <algorithm>
#include <iterator>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

using namespace txeo;
using namespace std;

namespace tf = tensorflow;

int main() {

  tf::Tensor M(tf::DT_DOUBLE, tf::TensorShape({2, 2, 2}));
  auto aux_M = M.tensor<double, 3>();
  aux_M(0, 0, 0) = 1;
  aux_M(0, 0, 1) = 2;
  aux_M(0, 1, 0) = 3;
  aux_M(0, 1, 1) = 4;
  aux_M(1, 0, 0) = 5;
  aux_M(1, 0, 1) = 6;
  aux_M(1, 1, 0) = 7;
  aux_M(1, 1, 1) = 8;

  auto root = tf::Scope::NewRootScope();

  auto aux = tf::ops::Unstack(root, M, 2, tf::ops::Unstack::Attrs().Axis(1));

  tf::ClientSession session(root);
  std::vector<tf::Tensor> outputs;
  TF_CHECK_OK(session.Run({aux.output}, &outputs));
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::cout << "Tensor " << i << ": " << outputs[i] << std::endl;
  }

  return 0;
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