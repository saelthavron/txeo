#include "txeo/Tensor.h"
#include "txeo/TensorIO.h"
#include <iostream>

using namespace txeo;
using namespace std;

int main() {

  // 3x3 tensor created from a list of float values in row major scheme
  txeo::Tensor<double> tensor({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

  txeo::TensorIO::write_textfile(tensor, "tensor.txt");

  auto loaded_tensor = txeo::TensorIO::read_textfile<double>("tensor.txt");

  std::cout << loaded_tensor << std::endl;

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