
#include "txeo/Tensor.h"
#include "txeo/TensorIO.h"
#include <iostream>

int main() {

  txeo::Tensor<double> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

  txeo::TensorIO::write_textfile(tensor, "tensor.txt");

  auto loaded_tensor = txeo::TensorIO::read_textfile<double>("tensor.txt");

  std::cout << loaded_tensor << std::endl;

  return 0;
}
