#include "txeo/Tensor.h"

#include <string>

/*

txeo::Tensor<double>({2,2})

*/

std::string Tensor::nome_completo() {
  return nome + " " + sobrenome;
}
