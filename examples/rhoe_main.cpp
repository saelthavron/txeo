#include "txeo/Tensor.h"
#include <iostream>

int main() {

  Tensor p{"roberto", "algarte"};

  std::cout << p.nome_completo() << std::endl;

  return 0;
}
