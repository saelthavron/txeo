#include "txeo/Tensor.h"
#include <iostream>
#include <string>
#include <vector>

int main() {

  std::vector<std::string> vec{"roberto", "dias", "algarte"};

  int a = 2;

  Tensor p{"roberto", "algarte"};

  std::cout << p.nome_completo() << std::endl;

  return 0;
}
