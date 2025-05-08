
#include <chrono>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>

int main() {

  std::ofstream x;

  std::cout << std::boolalpha << x.is_open() << std::endl;
}
