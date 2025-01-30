#ifndef TENSOR_H
#define TENSOR_H
#pragma once

#include <string>

class Tensor {

  private:
    std::string nome;
    std::string sobrenome;

  public:
    Tensor(std::string nome, std::string sobrenome)
        : nome(std::move(nome)), sobrenome(std::move(sobrenome)) {}

    std::string nome_completo();
};

#endif