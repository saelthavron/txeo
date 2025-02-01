#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#pragma once

#include <string>

/**
 * @brief Implements the mathematical concept of tensor, which is a magnitude of multiple order. A
 * tensor of order zero is defined to be a scalar, of order one a vector, of order two a matrix.
 * Each order of the tensor has a dimension.
 *
 */
class Tensor {

  private:
    std::string nome;
    std::string sobrenome;

  public:
    Tensor(std::string nome, std::string sobrenome)
        : nome(std::move(nome)), sobrenome(std::move(sobrenome)) {}

    std::string nome_completo();
};

#endif // TENSOR_H
