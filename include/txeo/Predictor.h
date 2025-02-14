#ifndef PREDIOCTOR_H
#define PREDIOCTOR_H
#include <optional>
#pragma once

#include <memory>
#include <string>

#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"

namespace txeo {

template <typename T = float>
class Predictor {
  private:
    struct Impl;
    std::unique_ptr<Impl> _impl{nullptr};

  public:
    using TensorInfo = std::vector<std::pair<std::string, txeo::TensorShape>>;
    using TensorIdent = std::vector<std::pair<std::string, txeo::Tensor<T>>>;

    explicit Predictor(std::string model_path);
    ~Predictor();

    [[nodiscard]] const TensorInfo &get_input_metadata() const;
    [[nodiscard]] const TensorInfo &get_output_metadata() const;

    [[nodiscard]] std::optional<txeo::TensorShape>
    get_input_metadata_shape(const std::string &name) const;
    [[nodiscard]] std::optional<txeo::TensorShape>
    get_output_metadata_shape(const std::string &name) const;

    [[nodiscard]] txeo::Tensor<T> predict(const txeo::Tensor<T> input) const;
    [[nodiscard]] std::vector<txeo::Tensor<T>> predict(const TensorIdent &inputs) const;
};

class PredictorError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo
#endif