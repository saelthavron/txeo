#ifndef PREDICTOR_H
#define PREDICTOR_H

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"

namespace txeo {

struct DeviceInfo {
    std::string name{};
    std::string device_type{};
    size_t memory_limit{};
};

template <typename T = float>
class Predictor {
  private:
    struct Impl;
    std::unique_ptr<Impl> _impl{nullptr};

    void load_model();

  public:
    using TensorInfo = std::vector<std::pair<std::string, txeo::TensorShape>>;
    using TensorIdent = std::vector<std::pair<std::string, txeo::Tensor<T>>>;

    explicit Predictor() = delete;
    Predictor(const Predictor &) = delete;
    Predictor(Predictor &&) = delete;
    Predictor &operator=(const Predictor &) = delete;
    Predictor &operator=(Predictor &&) = delete;

    explicit Predictor(std::filesystem::path model_path);
    ~Predictor();

    [[nodiscard]] const TensorInfo &get_input_metadata() const noexcept;
    [[nodiscard]] const TensorInfo &get_output_metadata() const noexcept;

    [[nodiscard]] std::optional<txeo::TensorShape>
    get_input_metadata_shape(const std::string &name) const;
    [[nodiscard]] std::optional<txeo::TensorShape>
    get_output_metadata_shape(const std::string &name) const;

    [[nodiscard]] std::vector<DeviceInfo> get_devices() const;

    [[nodiscard]] txeo::Tensor<T> predict(const txeo::Tensor<T> input) const;
    [[nodiscard]] std::vector<txeo::Tensor<T>> predict_batch(const TensorIdent &inputs) const;

    void enable_xla(bool enable);
};

class PredictorError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo
#endif