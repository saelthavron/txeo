#ifndef PREDICTOR_H
#define PREDICTOR_H

#pragma once

#include "txeo/Logger.h"
#include "txeo/LoggerConsole.h"
#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include "types.h"

#include <filesystem>
#include <optional>
#include <string>

namespace txeo {

/**
 * @brief Class that deals with the main tasks of prediction (inference)
 *
 * @tparam T Specifies the data type of the model involved
 */
template <typename T = float>
class Predictor {
  public:
    using TensorInfo = std::vector<std::pair<std::string, txeo::TensorShape>>;
    using TensorIdent = std::vector<std::pair<std::string, txeo::Tensor<T>>>;

    explicit Predictor() = delete;
    Predictor(const Predictor &) = delete;
    Predictor(Predictor &&) = delete;
    Predictor &operator=(const Predictor &) = delete;
    Predictor &operator=(Predictor &&) = delete;
    ~Predictor();

    /**
     * @brief Constructs a Predictor from a TensorFlow SavedModel directory.
     *
     * @details The directory must contain a valid SavedModel (typically with a .pb file).
     * For best performance, use models with frozen weights.
     *
     * @param model_path Path to the directory of the .pb saved model
     *
     * @throw PredictorError
     *
     * @note **Freezing Recommendation** (Python example):
     * @code
     *import tensorflow as tf
     *
     * # Load SavedModel
     * model = tf.saved_model.load("path/to/trained_model")
     *
     * # Freeze and save
     * concrete_func = model.signatures["serving_default"]
     * frozen_func = tf.python.framework.convert_to_constants.convert_variables_to_constants_v2(
     *     concrete_func
     * )
     * tf.io.write_graph(
     *     frozen_func.graph.as_graph_def(),
     *     "path/to/frozen_model",
     *     "frozen.pb",
     *     as_text=False
     * )
     * @endcode
     */
    explicit Predictor(std::filesystem::path model_path,
                       txeo::Logger &logger = txeo::LoggerConsole::instance());

    /**
     * @brief Returns the input tensor metadata for the loaded model
     *
     * @return const reference to vector of (name, shape) pairs
     *
     * @par Example:
     * @code
     * txeo::Predictor<float> predictor("model_dir");
     * for (const auto& [name, shape] : predictor.get_input_metadata()) {
     *     std::cout << "Input: " << name << " Shape: " << shape << "\n";
     * }
     * @endcode
     */
    [[nodiscard]] const TensorInfo &get_input_metadata() const noexcept;

    /**
     * @brief Returns the output tensor metadata for the loaded model
     *
     * @return const reference to vector of (name, shape) pairs
     *
     * @par Example:
     * @code
     * auto outputs = predictor.get_output_metadata();
     * std::cout << "Model produces " << outputs.size() << " outputs\n";
     * @endcode
     */
    [[nodiscard]] const TensorInfo &get_output_metadata() const noexcept;

    /**
     * @brief Returns shape for specific input tensor by name
     *
     * @param name Tensor name from model signature
     * @return std::optional containing shape if found
     *
     * @par Example:
     * @code
     * if (auto shape = predictor.get_input_metadata_shape("image_input")) {
     *     std::cout << "Input requires shape: " << *shape << "\n";
     * }
     * @endcode
     */
    [[nodiscard]] std::optional<txeo::TensorShape>
    get_input_metadata_shape(const std::string &name) const;

    /**
     * @brief Get shape for specific output tensor by name
     *
     * @param name Tensor name from model signature
     * @return std::optional containing shape if found
     *
     * @par Example:
     * @code
     * auto output_shape = predictor.get_output_metadata_shape("embeddings")
     *                   .value_or(txeo::TensorShape{0});
     * @endcode
     */
    [[nodiscard]] std::optional<txeo::TensorShape>
    get_output_metadata_shape(const std::string &name) const;

    /**
     * @brief Returns the available compute devices
     *
     * @return Vector of DeviceInfo structures
     *
     * @par Example:
     * @code
     * for (const auto& device : predictor.get_devices()) {
     *     std::cout << device.device_type << " device: " << device.name << "\n";
     * }
     * @endcode
     */
    [[nodiscard]] std::vector<DeviceInfo> get_devices() const;

    /**
     * @brief Perform single input/single output inference
     *
     * @param input Input tensor matching model's expected shape
     * @return Output tensor with inference results
     *
     * @throw PredictorError
     *
     * @par Example:
     * @code
     * Tensor<float> input({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
     * auto output = predictor.predict(input);
     * std::cout << "Prediction: " << output(0) << "\n";
     * @endcode
     */
    [[nodiscard]] txeo::Tensor<T> predict(const txeo::Tensor<T> &input) const;

    /**
     * @brief Perform batch inference with multiple named inputs
     *
     * @param inputs Vector of (name, tensor) pairs
     * @return Vector of output tensors
     *
     * @throw PredictorError
     *
     * @par Example:
     * @code
     * std::vector<std::pair<std::string, txeo::Tensor<float>>> inputs {
     *     {"image", image_tensor},
     *     {"metadata", meta_tensor}
     * };
     * auto results = predictor.predict_batch(inputs);
     * @endcode
     */
    [[nodiscard]] std::vector<txeo::Tensor<T>> predict_batch(const TensorIdent &inputs) const;

    /**
     * @brief Enable/disable XLA (Accelerated Linear Algebra) compilation
     * @param enable Whether to enable XLA optimizations
     *
     * @note Requires model reloading - prefer calling before first inference
     *
     * @par Example:
     * @code
     * predictor.enable_xla(true);  // Enable hardware acceleration
     * auto result = predictor.predict(input);  // Uses XLA-optimized graph
     * @endcode
     */
    void enable_xla(bool enable);

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl{nullptr};
    txeo::Logger *_logger;

    void load_model();
};

class PredictorError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo
#endif