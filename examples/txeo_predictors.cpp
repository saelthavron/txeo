#include "txeo/Predictor.h"
#include "txeo/Tensor.h"
#include <iostream>

int main() {
  // ======================================================================================
  // 1. Basic Initialization & Metadata Inspection
  // ======================================================================================
  try {
    std::cout << "=== 1. Initializing Predictor & Inspecting Model ===" << "\n";
    txeo::Predictor<float> predictor("path/to/saved_model");

    // Inspect input requirements
    std::cout << "\nInput Metadata:\n";
    for (const auto &[name, shape] : predictor.get_input_metadata()) {
      std::cout << "- " << name << " : " << shape << "\n";
    }

    // Inspect output specifications
    std::cout << "\nOutput Metadata:\n";
    for (const auto &[name, shape] : predictor.get_output_metadata()) {
      std::cout << "- " << name << " : " << shape << "\n";
    }
  } catch (const txeo::PredictorError &e) {
    std::cerr << "Initialization failed: " << e.what() << "\n";
    return 1;
  }

  // ======================================================================================
  // 2. Single Input/Output Inference
  // ======================================================================================
  {
    std::cout << "\n=== 2. Single Input Inference ===" << "\n";
    txeo::Predictor<float> predictor("path/to/saved_model");

    // Create input tensor matching model requirements
    txeo::Tensor<float> input({1, 224, 224, 3}); // Example image input shape
    input.fill_with_uniform_random(0.0f, 1.0f);  // Simulate normalized image data

    // Perform inference
    try {
      auto output = predictor.predict(input);
      std::cout << "Inference succeeded. Output shape: " << output.shape() << "\n";
      std::cout << "First 5 output values: ";
      for (size_t i = 0; i < 5; ++i) {
        std::cout << output(i) << " ";
      }
      std::cout << "\n";
    } catch (const txeo::PredictorError &e) {
      std::cerr << "Inference failed: " << e.what() << "\n";
    }
  }

  // ======================================================================================
  // 3. Multi-Input Batch Inference
  // ======================================================================================
  {
    std::cout << "\n=== 3. Batch Inference ===" << "\n";
    txeo::Predictor<float> predictor("path/to/multi_input_model");

    // Prepare batch of named inputs
    txeo::Predictor<float>::TensorIdent inputs = {
        {"image_input", txeo::Tensor<float>({2, 128, 128, 3})}, // Batch of 2 images
        {"meta_input", txeo::Tensor<float>({2, 10})}            // Accompanying metadata
    };

    // Fill with sample data
    inputs[0].second.fill(0.5f); // Simulate image data
    inputs[1].second.fill(1.0f); // Simulate metadata

    try {
      auto outputs = predictor.predict_batch(inputs);
      std::cout << "Batch inference completed. Number of outputs: " << outputs.size() << "\n";
      for (size_t i = 0; i < outputs.size(); ++i) {
        std::cout << "Output " << i << " shape: " << outputs[i].shape() << "\n";
      }
    } catch (const txeo::PredictorError &e) {
      std::cerr << "Batch inference failed: " << e.what() << "\n";
    }
  }

  // ======================================================================================
  // 4. Device Information & Hardware Utilization
  // ======================================================================================
  {
    std::cout << "\n=== 4. Device Information ===" << "\n";
    txeo::Predictor<float> predictor("path/to/saved_model");

    auto devices = predictor.get_devices();
    std::cout << "Available compute devices:\n";
    for (const auto &device : devices) {
      std::cout << "- " << device.device_type << " : " << device.name
                << " (Memory: " << device.memory_limit / (1024 * 1024) << " MB)\n";
    }
  }

  // ======================================================================================
  // 5. Performance Optimization with XLA
  // ======================================================================================
  {
    std::cout << "\n=== 5. XLA Acceleration ===" << "\n";
    txeo::Predictor<float> predictor("path/to/saved_model");

    predictor.enable_xla(true); // Enable accelerated execution
    txeo::Tensor<float> input({1, 256, 256, 3});
    input.fill(0.1f);

    try {
      auto output = predictor.predict(input);
      std::cout << "XLA-accelerated inference completed successfully\n";
    } catch (const txeo::PredictorError &e) {
      std::cerr << "XLA inference failed: " << e.what() << "\n";
    }
  }

  // ======================================================================================
  // 6. Error Handling Demonstration
  // ======================================================================================
  {
    std::cout << "\n=== 6. Error Handling ===" << "\n";
    try {
      // Attempt to load invalid model
      txeo::Predictor<float> predictor("invalid/model/path");
    } catch (const txeo::PredictorError &e) {
      std::cerr << "Caught model loading error: " << e.what() << "\n";
    }

    try {
      // Valid model but invalid input
      txeo::Predictor<float> predictor("path/to/saved_model");
      txeo::Tensor<float> bad_input({1, 100}); // Incorrect shape
      auto output = predictor.predict(bad_input);
    } catch (const txeo::PredictorError &e) {
      std::cerr << "Caught inference error: " << e.what() << "\n";
    }
  }

  return 0;
}