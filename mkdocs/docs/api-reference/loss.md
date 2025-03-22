/**
 * @example loss_example.cpp
 * @brief Demonstration of Loss class functionality
 * 
 * Full example code:
 * @code
 * #include "txeo/Loss.h"
 * 
 * int main() {
 *     // Create validation data
 *     txeo::Tensor<float> valid({4}, {1.5f, 2.0f, 3.2f, 4.8f});
 *     
 *     // Initialize loss calculator with default (MSE)
 *     txeo::Loss<float> loss(valid);
 *     
 *     // Generate predictions
 *     txeo::Tensor<float> pred({4}, {1.6f, 1.9f, 3.0f, 5.0f});
 *     
 *     // Calculate and compare different losses
 *     std::cout << "MSE: " << loss.get_loss(pred) << std::endl;
 *     std::cout << "Direct MAE: " << loss.mae(pred) << std::endl;
 *     
 *     // Switch to MSLE and calculate
 *     loss.set_loss(txeo::LossFunc::MSLE);
 *     std::cout << "MSLE: " << loss.get_loss(pred) << std::endl;
 *     
 *     return 0;
 * }
 * @endcode
 */