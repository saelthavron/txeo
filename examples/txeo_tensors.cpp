#include "txeo/Tensor.h"
#include <iostream>

int main() {
  // ============================================================================================
  // 1. Basic Tensor Creation & Initialization
  // ============================================================================================
  {
    std::cout << "\n=== 1. Basic Tensor Creation ===\n";

    // Create 3x4 tensor filled with 5
    txeo::Tensor<int> filled_tensor({3, 4}, 5);
    std::cout << "Filled Tensor:\n" << filled_tensor << "\n";

    // Create 2x3 tensor from nested initializer list
    txeo::Tensor<float> matrix{{1.1f, 2.2f, 3.3f}, {4.4f, 5.5f, 6.6f}};
    std::cout << "\nMatrix Tensor:\n" << matrix << "\n";

    // 3D tensor from triple nested initializer
    txeo::Tensor<double> cube{{{1.1, 2.2}, {3.3, 4.4}}, {{5.5, 6.6}, {7.7, 8.8}}};
    std::cout << "\n3D Tensor:\n" << cube << std::endl;
  }

  // ============================================================================================
  // 2. Element Access & Modification
  // ============================================================================================
  {
    std::cout << "\n\n=== 2. Element Access ===\n";
    txeo::Tensor<int> tensor{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // Safe access with bounds checking
    std::cout << "Element at (1,2): " << tensor.at(1, 2) << "\n";

    // Unsafe but faster access
    tensor(2, 1) = 42;
    std::cout << "\nModified Tensor:\n" << tensor << "\n";

    try {
      std::cout << tensor.at(3, 0); // Throws TensorError
    } catch (const txeo::TensorError &e) {
      std::cerr << "\nError: " << e.what() << std::endl;
    }
  }

  // ============================================================================================
  // 3. Tensor Operations & Transformations
  // ============================================================================================
  {
    std::cout << "\n\n=== 3. Tensor Operations ===\n";
    txeo::Tensor<float> original{{1, 2}, {3, 4}, {5, 6}};

    // Slice first two rows
    auto sliced = original.slice(0, 2);
    std::cout << "Sliced Tensor:\n" << sliced << "\n";

    // Reshape to 2x3
    sliced.reshape({2, 2});
    std::cout << "\nReshaped Tensor:\n" << sliced << "\n";

    // Flatten to 1D
    auto flat = sliced.flatten();
    std::cout << "\nFlattened Tensor:\n" << flat << std::endl;
  }

  // ============================================================================================
  // 4. Batch Processing with TensorIterator
  // ============================================================================================
  {
    std::cout << "\n\n=== 4. Batch Processing ===\n";
    txeo::Tensor<float> batch{
        {{1, 2}, {3, 4}}, // Sample 1
        {{5, 6}, {7, 8}}  // Sample 2
    };

    // Process using range-based for loop
    std::cout << "Batch Values: ";
    for (const auto &val : batch) {
      std::cout << val << " ";
    }

    // Direct data access
    float *data = batch.data();
    std::cout << "\nFirst element: " << *data << std::endl;
  }

  // ============================================================================================
  // 5. Advanced Initialization & Utilities
  // ============================================================================================
  {
    std::cout << "\n\n=== 5. Advanced Initialization ===\n";
    // From vector with explicit shape
    std::vector<double> values = {1.1, 2.2, 3.3, 4.4};
    txeo::Tensor<double> vector_tensor({4}, values);
    std::cout << "Vector Tensor:\n" << vector_tensor << "\n";

    // Random initialization
    txeo::Tensor<float> random_tensor({3, 3});
    random_tensor.fill_with_uniform_random(0.0f, 1.0f, 42, 123);
    std::cout << "\nRandom Tensor:\n" << random_tensor << "\n";

    // Squeeze singleton dimension
    txeo::Tensor<int> squeezed({{1}, {2}, {3}});
    squeezed.squeeze();
    std::cout << "\nSqueezed Tensor:\n" << squeezed << std::endl;
  }

  // ============================================================================================
  // 6. Tensor Comparison & Shape Operations
  // ============================================================================================
  {
    std::cout << "\n\n=== 6. Tensor Comparison ===\n";
    txeo::Tensor<int> a{{1, 2}, {3, 4}};
    txeo::Tensor<int> b{{5, 6}, {7, 8}};

    // Shape comparison
    if (a.is_equal_shape(b)) {
      std::cout << "Tensors have matching shapes\n";
    }

    // Deep copy demonstration
    auto c = a.clone();
    std::cout << "\nOriginal Tensor:\n" << a << "\n";
    std::cout << "Clone created:\n" << c << "\n";

    // In-place modification
    c.fill(0);
    std::cout << "\nModified clone:\n" << c << std::endl;
  }

  return 0;
}