#include "txeo/TensorShape.h"
#include <iostream>

int main() {
  try {
    // --------------------------
    // 1. Construction Examples
    // --------------------------
    std::cout << "=== Construction Examples ===\n";

    // Empty shape (scalar)
    txeo::TensorShape scalar_shape({});
    std::cout << "Scalar shape: " << scalar_shape << "\n";

    // Uniform dimensions
    txeo::TensorShape uniform_shape(3, 5); // 3 axes, each dim 5
    std::cout << "Uniform shape: " << uniform_shape << "\n";

    // From vector
    std::vector<size_t> vec_dims{2, 3, 4};
    txeo::TensorShape vector_shape(vec_dims);
    std::cout << "Vector constructed shape: " << vector_shape << "\n";

    // From initializer list
    txeo::TensorShape init_list_shape({4, 5, 6});
    std::cout << "Initializer list shape: " << init_list_shape << "\n";

    // --------------------------
    // 2. Shape Manipulation
    // --------------------------
    std::cout << "\n=== Shape Manipulation ===\n";

    txeo::TensorShape dynamic_shape({2, 3});
    std::cout << "Original shape: " << dynamic_shape << "\n";

    // Add axis at end
    dynamic_shape.push_axis_back(4);
    std::cout << "After push_back(4): " << dynamic_shape << "\n";

    // Insert axis at position 1
    dynamic_shape.insert_axis(1, 5);
    std::cout << "After insert(1, 5): " << dynamic_shape << "\n";

    // Remove axis at position 2
    dynamic_shape.remove_axis(2);
    std::cout << "After remove(2): " << dynamic_shape << "\n";

    // Modify existing dimension
    dynamic_shape.set_dim(1, 7);
    std::cout << "After set_dim(1,7): " << dynamic_shape << "\n";

    // --------------------------
    // 3. Shape Inspection
    // --------------------------
    std::cout << "\n=== Shape Inspection ===\n";

    const txeo::TensorShape shape({2, 3, 4});
    std::cout << "Inspecting shape: " << shape << "\n";

    // Access individual dimensions
    std::cout << "Axis 0 dim: " << shape.axis_dim(0) << "\n";
    std::cout << "Axis 1 dim: " << shape.axis_dim(1) << "\n";
    std::cout << "Axis 2 dim: " << shape.axis_dim(2) << "\n";

    // Get all dimensions
    auto dims = shape.axes_dims();
    std::cout << "All dimensions: ";
    for (auto d : dims)
      std::cout << d << " ";
    std::cout << "\n";

    // Memory stride calculation
    auto strides = shape.stride();
    std::cout << "Memory strides: ";
    for (auto s : strides)
      std::cout << s << " ";
    std::cout << "\n";

    // --------------------------
    // 4. Advanced Operations
    // --------------------------
    std::cout << "\n=== Advanced Operations ===\n";

    // Capacity calculation
    std::cout << "Total elements: " << shape.calculate_capacity() << "\n";

    // Cloning
    auto cloned_shape = shape.clone();
    cloned_shape.set_dim(0, 5);
    std::cout << "Original: " << shape << "\n";
    std::cout << "Modified clone: " << cloned_shape << "\n";

    // Validity checks
    std::cout << "Is fully defined? " << std::boolalpha << shape.is_fully_defined() << "\n";

    // --------------------------
    // 5. Error Handling
    // --------------------------
    std::cout << "\n=== Error Handling ===\n";

    try {
      txeo::TensorShape invalid_shape(0, 5); // 0 axes
    } catch (const txeo::TensorShapeError &e) {
      std::cout << "Caught error: " << e.what() << "\n";
    }

    try {
      auto aux = dynamic_shape.axis_dim(10); // Invalid axis
      std::cout << aux << std::endl;
    } catch (const txeo::TensorShapeError &e) {
      std::cout << "Caught error: " << e.what() << "\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}