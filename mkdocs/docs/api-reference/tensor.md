# Tensor

**txeo**'s `Tensor` class implements a powerful and intuitive representation of mathematical tensors, supporting operations from basic scalar handling to advanced multidimensional tensor manipulations. This page details the API, usage examples, and fundamental concepts.

---

## Overview

A **tensor** in Txeo generalizes scalars, vectors, and matrices to higher dimensions:

- **0th order:** Scalar (single numeric value)
- **1st order:** Vector
- **2nd order:** Matrix
- **N-th order:** General tensor with N dimensions

Txeo tensors support advanced functionality such as reshaping, slicing, arithmetic operations, and more.

---

## API Reference

| Method                      | Description                                     |
|-----------------------------|-------------------------------------------------|
| `shape()`                   | Returns the shape of the tensor as a `TensorShape` object. |
| `reshape(new_shape)`        | Reshapes the tensor to the specified shape.     |
| `slice(start_indices, sizes)` | Extracts a sub-tensor from the tensor.          |
| `flatten()`                 | Returns a 1D view of the tensor.                |
| `fill(value)`               | Fills the tensor with the specified value.      |
| `operator()(indices...)`    | Accesses or modifies tensor elements using multi-dimensional indexing. |
| `at(indices...)`            | Accesses tensor elements with bounds checking.  |
| `data()`                    | Returns a pointer to the underlying data.       |
| `clone()`                   | Returns a deep copy of the tensor.              |
| `operator+`, `operator-`, `operator*`, `operator/` | Performs element-wise arithmetic operations. |
| `operator+=`, `operator-=`, `operator*=`, `operator/=` | Performs in-place arithmetic operations. |
| `begin()`, `end()`          | Returns iterators for traversing tensor elements. |
| `is_equal_shape(other)`     | Checks if the tensor has the same shape as another tensor. |
| `fill_with_uniform_random(min, max)` | Fills the tensor with uniformly distributed random values. |
| `squeeze()`                 | Removes singleton dimensions from the tensor.   |

---

## Creating Tensors

### Basic Construction

Create a tensor by specifying its shape:

```cpp
#include <iostream>
#include "**txeo**/Tensor.h"

int main() {
    **txeo**::Tensor<int> tensor({3, 4}); // Create a 3x4 tensor
    std::cout << "Shape: " << tensor.shape() << std::endl;
}
```

### Initializing with Values

```cpp
**txeo**::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6}); // 2x3 Tensor with predefined values
```

### Filling with a Specific Value

```cpp
**txeo**::Tensor<float> tensor({2, 2}, 5.0f); // Tensor initialized with 5.0
```

### Nested Initialization

```cpp
**txeo**::Tensor<int> tensor{{1, 2, 3}, {4, 5, 6}}; // 2x3 Tensor
```

---

## Tensor Operations

### Element-wise Arithmetic

```cpp
**txeo**::Tensor<int> a({2,2}, {1,2,3,4});
**txeo**::Tensor<int> b({2,2}, {5,6,7,8});

// Addition
a + b; // [[6, 8], [10, 12]]

// Scalar multiplication
a * 2; // [[2, 4], [6, 8]]
```

### Reshaping

```cpp
**txeo**::Tensor<int> tensor{{1, 2, 3, 4}}; // Shape (1, 4)
tensor.reshape({2, 2}); // Reshape to (2, 2)
```

### Slicing

```cpp
**txeo**::Tensor<int> tensor{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
auto sliced = tensor.slice(0, 2); // First two rows
```

### Random Initialization

```cpp
**txeo**::Tensor<float> tensor({3,3});
tensor.fill_with_uniform_random(0.0f, 1.0f);
```

---

## Accessing Elements

### Direct Access (Unchecked)

```cpp
tensor(1, 2) = 42; // Set value at position (1,2)
```

### Checked Access

```cpp
try {
    tensor.at(10, 5) = 42; // Throws exception if out of bounds
} catch (const **txeo**::TensorError& e) {
    std::cerr << e.what() << std::endl;
}
```

---

## Tensor Information

- **Shape:** `tensor.shape()`
- **Order (Dimensions):** `tensor.order()`
- **Total elements:** `tensor.number_of_elements()`
- **Memory usage:** `tensor.memory_size()` bytes

---

## Iterator Support

```cpp
for (auto &value : tensor) {
    std::cout << value << " ";
}
```

---

## Common Errors

`TensorError` is thrown if operations fail due to inconsistent initialization, index out-of-bounds access, or invalid reshaping.

```cpp
catch (const **txeo**::TensorError& e) {
    std::cerr << "Tensor Error: " << e.what() << std::endl;
}
```

---

## Performance Notes

- **Memory Efficiency:** Txeo avoids unnecessary data copies (e.g., `slice`, `reshape`).
- **Deep Copy Behavior:** Assignment and copy constructors perform deep copies, ensuring independent data between tensors.

---

For detailed API references, see individual method documentation at [**txeo**::Tensor](https://**txeo**-doc.netlify.app/class**txeo**_1_1_tensor.html).
