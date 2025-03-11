# TensorOp

The `TensorOp` class provides utility methods for performing common mathematical operations on tensors, vectors, and matrices in the **txeo** library.

---

## Overview

`TensorOp` offers a collection of static functions, including arithmetic operations, scalar operations, and higher-level linear algebra functions, specifically tailored to simplify tensor computations.

---

## API Reference

| Method                           | Description                                               |
|----------------------------------|-----------------------------------------------------------|
| `sum(tensor1, tensor2)`          | Element-wise sum                                          |
| `sum(tensor, scalar)`            | Adds scalar to each tensor element                        |
| `subtract(tensor1, tensor2)`     | Element-wise subtraction                                  |
| `multiply(tensor, scalar)`       | Scalar multiplication                                     |
| `divide(tensor, scalar)`         | Scalar division                                           |
| `hadamard_prod(tensor1, tensor2)`| Element-wise multiplication                              |
| `product(matrix1, matrix2)`      | Matrix multiplication                                     |
| `dot(vector1, vector2)`          | Computes the dot product of two vectors                   |

---

## Arithmetic Operations

### Sum

Sum two tensors element-wise:

```cpp
#include "txeo/Tensor.h"
#include "txeo/TensorOp.h"

int main() {
    txeo::Tensor<int> a({2,2}, {1,2,3,4});
    txeo::Tensor<int> b({2,2}, {5,6,7,8});

    auto result = txeo::TensorOp<int>::sum(a, b);
    // result: [6, 8, 10, 12]
}
```

### Subtraction Operations

Element-wise subtraction:

```cpp
txeo::Tensor<int> result = txeo::TensorOp<int>::subtract(a, b);
```

### Scalar Operations

- **Addition:** `TensorOp::sum(tensor, scalar)`
- **Subtraction:** `TensorOp::subtract(tensor, scalar)`
- **Multiplication:** `TensorOp::multiply(tensor, scalar)`
- **Division:** `TensorOp::divide(tensor, scalar)`

Example:

```cpp
txeo::Tensor<float> tensor({3}, {1.0f, 2.0f, 3.0f});
auto result = txeo::TensorOp<float>::multiply(tensor, 2.0f); // [2.0, 4.0, 6.0]
```

### Hadamard Product (Element-wise multiplication)

```cpp
txeo::Tensor<float> result = txeo::TensorOp<float>::hadamard_prod(tensor1, tensor2);
```

---

## Matrix Operations

### Matrix Multiplication

```cpp
txeo::Matrix<int> mat1(2, 3, {1, 2, 3, 4, 5, 6});
txeo::Matrix<int> mat2(3, 2, {7, 8, 9, 10, 11, 12});

auto result = txeo::TensorOp<int>::product(mat1, mat2); // [[58, 64], [139, 154]]
```

### Vector Dot Product

```cpp
txeo::Vector<int> vec1({1, 2, 3});
txeo::Vector<int> vec2({4, 5, 6});

int dot_product = txeo::TensorOp<int>::dot(vec1, vec2); // 32
```

---

## Exception Handling

Operations that violate tensor shape constraints throw `TensorOpError`:

```cpp
try {
    auto result = txeo::TensorOp<int>::sum(tensor1, incompatible_tensor);
} catch (const txeo::TensorOpError &e) {
    std::cerr << e.what() << std::endl;
}
```

---

## Examples

### Tensor Arithmetic

```cpp
#include "txeo/Tensor.h"
#include "txeo/TensorOp.h"

int main() {
    txeo::Tensor<double> t1({2}, {1.5, 2.5});
    auto result = txeo::TensorOp<double>::sum(t1, 4.5);
    std::cout << result << std::endl;
}
```

---

For detailed API references, see individual method documentation at [txeo::TensorOp](https://txeo-doc.netlify.app/classtxeo_1_1_tensor_op.html).
