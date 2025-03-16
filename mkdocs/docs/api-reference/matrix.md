# Matrix

The `Matrix` class is a specialized 2nd-order tensor provided by the **txeo** library, facilitating intuitive handling of matrix-specific operations.

## Overview

The `Matrix` class in **txeo** is a specialized tensor explicitly tailored for second-order data structures. It provides simplified constructors, clear interfaces, and seamless interoperability with tensors.

---

## API Reference

| Method                   | Description                                                |
|---------------------------|-------------------------------------------------------------|
| `size()`                  | Returns total number of matrix elements                      |
| `to_matrix(tensor)`   | Converts a second-order tensor to matrix (move semantics)  |
| `to_tensor(matrix)`  | Converts a matrix to tensor, supports copy and move semantics |
| `normalize_columns(type)` | Normalize all the columns of this matrix according to a normalization type |
| `normalize_rows(type)` | Normalize all the rows of this matrix according to a normalization type |

---

## Creating Matrices

### Basic Matrix Creation

```cpp
#include <iostream>
#include "txeo/Matrix.h"

int main() {
    txeo::Matrix<int> matrix(3, 3); // Creates a 3x3 matrix
    std::cout << matrix << std::endl;
}
```

### Initialization with Specific Values

```cpp
txeo::Matrix<int> matrix(2, 3, 5);  // 2x3 matrix filled with value 5
```

### Initialization from Vector

```cpp
txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});
```

### Nested Initializer Lists

```cpp
txeo::Matrix<int> matrix{{1, 2, 3}, {4, 5, 6}};  // 2x3 matrix
```

---

## Normalization

```cpp
// Example: Min-max normalize columns
txeo::Matrix<double> mat({{10.0, 20.0},  // Column 1: [10, 30]
                         {30.0, 40.0}}); // Column 2: [20, 40]
mat.normalize_columns(txeo::NormalizationType::MIN_MAX);
// Column 1 becomes [0.0, 1.0]
// Column 2 becomes [0.0, 1.0]

// Example: Z-score normalize columns
txeo::Matrix<float> m({{1.0f, 4.0f},   // Column 1: μ=2.0, σ=1.414
                      {3.0f, 6.0f}});  // Column 2: μ=5.0, σ=1.414
m.normalize_columns(txeo::NormalizationType::Z_SCORE);
// Column 1 becomes [-0.707, 0.707]
// Column 2 becomes [-0.707, 0.707]
```

---

## Conversion between Matrix and Tensor

### Matrix to Tensor (Move Constructor)

```cpp
txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
txeo::Matrix<int> matrix(std::move(tensor));
```

### Tensor to Matrix (Move Semantics)

```cpp
txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
auto matrix = txeo::Matrix<int>::to_matrix(std::move(tensor));
```

### Matrix to Tensor (Copy)

```cpp
txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});
txeo::Tensor<int> tensor = txeo::Matrix<int>::to_tensor(matrix);
```

### Matrix to Tensor (Move Semantics)

```cpp
txeo::Tensor<int> tensor = txeo::Matrix<int>::to_tensor(std::move(matrix));
```

---

## Exception Handling

Matrix-related errors throw `MatrixError` exceptions:

```cpp
try {
    txeo::Matrix<int> matrix(2, 3, {1,2,3});  // Invalid initialization
} catch (const txeo::MatrixError &e) {
    std::cerr << e.what() << std::endl;
}
```

---

## Examples

### Basic Matrix Manipulation

```cpp
#include <iostream>
#include "txeo/Tensor.h"
#include "txeo/Matrix.h"

int main() {
    txeo::Matrix<int> mat{{1, 2}, {3, 4}};
    mat(0, 1) = 10;
    std::cout << "Matrix:\n" << mat << std::endl;
}
```

### Matrix and Scalar Operations

```cpp
#include <iostream>
#include "txeo/Matrix.h"

int main() {
    txeo::Matrix<int> mat(2, 2, {1, 2, 3, 4});
    auto mat2 = mat * 3;  // Scalar multiplication

    std::cout << "Resulting matrix: " << mat2 << std::endl;
    return 0;
}
```

---

For detailed API references, see individual method documentation at [txeo::Matrix](https://txeo-doc.netlify.app/classtxeo_1_1_matrix.html).
