# Vector

The `Vector` class in **txeo** is a specialized subclass of `txeo::Tensor`, specifically designed for first-order tensors, providing a straightforward interface for vector operations.

---

## Overview

The `Vector` class simplifies handling 1st-order tensor operations, offering intuitive constructors, easy initialization methods, and seamless conversion between vectors and general tensors.

---

## API Reference

| Method                          | Description                                           |
|---------------------------------|-------------------------------------------------------|
| `normalize(type)` | Normalize this vector according to a normalization type |
| `reshape(shape)`                | Changes the shape of the vector                        |
| `to_tensor(vector)`             | Converts a vector to tensor (copy/move)                |
| `to_vector(tensor)`             | Converts a first-order tensor to vector (copy/move)    |

---

## Creating Vectors

### Basic Construction

Create an uninitialized vector by specifying its dimension:

```cpp
#include <iostream>
#include "txeo/Vector.h"

int main() {
    txeo::Vector<int> vector(3); // Creates a vector of size 3
    std::cout << vector << std::endl;
}
```

### Initialization with Specific Values

```cpp
txeo::Vector<int> vector(3, 5); // Creates a vector of size 3 filled with 5
```

### Initialization from Vector

```cpp
txeo::Vector<double> vector(3, {1.0, 2.0, 3.0});
```

### Initialization Using Initializer Lists

```cpp
txeo::Vector<int> vector({1, 2, 3}); // Vector of size 3
```

---

## Conversion between Vector and Tensor

### Tensor to Vector

```cpp
txeo::Tensor<int> tensor({4}, {1, 2, 3, 4});
auto vector = txeo::Vector<int>::to_vector(std::move(tensor));
```

### Vector to Tensor

```cpp
txeo::Vector<int> vector({1, 2, 3, 4});
// Copy semantics
txeo::Tensor<int> tensor = txeo::Vector<int>::to_tensor(vector);
```

```cpp
// Using move semantics
txeo::Tensor<int> tensor = txeo::Vector<int>::to_tensor(std::move(vector));
```

### Vector to Tensor (Move Constructor)

```cpp
txeo::Tensor<int> tensor({1, 2, 3, 4});
txeo::Vector<int> vector(std::move(tensor));
```

---

## Normalization

### Min-Max

```cpp
Vector<double> vec({2.0, 4.0, 6.0});
vec.normalize(txeo::NormalizationType::MIN_MAX);
// vec becomes [0.0, 0.5, 1.0] (original min=2, max=6)
```

### Z-Score

```cpp
Vector<float> v({2.0f, 4.0f, 6.0f});
v.normalize(txeo::NormalizationType::Z_SCORE);
// v becomes approximately [-1.2247, 0.0, 1.2247]
// (μ=4.0, σ≈1.63299)
```

## Exception Handling

Invalid vector operations throw `VectorError` exceptions:

```cpp
try {
    txeo::Tensor<int> tensor({2, 2}, {1, 2, 3, 4});
    // Throws VectorError
    auto vector = txeo::Vector<int>::to_vector(std::move(tensor));
} catch (const txeo::VectorError &e) {
    std::cerr << e.what() << std::endl;
}
```

---

## Examples

### Vector Arithmetic Operations

```cpp
#include <iostream>
#include "txeo/Vector.h"

int main() {
    txeo::Vector<int> vec1({1, 2, 3});
    auto vec2 = vec1 * 2;  // Scalar multiplication

    std::cout << "Resulting vector: " << vec2 << std::endl;
}
```

### Accessing and Modifying Vector Elements

```cpp
vec1(0) = 10;  // Sets the first element to 10
```

---

For detailed API references, see individual method documentation at [txeo::Vector](https://txeo-doc.netlify.app/classtxeo_1_1_vector.html).
