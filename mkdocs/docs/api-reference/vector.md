# Vector

The `Vector` class in **txeo** is a specialized subclass of `txeo::Tensor`, specifically designed for first-order (1-dimensional) tensors, providing a straightforward interface for vector operations.

---

## Overview

The `Vector` class simplifies handling 1-dimensional tensor operations, offering intuitive constructors, easy initialization methods, and seamless conversion between vectors and general tensors.

---

## API Reference

| Method                          | Description                                           |
|---------------------------------|-------------------------------------------------------|
| `reshape(shape)`                | Changes the shape of the vector                        |
| `to_vector(tensor)`             | Converts a first-order tensor to vector (copy/move)    |
| `to_tensor(vector)`             | Converts a vector to tensor (copy/move)                |

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
txeo::Tensor<int> tensor = txeo::Vector<int>::to_tensor(vector); // Copy semantics
```

```cpp
// Using move semantics
txeo::Tensor<int> tensor = txeo::Vector<int>::to_tensor(std::move(vector));
```

---

## Exception Handling

Invalid vector operations throw `VectorError` exceptions:

```cpp
try {
    txeo::Tensor<int> tensor({2, 2}, {1, 2, 3, 4});
    auto vector = txeo::Vector<int>::to_vector(std::move(tensor)); // Throws VectorError
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
