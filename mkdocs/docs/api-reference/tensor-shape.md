# TensorShape

The `TensorShape` class defines the dimensional structure of tensors in the **txeo** library. It describes the dimensions and axes of tensors, serving as the foundation for tensor creation, manipulation, and indexing.

## Overview

A `TensorShape` represents the dimensions of a tensor:

- **Axes**: Positions labeled from zero, each associated with a tensor dimension.
- **Dimensions**: Size along each axis.

Examples:

- Scalar (no axes): `TensorShape()`
- Vector: `{3}`
- Matrix: `{3, 4}`
- 3-dimensional tensor: `{2, 3, 4}`

---

## API Reference

| Method                      | Description                                     |
|-----------------------------|-------------------------------------------------|
| `size()`                    | Returns number of axes                          |
| `number_of_axes()`          | Synonym for `size()`                            |
| `axes_dims()`               | Returns dimensions of each axis                 |
| `stride()`                  | Returns strides for efficient indexing          |
| `set_dim(axis, dim)`        | Changes size of specified axis                  |
| `insert_axis(axis, dim)`    | Inserts axis at specified position              |
| `push_axis_back(dim)`       | Adds an axis at the end                         |
| `remove_axis(axis)`         | Removes specified axis                          |
| `remove_all_axes()`         | Removes all axes                                |
| `clone()`                   | Returns a deep copy of shape                    |

---

## Creating Tensor Shapes

### Constructing from dimensions

```cpp
**txeo**::TensorShape shape({2, 3, 4}); // Creates a shape with dimensions 2x3x4
```

### Creating uniform dimensions

```cpp
**txeo**::TensorShape shape(3, 5); // Shape with three axes, each of dimension 5
```

## Common Operations

### Accessing Dimensions

You can access the shape dimensions:

```cpp
std::vector<int64_t> dims = shape.axes_dims();
```

### Comparing shapes

```cpp
if (shape1 == shape2) {
    std::cout << "Shapes are equal.";
}
```

## Manipulating Axes

### Inserting an Axis

```cpp
shape.insert_axis(1, 5); // Inserts a new axis at position 1 with dimension 5
```

### Removing an Axis

```cpp
shape.remove_axis(2); // Removes the axis at position 2
```

### Changing a Dimension

```cpp
shape.set_dim(0, 10); // Sets the first axis dimension size to 10
```

### Removing All Axes

```cpp
shape.remove_axis(0); // Removes axis at position 0
shape.remove_axis(1); // Removes axis at new position 1
```

or simply:

```cpp
shape.remove_all_axes();
```

## Stride

Tensor strides represent the memory step size for each dimension:

```cpp
auto strides = shape.stride();

for (size_t s : strides)
    std::cout << s << ' ';
```

---

## Examples

### Checking Shape Equality

```cpp
#include <iostream>
#include "**txeo**/TensorShape.h"

int main() {
    **txeo**::TensorShape shape1({3, 4});
    **txeo**::TensorShape shape2({3, 4});

    if (shape == shape2) {
        std::cout << "Shapes match!" << std::endl;
    }
}
```

### Modifying a TensorShape

```cpp
#include <iostream>
#include "**txeo**/TensorShape.h"

int main() {
    **txeo**::TensorShape shape({2, 3});

    shape.push_axis_back(4);       // shape now (2,3,4)
    shape.set_dim(1, 5);            // Update dimension of axis 1 from 3 to 5

    std::cout << "Updated Shape: " << shape << std::endl;
}
```

---

## Exception Handling

All invalid operations throw a clear `TensorShapeError`:

```cpp
try {
    shape.insert_axis(10, 2); // invalid operation
} catch (const **txeo**::TensorShapeError &e) {
    std::cerr << "Error: " << e.what();
}
```

---

For detailed API references, see individual method documentation at [**txeo**::TensorShape](https://**txeo**-doc.netlify.app/class**txeo**_1_1_tensor_shape.html).
