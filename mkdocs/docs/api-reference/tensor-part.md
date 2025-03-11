# TensorPart

The `TensorPart` class of **txeo** library provides static methods for partitioning and slicing tensors. It enables operations such as unstacking a tensor along a specified axis and creating views (slices) without copying data.

## Usage

Include the header:

```cpp
#include "txeo/TensorPart.h"
```

## Methods

### Unstacking Tensors

```cpp
static std::vector<txeo::Tensor<T>> unstack(const txeo::Tensor<T> &tensor, size_t axis);
```

Splits a tensor along a specified axis into multiple tensors.

#### Parameters

- **tensor**: The tensor to be unstacked.
- **axis**: The axis along which to perform the unstack operation.

#### Returns

A `std::vector` containing tensors resulting from the unstack operation.

#### Example

```cpp
#include "txeo/TensorPart.h"
#include "txeo/Tensor.h"
#include <iostream>

int main() {
    txeo::Tensor<int> tensor({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});

    auto unstacked_tensors = txeo::TensorPart<int>::unstack(tensor, 0);

    for (size_t i = 0; i < unstacked_tensors.size(); ++i) {
        std::cout << "Unstacked Tensor " << i << ":\n" << unstacked_tensors[i] << std::endl;
    }

    return 0;
}
```

#### Output

```bash
Unstacked Tensor 0:
[[1 2 3]
 [4 5 6]]

Unstacked Tensor 1:
[[7 8 9]
 [10 11 12]]
```

### Slicing Tensors

```cpp
static txeo::Tensor<T> slice(const txeo::Tensor<T> &tensor, size_t first_axis_begin, size_t first_axis_end);
```

Creates a view of the tensor from a specified range along the first axis without copying data.

#### Parameters

- **tensor**: The tensor to slice.
- **first_axis_begin**: The start index (inclusive) along the first axis.
- **first_axis_end**: The end index (exclusive) along the first axis.

#### Returns

A tensor view from the specified indices.

#### Example

```cpp
#include <iostream>
#include "txeo/Tensor.h"
#include "txeo/TensorPart.h"

int main() {
    txeo::Tensor<int> tensor{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto sliced_tensor = txeo::TensorPart<int>::slice(tensor, 0, 2);

    std::cout << "Sliced Tensor: " << sliced_tensor << std::endl;
    return 0;
}
```

#### Output

```bash
Sliced Tensor: {{1, 2}, {4, 5}, {7, 8}}
```

## Exceptions

### `TensorPartError`

An exception thrown if invalid arguments are provided or if an operation fails.

## Notes

- Operations like `slice` do not copy data but create views into the original tensor, improving efficiency.

---

For detailed API references, see individual method documentation at [txeo::TensorPart](https://txeo-doc.netlify.app/classtxeo_1_1_tensor_part.html).
