# TensorPart

## Overview

`txeo::TensorPart` is a utility class that provides static methods to **partition and manipulate tensors and matrices**. It is especially useful for preprocessing operations like slicing, unstacking, or extracting submatrices.


## Template Parameter

- `T`: Data type of tensor or matrix elements (e.g., `int`, `float`, `double`)

## Methods

### `unstack(tensor, axis)`

Unstacks a tensor along the specified axis.

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

---

### `slice(tensor, first_axis_begin, first_axis_end)`

Returns a slice of the tensor along its first axis (no copying).

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

---

### `increase_dimension(tensor, axis, value)`

Returns a new tensor with an inserted dimension filled with a specific value.

```cpp
txeo::Tensor<T> increase_dimension(const txeo::Tensor<T>& tensor, size_t axis, T value);
```

---

### `increase_dimension_by(tensor, axis, value)`

Modifies the tensor **in-place**, inserting a new dimension.

```cpp
txeo::Tensor<T>& increase_dimension_by(txeo::Tensor<T>& tensor, size_t axis, T value);
```

---

### `sub_matrix_cols(matrix, cols)`

Returns a submatrix with only the selected columns.

```cpp
txeo::Matrix<T> sub_matrix_cols(const txeo::Matrix<T>& matrix, const std::vector<size_t>& cols);
```

---

### `sub_matrix_cols_exclude(matrix, cols)`

Returns a submatrix **excluding** specified columns.

```cpp
txeo::Matrix<T> sub_matrix_cols_exclude(const txeo::Matrix<T>& matrix, const std::vector<size_t>& cols);
```

---

### `sub_matrix_rows(matrix, rows)`

Returns a submatrix with the specified rows.

```cpp
txeo::Matrix<T> sub_matrix_rows(const txeo::Matrix<T>& matrix, const std::vector<size_t>& rows);
```

---

## Exceptions

### `TensorPartError`

Thrown when a tensor or matrix operation fails.

```cpp
class TensorPartError : public std::runtime_error;
```

---

## Example: Unstacking

```cpp
txeo::Tensor<int> t({{{1,2,3}, {4,5,6}}, {{7,8,9}, {10,11,12}}});
auto slices = txeo::TensorPart<int>::unstack(t, 0);
```

## Example: Column Submatrix

```cpp
txeo::Matrix<double> m(2, 3, {1.1, 2.2, 3.3, 4.4, 5.5, 6.6});
auto sub = txeo::TensorPart<double>::sub_matrix_cols(m, {0, 2});
```

For detailed API references, see individual method documentation at [txeo::TensorPart](https://txeo-doc.netlify.app/classtxeo_1_1_tensor_part.html).
