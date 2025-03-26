# Tensor Functions

The class `TensorFunc` of **txeo** provides common mathematical functions that can be applied element-wise to tensors.

## Overview

`TensorFunc` offers element-wise mathematical functions such as potentiation, square, square root, absolute value, permutation of tensor axes, and matrix transposition.

## API Reference

| Method                      | Description                                                  |
|-----------------------------|--------------------------------------------------------------|
| `abs_by`                    | Computes element-wise absolute value in-place                |
| `abs`                       | Computes element-wise absolute value                         |
| `permute_by`                | Permutes axes of a tensor in-place                           |
| `permute`                   | Permutes axes of a tensor                                    |
| `power_elem_by`             | Computes element-wise power in-place                         |
| `power_elem`                | Computes element-wise power of tensor elements               |
| `sqrt_by`                   | Computes element-wise square root in-place                   |
| `sqrt`                      | Computes element-wise square root                            |
| `square_by`                 | Computes element-wise square in-place                        |
| `square`                    | Computes element-wise square                                 |
| `transpose_by`              | Transposes a matrix in-place                                 |
| `transpose`                 | Transposes a matrix                                          |

---

## Examples

### Element-wise Power

```cpp
txeo::Tensor<float> a({3}, {2.0f, 3.0f, 4.0f});
auto b = TensorFunc<float>::power_elem(a, 2.0f);  // [4.0, 9.0, 16.0]
```

### In-place Element-wise Square

```cpp
txeo::Tensor<int> tensor({3}, {1, 2, 3});
TensorFunc<int>::square_by(tensor);  // tensor becomes [1, 4, 9]
```

### Square Root

```cpp
txeo::Tensor<double> tensor({3}, {1.0, 4.0, 9.0});
auto result = TensorFunc<double>::sqrt(tensor);  // [1.0, 2.0, 3.0]
```

### Absolute Value

```cpp
txeo::Tensor<int> tensor({3}, {-1, 2, -3});
auto result = TensorFunc<int>::abs(tensor);  // [1, 2, 3]
```

### Permute Axes

```cpp
txeo::Tensor<int> tensor({2, 3, 4}, {1, 2, ..., 24});
auto result = TensorFunc<int>::permute(tensor, {1, 2, 0});  // shape: (3, 4, 2)
```

### Normalization

```cpp
enum class NormalizationType { MIN_MAX, Z_SCORE };
```

```cpp
  txeo::Tensor<double> tensor({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  txeo::TensorFunc<double>::normalize_by(tensor, txeo::NormalizationType::MIN_MAX);
  std::cout << tens << std::endl; //  [0 0.125 0.25][0.375 0.5 0.625][0.75 0.875 1]
```

### Matrix Transpose

```cpp
txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});
auto result = TensorFunc<int>::transpose(matrix);  // shape: (3, 2)
```

## Exceptions

`TensorFuncError` is thrown if operations encounter invalid arguments, such as mismatched tensor shapes or invalid axis permutations.

---

For detailed API references, see individual method documentation at [txeo::TensorFunc](https://txeo-doc.netlify.app/classtxeo_1_1_tensor_func.html).
