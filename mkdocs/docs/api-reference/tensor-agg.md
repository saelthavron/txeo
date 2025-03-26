# TensorAgg

The `TensorAgg` class of **txeo** library provides aggregation functions to simplify statistical and reduction operations on tensors in the **txeo** library.

---

## Overview

`TensorAgg` provides methods for computing sums, products, statistical measures (mean, variance, median), norms, and logical reductions along specified tensor axes.

---

## API Reference

| Method                           | Description                                      |
|----------------------------------|--------------------------------------------------|
| `arg_max(tensor, axis)`          | Indices of max values along axis                 |
| `arg_min(tensor, axis)`          | Indices of min values along axis                 |
| `count_non_zero(tensor, axis)`   | Counts non-zero elements along axis              |
| `cumulative_prod(tensor, axis)`  | Computes cumulative product along axis           |
| `cumulative_sum(tensor, axis)`   | Computes cumulative sum along axis               |
| `reduce_all(tensor, axes)`       | Computes logical AND along axes                  |
| `reduce_any(tensor, axes)`       | Computes logical OR along axes                   |
| `reduce_euclidean_norm(tensor, axes)` | Computes Euclidean norm along axes             |
| `reduce_geometric_mean(tensor, axis)` | Computes geometric mean along axis            |
| `reduce_max(tensor, axes)`       | Finds maximum values along axes                  |
| `reduce_maximum_norm(tensor, axis)`   | Computes maximum norm along axis               |
| `reduce_mean(tensor, axes)`      | Computes mean along axes                         |
| `reduce_median(tensor, axis)`    | Computes median along axis                       |
| `reduce_min(tensor, axes)`       | Finds minimum values along axes                  |
| `reduce_prod(tensor, axes)`      | Computes product along axes                      |
| `reduce_standard_deviation(tensor, axis)` | Computes standard deviation along axis      |
| `reduce_sum(tensor, axes)`       | Computes sum along axes                          |
| `reduce_variance(tensor, axis)`  | Computes variance along axis                     |
| `sum_all(tensor)`                | Sums all elements in tensor                      |

---

## Aggregation Operations

### Reduction Operations

Compute sum, product, mean, max, min along specified axes:

```cpp
#include "txeo/Tensor.h"
#include "txeo/TensorAgg.h"

int main() {
    txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});

    auto sum_result = txeo::TensorAgg<int>::reduce_sum(tensor, {1}); // [6, 15]
    auto prod_result = txeo::TensorAgg<int>::reduce_prod(tensor, {1}); // [6, 120]
    auto mean_result = txeo::TensorAgg<int>::reduce_mean(tensor, {1}); // [2, 5]
    auto max_result = txeo::TensorAgg<int>::reduce_max(tensor, {1}); // [3, 6]
    auto min_result = txeo::TensorAgg<int>::reduce_min(tensor, {1}); // [1, 4]
}
```

### Norm Operations

Compute various norms along specified axes:

```cpp
// Euclidean norm
auto norm_result = txeo::TensorAgg<double>::reduce_euclidean_norm(tensor, {1}); // [3.74166, 8.77496]

// Maximum norm
auto max_norm_result = txeo::TensorAgg<int>::reduce_maximum_norm(tensor, 1); // [3, 6]
```

### Statistical Operations

Calculate variance, standard deviation, median, and geometric mean:

```cpp
auto var_result = txeo::TensorAgg<double>::reduce_variance(tensor, 1); // [1.0, 1.0]
auto std_result = txeo::TensorAgg<double>::reduce_standard_deviation(tensor, 1); // [1.0, 1.0]
auto median_result = txeo::TensorAgg<int>::reduce_median(tensor, 1); // [2, 5]
auto geo_mean_result = txeo::TensorAgg<double>::reduce_geometric_mean(tensor, 1); // [1.81712, 4.93242]
```

### Logical Operations

Perform logical reductions (`all`, `any`):

```cpp
txeo::Tensor<bool> logical_tensor({2, 3}, {true, false, true, true, true, false});
auto all_result = txeo::TensorAgg<bool>::reduce_all(logical_tensor, {1}); // [false, false]
auto any_result = txeo::TensorAgg<bool>::reduce_any(logical_tensor, {1}); // [true, true]
```

### Cumulative Operations

Calculate cumulative sums and products:

```cpp
auto cum_sum_result = txeo::TensorAgg<int>::cumulative_sum(tensor, 1); // [[1,3,6],[4,9,15]]
auto cum_prod_result = txeo::TensorAgg<int>::cumulative_prod(tensor, 1); // [[1,2,6],[4,20,120]]
```

### Argmax and Argmin

Identify indices of max/min elements:

```cpp
auto argmax_result = txeo::TensorAgg<int>::arg_max(tensor, 1); // [2, 2]
auto argmin_result = txeo::TensorAgg<int>::arg_min(tensor, 1); // [0, 0]
```

### Count Operations

Count non-zero elements:

```cpp
auto count_result = txeo::TensorAgg<int>::count_non_zero(tensor, 1); // [3, 3]
```

### Global Sum

Compute sum of all tensor elements:

```cpp
auto total_sum = txeo::TensorAgg<int>::sum_all(tensor); // 21
```

---

For detailed API references, see individual method documentation at [txeo::TensorAgg](https://txeo-doc.netlify.app/classtxeo_1_1_tensor_agg.html).
