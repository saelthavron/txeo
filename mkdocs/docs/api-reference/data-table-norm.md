
# DataTableNorm

A normalizer for `DataTable` objects that handles feature scaling, supporting both **Min-Max** and **Z-Score** normalization.

## Overview

The `DataTableNorm` class provides feature normalization capabilities for machine learning datasets stored in `DataTable` format. It supports two normalization techniques:

- **Min-Max Scaling**: Transforms features to [0, 1] range
- **Z-Score Standardization**: Transforms features to mean=0, std=1

**Key Features**:

- Computes parameters from training data
- Handles both in-place and copied normalization
- Supports efficient rvalue operations
- Works with evaluation/test splits

---

## Constructor

### `DataTableNorm(const DataTable<T> &data, NormalizationType type = MIN_MAX)`

Initializes the normalizer from a data table.

**Example:**

```cpp
txeo::DataTable<double> data = load_my_dataset();
txeo::DataTableNorm<double> normalizer(data, txeo::NormalizationType::Z_SCORE);
```

---

## Member Functions

### `const DataTable<T>& data_table() const`

Returns the internal reference to the associated `DataTable`.

**Example:**

```cpp
const auto& dt = normalizer.data_table();
std::cout << dt.x_train().rows() << std::endl;
```

### `void set_data_table(const DataTable<T>& data)`

Sets a new data table for normalization.

**Example:**

```cpp
txeo::DataTable<double> new_data = load_updated_dataset();
normalizer.set_data_table(new_data);
```

### `NormalizationType type() const`

Returns the type of normalization currently used.

**Example:**

```cpp
if (normalizer.type() == txeo::NormalizationType::MIN_MAX)
    std::cout << "Using Min-Max normalization" << std::endl;
```

### `Matrix<T> normalize(Matrix<T>&& x) const`

Normalizes a matrix **in-place** using rvalue semantics.

**Example:**

```cpp
txeo::Matrix<double> large_matrix = generate_large_data();
auto normalized = normalizer.normalize(std::move(large_matrix));
```

### `Matrix<T> normalize(const Matrix<T>& x) const`

Normalizes a matrix **by copy**.

**Example:**

```cpp
txeo::Matrix<double> original = {{1.0}, {2.0}, {3.0}};
auto normalized = normalizer.normalize(original);
```

### `Matrix<T> x_train_normalized()`

Returns normalized training data.

**Example:**

```cpp
auto x_train_norm = normalizer.x_train_normalized();
model.train(x_train_norm, normalizer.data_table().y_train());
```

### `Matrix<T> x_eval_normalized()`

Returns normalized evaluation data.

**Example:**

```cpp
auto x_eval_norm = normalizer.x_eval_normalized();
model.evaluate(x_eval_norm, normalizer.data_table().y_eval());
```

### `Matrix<T> x_test_normalized()`

Returns normalized test data.

**Example:**

```cpp
auto x_test_norm = normalizer.x_test_normalized();
model.test(x_test_norm, normalizer.data_table().y_test());
```

## Exceptions

- `DataTableNormError`: Thrown if normalization parameters are invalid or data table is inconsistent.

---

## Notes

- Normalization parameters are computed from training data only.
- Be sure to properly configure your `DataTable` before using `DataTableNorm`.


For detailed API references, see individual method documentation at [txeo::DataTableNorm](https://txeo-doc.netlify.app/classtxeo_1_1_data_table_norm.html).