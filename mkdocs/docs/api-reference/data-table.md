# DataTable

A C++ template class in the **txeo** namespace designed to handle and organize datasets for machine learning workflows. It supports splitting data into training, evaluation, and test sets.

## Overview

`DataTable<T>` is a data container for supervised learning scenarios. It offers flexibility for defining feature and label columns and supports optional evaluation and test splits.

---

## Template Parameters

- `T`: Numeric type (e.g., `float`, `double`) used in the underlying `Matrix<T>`.

---

## Constructors

### DataTable with X/Y columns

```cpp
DataTable(const Matrix<T>& data, std::vector<size_t> x_cols, std::vector<size_t> y_cols);
```

Split based on specified feature and label columns.

### DataTable with only Y columns (auto-infer X columns)

```cpp
DataTable(const Matrix<T>& data, std::vector<size_t> y_cols);
```

All columns not in `y_cols` are considered feature columns.

### DataTable with evaluation split

```cpp
DataTable(const Matrix<T>& data, std::vector<size_t> x_cols, std::vector<size_t> y_cols,
          size_t eval_percent);

DataTable(const Matrix<T>& data, std::vector<size_t> y_cols, size_t eval_percent);
```

Reserves a percentage of the data for evaluation.

### DataTable with evaluation and test splits

```cpp
DataTable(const Matrix<T>& data, std::vector<size_t> x_cols, std::vector<size_t> y_cols,
          size_t eval_percent, size_t eval_test);

DataTable(const Matrix<T>& data, std::vector<size_t> y_cols, size_t eval_percent,
          size_t eval_test);
```

Splits dataset into training, evaluation, and test.

### DataTable with explicit splits

```cpp
DataTable(const Matrix<T>& x_train, const Matrix<T>& y_train,
          const Matrix<T>& x_eval, const Matrix<T>& y_eval,
          const Matrix<T>& x_test, const Matrix<T>& y_test);

DataTable(const Matrix<T>& x_train, const Matrix<T>& y_train,
          const Matrix<T>& x_eval, const Matrix<T>& y_eval);

DataTable(const Matrix<T>& x_train, const Matrix<T>& y_train);
```

Use pre-split matrices directly. If rvalues are passed, copy is avoided.

---

## Accessors

### Training Data

```cpp
const Matrix<T>& x_train() const;
const Matrix<T>& y_train() const;
```

### Evaluation Data

```cpp
const Matrix<T>* x_eval() const;
const Matrix<T>* y_eval() const;
```

Returns nullptr if evaluation was not set.

### Test Data

```cpp
const Matrix<T>* x_test() const;
const Matrix<T>* y_test() const;
```

Returns nullptr if test was not set.

---

## Metadata

### Input/Output Dimensions

```cpp
size_t x_dim() const;
size_t y_dim() const;
```

### Row Count

```cpp
size_t row_size() const;
```

Number of rows in the training set.

---

## Exceptions

### `txeo::DataTableError`

Thrown when invalid inputs or split percentages are provided.

---

## Example Usage

```cpp
txeo::Matrix<float> data = {{1, 2, 3, 4}, {5, 6, 7, 8}};
DataTable<float> dt(data, {3}, 50); // 50% eval split

assert(dt.x_train().rows() == 1);
assert(dt.x_eval()->rows() == 1);
```

For detailed API references, see individual method documentation at [txeo::DataTable](https://txeo-doc.netlify.app/classtxeo_1_1_data_table.html).