# OlsGDTrainer

## Overview

`txeo::OlsGDTrainer` is a concrete implementation of the `txeo::Trainer<T>` abstract class. It performs **Ordinary Least Squares (OLS)** linear regression using **Gradient Descent**.

## Features

- Implements gradient descent for linear regression
- Supports **learning rate tuning**, **convergence tolerance**, and **early stopping**
- Optionally uses the **Barzilai-Borwein method** for adaptive learning rate
- Access to learned **weight/bias matrix**

## Template Parameter

- `T`: Floating-point type (e.g., `float`, `double`)

## Example Usage

```cpp
// Create training data (y = 2x + 1)
txeo::Matrix<double> X({{1.0}, {2.0}, {3.0}});
txeo::Matrix<double> y({{3.0}, {5.0}, {7.0}});

OlsGDTrainer<double> trainer(txeo::DataTable<double>(X, y));
trainer.set_tolerance(1e-5);
trainer.fit(1000, LossFunc::MSE, 10);

if (trainer.is_converged()) {
    auto weights = trainer.weight_bias();
    std::cout << "Model: y = " << weights(0,0) << "x + " << weights(1,0) << std::endl;

    txeo::Matrix<double> test_input(1,1,{4.0});
    auto prediction = trainer.predict(test_input);
    std::cout << "Prediction for x=4: " << prediction(0,0) << std::endl;
}
```

---

## Constructors

### `Trainer(const txeo::DataTable<T> &data)`

Creates a trainer using a data table object.

```cpp
txeo::Trainer(const txeo::DataTable<T> &data);
```

---

## Public Methods

### `predict(input)`

Performs prediction on new input data.

```cpp
txeo::Tensor<T> predict(const txeo::Tensor<T>& input);
```

### `learning_rate()`

Returns the current learning rate.

```cpp
T learning_rate() const;
```

### `set_learning_rate(value)`

Sets the learning rate used in training.

```cpp
void set_learning_rate(T value);
```

### `enable_variable_lr()` / `disable_variable_lr()`

Toggles the use of the Barzilai-Borwein adaptive learning rate.

```cpp
void enable_variable_lr();
void disable_variable_lr();
```

### `weight_bias()`

Returns the model weight-bias matrix.

```cpp
const txeo::Matrix<T>& weight_bias() const;
```

### `tolerance()` / `set_tolerance(value)`

Gets or sets the convergence tolerance.

```cpp
T tolerance() const;
void set_tolerance(const T& value);
```

### `is_converged()`

Checks if convergence was reached during training.

```cpp
bool is_converged() const;
```

### `min_loss()`

Returns the minimum loss encountered during training.

```cpp
T min_loss() const;
```

---

## Exceptions

### `OlsGDTrainerError`

Exception type used for runtime errors within the trainer.

```cpp
class OlsGDTrainerError : public std::runtime_error;
```

---

## Inheritance

- Inherits from: `txeo::Trainer<T>`
- Implements:
  - `predict()`
  - `train()`

For detailed API references, see individual method documentation at [txeo::OlsGDTrainer](https://txeo-doc.netlify.app/classtxeo_1_1_ols_g_d_trainer.html).