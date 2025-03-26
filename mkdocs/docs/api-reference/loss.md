# Loss

## Overview

The `txeo::Loss` class computes error metrics between predicted and ground truth tensors. It supports multiple standard loss functions that are selectable at runtime.

> ✅ Compatible with any numeric tensor type (float/double recommended)

## Supported Loss Functions

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **MSLE**: Mean Squared Logarithmic Error
- **LCHE**: Log-Cosh Error

---

## Template Parameter

- `T`: Numeric type of tensor elements (`float`, `double`, etc.)

---

## Constructor

### `Loss(const Tensor<T>& valid, LossFunc func = LossFunc::MSE)`

Creates a new loss evaluator object.

```cpp
txeo::Loss<float> loss(y_true, txeo::LossFunc::MAE);
```

---

## Public Methods

### `T get_loss(const Tensor<T>& pred) const`

Computes loss using the current selected function.

```cpp
auto error = loss.get_loss(pred);
```

### `void set_loss(LossFunc func)`

Sets the active loss function.

```cpp
loss.set_loss(txeo::LossFunc::LCHE);
```

---

## Specific Loss Functions

All functions require prediction tensors with the same shape as the validation tensor.

### `mean_squared_error(pred)` / `mse(pred)`

Computes:
$$
MSE = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

### `mean_absolute_error(pred)` / `mae(pred)`

Computes:
$$
MAE = \frac{1}{N} \sum_{i=1}^{N}|y_i - \hat{y}_i|
$$

### `mean_squared_logarithmic_error(pred)` / `msle(pred)`

Computes:
$$
MSLE = \frac{1}{N} \sum_{i=1}^{N}(\log(1+y_i) - \log(1+\hat{y}_i))^2
$$
> ⚠ Requires all values to be non-negative.

### `log_cosh_error(pred)` / `lche(pred)`

Computes:
$$
LCHE = \frac{1}{N} \sum_{i=1}^{N}\log(\cosh(y_i - \hat{y}_i))
$$

---

## Shorthand Aliases

| Alias | Full Function |
|-------|----------------|
| `lche(pred)` | `log_cosh_error(pred)` |
| `mae(pred)` | `mean_absolute_error(pred)` |
| `mse(pred)` | `mean_squared_error(pred)` |
| `msle(pred)` | `mean_squared_logarithmic_error(pred)` |

---

## Exceptions

### `LossError`

Thrown on shape mismatch or invalid inputs:

```cpp
class LossError : public std::runtime_error;
```

---

## Example

```cpp
 #include "txeo/Loss.h"
 
 int main() {
     // Create validation data
     txeo::Tensor<float> valid({4}, {1.5f, 2.0f, 3.2f, 4.8f});
     
     // Initialize loss calculator with default (MSE)
     txeo::Loss<float> loss(valid);
     
     // Generate predictions
     txeo::Tensor<float> pred({4}, {1.6f, 1.9f, 3.0f, 5.0f});
     
     // Calculate and compare different losses
     std::cout << "MSE: " << loss.get_loss(pred) << std::endl;
     std::cout << "Direct MAE: " << loss.mae(pred) << std::endl;
     
     // Switch to MSLE and calculate
     loss.set_loss(txeo::LossFunc::MSLE);
     std::cout << "MSLE: " << loss.get_loss(pred) << std::endl;
     
     return 0;
 }
```

---

## Notes

- Tensors must have the same shape.
- First dimension is assumed to be the sample axis.
- Negative values in MSLE will throw `LossError`.
- Loss functions are interchangeable at runtime.

For detailed API references, see individual method documentation at [txeo::Loss](https://txeo-doc.netlify.app/classtxeo_1_1_loss.html).
