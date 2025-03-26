# Trainer

## Overview
The `txeo::Trainer` class is an **abstract base class** that provides the interface for training machine learning models in **txeo**. It handles training/evaluation data, common training parameters, and the training lifecycle.

Derived classes must implement the `predict()` and `train()` methods.

## Features

- Abstract base class with pure virtual methods.
- Manages training and evaluation datasets.
- Supports early stopping.
- Tracks whether the model has been trained.

## Template Parameter

- `T`: The numeric type used in tensors (e.g., `float`, `double`).

## Constructors

### **Trainer(x_train, y_train, x_eval, y_eval)**

Initializes the trainer with training and evaluation datasets.

```cpp
Trainer(const txeo::Tensor<T>& x_train,
        const txeo::Tensor<T>& y_train,
        const txeo::Tensor<T>& x_eval,
        const txeo::Tensor<T>& y_eval);
```

### **Trainer(x_train, y_train)**

Initializes the trainer using the training data for evaluation as well.

```cpp
Trainer(const txeo::Tensor<T>& x_train,
        const txeo::Tensor<T>& y_train);
```

## Public Methods

### **fit(epochs, metric)**

Trains the model for a fixed number of epochs.

```cpp
void fit(size_t epochs, txeo::LossFunc metric);
```

### **fit(epochs, metric, patience)**

Trains the model with early stopping.

```cpp
void fit(size_t epochs, txeo::LossFunc metric, size_t patience);
```

### **predict(input)**

Pure virtual method to generate predictions from a trained model.
Must be implemented in derived classes.

```cpp
txeo::Tensor<T> predict(const txeo::Tensor<T>& input) = 0;
```

### **is_trained()**

Returns `true` if the model has been trained.

```cpp
bool is_trained() const;
```

## Exceptions

### **TrainerError**

Exception type thrown by `Trainer` operations.

```cpp
class TrainerError : public std::runtime_error;
```

## Example Usage

```cpp
class MyTrainer : public txeo::Trainer<float> {
  public:
    using txeo::Trainer<float>::Trainer;

    txeo::Tensor<float> predict(const txeo::Tensor<float>& input) override {
        // your prediction logic
    }

  protected:
    void train(size_t epochs, txeo::LossFunc loss_func) override {
        // your training logic
    }
};
```

For detailed API references, see individual method documentation at [txeo::Predictor](https://txeo-doc.netlify.app/classtxeo_1_1_predictor.html).