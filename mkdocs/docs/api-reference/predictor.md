# Predictor

**txeo**'s `Predictor` class handles inference tasks using TensorFlow SavedModels. It loads models, performs predictions, and provides metadata about model inputs, outputs, and devices.

## Constructors

### Initialization with path to model

```cpp
explicit Predictor(std::filesystem::path model_path);
```

Constructs a `Predictor` object from a TensorFlow SavedModel directory containing a `.pb` file.

**Example (Python Model Freezing):**

```python
import tensorflow as tf

model = tf.saved_model.load("path/to/trained_model")
concrete_func = model.signatures["serving_default"]
frozen_func = tf.python.framework.convert_to_constants.convert_variables_to_constants_v2(concrete_func)
tf.io.write_graph(
    frozen_func.graph.as_graph_def(),
    "path/to/frozen_model",
    "frozen.pb",
    as_text=False
)
```

---

## Methods

### `get_input_metadata`

Returns input tensor metadata (names and shapes).

```cpp
const TensorInfo &get_input_metadata() const noexcept;
```

### `get_output_metadata`

Returns output tensor metadata (names and shapes).

```cpp
const TensorInfo &get_output_metadata() const noexcept;
```

### `get_input_metadata_shape`

Returns shape for a specified input tensor by name.

```cpp
std::optional<txeo::TensorShape> get_input_metadata_shape(const std::string &name) const;
```

### `get_output_metadata_shape`

Returns shape for a specified output tensor by name.

```cpp
std::optional<txeo::TensorShape> get_output_metadata_shape(const std::string &name) const;
```

### `get_devices`

Returns available compute devices.

```cpp
std::vector<DeviceInfo> get_devices() const;
```

### `predict`

Performs single input/output inference.

```cpp
txeo::Tensor<T> predict(const txeo::Tensor<T> &input) const;
```

**Example:**

```cpp
Tensor<float> input({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
auto output = predictor.predict(input);
```

### `predict_batch`

Performs batch inference with multiple named inputs.

```cpp
std::vector<txeo::Tensor<T>> predict_batch(const TensorIdent &inputs) const;
```

**Example:**

```cpp
std::vector<std::pair<std::string, txeo::Tensor<float>>> inputs {
    {"image", image_tensor},
    {"metadata", meta_tensor}
};
auto results = predictor.predict_batch(inputs);
```

### `enable_xla`

Enables or disables XLA (Accelerated Linear Algebra) compilation.

```cpp
void enable_xla(bool enable);
```

**Note:** Prefer enabling XLA before the first inference call.

---

## Structures

### DeviceInfo

| Member          | Description            |
|-----------------|------------------------|
| `name`          | Device name            |
| `device_type`   | Type of device (CPU/GPU)|
| `memory_limit`  | Memory limit in bytes  |

---

## Exceptions

### PredictorError

Exception thrown when predictor operations fail.

```cpp
class PredictorError : public std::runtime_error;
```

---

For detailed API references, see individual method documentation at [txeo::Predictor](https://txeo-doc.netlify.app/classtxeo_1_1_predictor.html).
