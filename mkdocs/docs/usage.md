# Basic Usage

This section provides **two simple C++ examples** to help you get started with **txeo**.

> **📌 Prerequisite:** Before compiling, ensure that TensorFlow and **txeo** are properly installed in `/opt/`.  
> If necessary, add the library paths:  
>
> ```sh
> export LD_LIBRARY_PATH=/opt/tensorflow/lib:/opt/txeo/lib:$LD_LIBRARY_PATH
> ```

## A Simple CMakeLists.txt

To compile a project using **txeo**, use the following **CMakeLists.txt** file.

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.25)
project(HelloTxeo LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Manually specify Txeo installation paths
set(TXEO_INCLUDE_DIR "/opt/txeo/include")
set(TXEO_LIBRARY "/opt/txeo/lib/libtxeo.so")

# Manually specify TensorFlow paths
set(TENSORFLOW_INCLUDE_DIR "/opt/tensorflow/include")
set(TENSORFLOW_CC_LIBRARY "/opt/tensorflow/lib/libtensorflow_cc.so")
set(TENSORFLOW_FRAMEWORK "/opt/tensorflow/lib/libtensorflow_framework.so")

# Create an executable
add_executable(hello_txeo main.cpp)

# Include directories for Txeo and TensorFlow
target_include_directories(hello_txeo PRIVATE ${TXEO_INCLUDE_DIR} ${TENSORFLOW_INCLUDE_DIR})

# Link Txeo and TensorFlow manually
target_link_libraries(hello_txeo PRIVATE ${TXEO_LIBRARY} ${TENSORFLOW_CC_LIBRARY} ${TENSORFLOW_FRAMEWORK})

# Optionally set rpath for runtime library search
set_target_properties(hello_txeo PROPERTIES INSTALL_RPATH "/opt/txeo/lib;/opt/tensorflow/lib")
```

💡 Note: If TensorFlow is installed in a different location, update TENSORFLOW_INCLUDE_DIR and TENSORFLOW_LIBRARY paths accordingly.

## Example 1: Tensor Basics

Here is a code sample where a 3x3 `txeo::Matrix` is defined, written to a file and then another instance is created from the saved file.

```cpp
//main.cpp
#include "txeo/Tensor.h"
#include "txeo/MatrixIO.h"
#include <iostream>

using namespace txeo;
using namespace std;

int main() {

  // 3×3 matrix created from a list of double values in row-major order
  Matrix<double> matrix({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

  // Save matrix to file
  MatrixIO::write_textfile(matrix, "matrix.txt");

  // Load matrix from file
  auto loaded_matrix = MatrixIO::read_textfile<double>("matrix.txt");

  // Convert matrix to second-order tensor
  auto loaded_tensor = Matrix::to_tensor(loaded_matrix);

  // Reshape second-order tensor to first-order
  loaded_tensor.reshape({9});

  // Display loaded tensor
  cout << loaded_tensor << endl;

  return 0;
}
```

## Example 2: Running Inference with a Saved Model

This example loads a saved TensorFlow model, performs inference on an input tensor, and saves the output:

```cpp
//main.cpp
#include "txeo/Predictor.h"
#include "txeo/Tensor.h"
#include "txeo/MatrixIO.h"

using namespace txeo;
using namespace std;

int main() {

  // Define paths to model and input/output tensors
  string model_path{"path/to/saved_model"};
  string input_path{"path/to/input_tensor.txt"};
  string output_path{"path/to/output_tensor.txt"};

  // Load the model
  Predictor<float> predictor{model_path};
  
  // Read input tensor from file
  auto input = MatrixIO::read_textfile<float>(input_path);
  
  // Run inference
  auto output = predictor.predict(input);
  
  // Save output tensor to file
  MatrixIO::write_textfile(output, output_path);

  return 0;
}
```

💡 Note: Ensure that "path/to/saved_model" contains a valid TensorFlow model before running this example.

📁 For more samples, please visit the [examples folder](https://github.com/rdabra/txeo/tree/main/examples).

👓 Doxygen Documentation with extensive usage examples is hosted at [Netlify](https://txeo-doc.netlify.app/).
