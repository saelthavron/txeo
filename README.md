<p align="center">
  <img src="images/txeo.png" alt="txeo logo">
</p>

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/rdabra/txeo)](https://github.com/rdabra/txeo/releases)
[![GitHub issues](https://img.shields.io/github/issues/rdabra/txeo)](https://github.com/rdabra/txeo/issues)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://txeo-doc.netlify.app/)
[![Last Commit](https://img.shields.io/github/last-commit/rdabra/txeo)](https://github.com/rdabra/txeo/commits/main)
[![Star Badge](https://img.shields.io/github/stars/rdabra/txeo-tf?style=social)]

# [txeo: a Modern C++ Wrapper for TensorFlow](https://rdabra.github.io/txeo/)

## 📝 Overview

**txeo** is a lightweight and intuitive **C++ wrapper for TensorFlow**, designed to **simplify TensorFlow C++ development** while preserving **high performance and flexibility**. Built entirely with **Modern C++**, **txeo** allows developers to use TensorFlow with the ease of a high-level API, eliminating the complexity of its low-level C++ interface.

**Library site:** https://rdabra.github.io/txeo/

## ✨ Features

- 📦 **Intuitive API** – A clean and modern C++ interface, simplifying TensorFlow C++ usage.
- 🔧 **High-Level Tensor Abstraction** – Easily create, manipulate, and operate on tensors.
- 💾 **Flexible Tensor IO** – Seamless reading and writing of tensors to text files.
- 🏗 **Simplified Model Loading** – Load and run saved TensorFlow models with minimal setup.
- ⚡ **XLA Acceleration** – Effortlessly enable or disable TensorFlow’s XLA optimizations.
- 🚀 **Near-Native Performance** – Achieves **99.35% to 99.79% of native TensorFlow speed** with negligible overhead.
- 🛡 **Encapsulated TensorFlow API** – Fully abstracts TensorFlow internals for a cleaner, more maintainable experience.

## 🚀 Performance Comparison

**txeo** was benchmarked against the native **TensorFlow C++ API** using inference from a saved **multiclassification convolution model**.

- **Model and other info:**
  - **279,610 parameters**
  - **1 Softmax Output Layer** with 10 classes
  - **3 Fully-Connected ReLU Convolutional Layers** with 200 units each
  - **Input**: 210,000 grayscale images (28×28).
  - **CPU**: AMD Ryzen 7 5700X CPU
  - **TensorFlow 2.0**: Compiled with CPU optimization

### 🔎 **Results Overview**

| Compiler | txeo (μs) | TensorFlow C++ (μs) | Difference (%) |
|----------|-----------|-----------------|----------------|
| GCC      | 233,994   | 232,494         | +0.65%         |
| Intel    | 234,489   | 232,683         | +0.78%         |
| Clang    | 236,858   | 234,016         | +1.21%         |

- The performance overhead is **negligible**, ranging from **0.65% to 1.21%**.
- **txeo’s abstraction layer** provides **ease of use** with almost no cost to performance.

## ⚡ Installation Guide

### **🔹Requirements**

- **Supported OS:** 🐧 **Linux** (Tested on Ubuntu and Manjaro).  
  - ⚠️ **Windows and macOS are not yet officially supported.**
- **Build Tools:** 🛠 Essential C/C++ development tools.
- **CMake:** 🏗 Built with **v3.25+**.
- **Compilers:** 💻 Requires a C++20-compatible compiler:
  - ✅ **Clang** (Tested with **v19.1.2**)
  - ✅ **GCC** (Tested with **v13.2.0**)
  - ✅ **Intel** (Tested with **v2025.0.4**)
  - 🛠 Supports **concepts, ranges, and other C++20 features**.
- **Dependencies:**
  - 🔗 **TensorFlow 2.18.0** → [GitHub](https://github.com/tensorflow/tensorflow)
  - 📜 **Protobuf 3.21.9** → [GitHub](https://github.com/protocolbuffers/protobuf)

### **🔹Option 1: Installation Steps with Precompiled Binaries (Fastest Way)**

This method **installs TensorFlow and Protobuf binaries** into `/opt/`.

#### **1️⃣ Download and install Protobuf**

```sh
wget https://github.com/rdabra/txeo/releases/download/v1.0.0/libprotobuf-3.21.9-linux-x64.tar.gz
sudo tar -xzf libprotobuf-3.21.9-linux-x64.tar.gz -C /opt/
echo "export Protobuf_ROOT_DIR=/opt/protobuf" >> ~/.bashrc
source ~/.bashrc
```

#### **2️⃣ Download and install TensorFlow**

Choose the correct version based on your system:

| Version | Download |
| -------- | ------- |
| 💻 Without CPU optimizations | [libtensorflow-2.18-linux-x64-cpu.tar.gz](https://github.com/rdabra/txeo/releases/download/v1.0.0/libtensorflow-2.18-linux-x64-cpu.tar.gz) |
| 🚀 With CPU optimizations: | [libtensorflow-2.18-linux-x64-cpu-opt.tar.gz](https://github.com/rdabra/txeo/releases/download/v1.0.0/libtensorflow-2.18-linux-x64-cpu-opt.tar.gz) |
| 🎮 With GPU support: | [libtensorflow-2.18-linux-x64-gpu.tar.gz](https://github.com/rdabra/txeo/releases/download/v1.0.0/libtensorflow-2.18-linux-x64-gpu.tar.gz) |

💡 **Important Note** : The Protobuf and TensorFlow source codes used for compiling the binaries above **were not modified** in any way. These assets are **only provided to simplify installation** for **txeo** users.

Installing TensorFlow binaries:

```sh
wget https://github.com/rdabra/txeo/releases/download/v1.0.0/libtensorflow-2.18-linux-x64-cpu.tar.gz
sudo tar -xzf libtensorflow-2.18-linux-x64-cpu.tar.gz -C /opt/
echo "export TensorFlow_ROOT_DIR=/opt/tensorflow" >> ~/.bashrc
source ~/.bashrc
```

#### **3️⃣ Clone and install txeo**

Installing **txeo** and making libraries visible via library path:

```sh
git clone https://github.com/rdabra/txeo.git
cd txeo
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
echo "export LD_LIBRARY_PATH=/opt/tensorflow/lib:/opt/txeo/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
```

### **🔹Option 2: Installation Steps with Protobuf and TensorFlow built from source (may take a long time)**

#### **1️⃣ Clone and install Protobuf**

```sh
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout refs/tags/v3.21.9
cmake -S. -Bcmake-out -G Ninja -DCMAKE_INSTALL_PREFIX="/opt/protobuf" -Dprotobuf_ABSL_PROVIDER=package -Dprotobuf_BUILD_TESTS=OFF
cd cmake-out
cmake --build .
sudo cmake --install .
echo "export Protobuf_ROOT_DIR=/opt/protobuf" >> ~/.bashrc
source ~/.bashrc
```

#### **2️⃣ Clone and install Tensorflow**

⚠️ Important:
Ensure Bazel is installed before proceeding. You can use Bazelisk to manage Bazel versions:
[Bazelisk GitHub](https://github.com/bazelbuild/bazelisk). For the gcc compiler, key `-std=gnu2x` must be removed.

```sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout refs/tags/v2.18.0
./configure
bazel build -c opt --copt=-std=gnu2x --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.1 --copt=-msse4.2 //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //tensorflow:install_headers
```

Copying libraries and includes accordingly:

```sh
cd bazel-bin
sudo mkdir /opt/tensorflow
sudo cp -r tensorflow/include /opt/tensorflow
sudo mkdir /opt/tensorflow/lib
sudo cp -r tensorflow/*.so* /opt/tensorflow/lib
echo "export TensorFlow_ROOT_DIR=/opt/tensorflow" >> ~/.bashrc
source ~/.bashrc 
```

#### **3️⃣ Installing txeo**

```sh
git clone https://github.com/rdabra/txeo.git
cd txeo
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
echo "export LD_LIBRARY_PATH=/opt/tensorflow/lib:/opt/txeo/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
```

## 🚗 Basic Usage

This section provides **two simple C++ examples** to help you get started with **txeo**.

> **📌 Prerequisite:** Before compiling, ensure that TensorFlow and **txeo** are properly installed in `/opt/`.  
> If necessary, add the library paths:  
>
> ```sh
> export LD_LIBRARY_PATH=/opt/tensorflow/lib:/opt/txeo/lib:$LD_LIBRARY_PATH
> ```

### **🔹 A Simple CMakeLists.txt**

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

### 🔢 **Example 1: Tensor Basics**

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

### 🔮 **Example 2: Running Inference with a Saved Model**

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

👓 Documentation with extensive usage examples is hosted at [Netlify](https://txeo-doc.netlify.app/).

## 📆 Roadmap

**txeo** is actively evolving! Here are some of the upcoming features:

### 🏋️ Training Capabilities

- **Model Training** - Enable training models using TensorFlow C++.
- **Backpropagation Support** - Implement automatic differentiation.
- **Gradient Descent & Optimizers** - Integrate optimizers like SGD and Adam.

### 🔢 Advanced Tensor Operations

- **Linear Algebra Functions (SVD, QR decomposition)** - Matrix Computations on tensors.

### 📊 Model Saving & Loading Enhancements

- **Checkpointing** - Save model weights at different training stages.
- **Frozen Graph Support** - Load & optimize frozen models for inference.

## 📬 Contact

For any inquiries or contributions:

- **GitHub Discussions:** [Start a discussion](https://github.com/rdabra/txeo/discussions)
- **Issue Reporting:** [Open an issue](https://github.com/rdabra/txeo/issues)
- **Email:** [robertodias70@outlook.com](mailto:robertodias70@outlook.com) *(for serious inquiries only)*

## 📜 License

**txeo** is licensed under the **Apache License 2.0**, meaning it is **open-source, free to use, modify, and distribute**, while requiring proper attribution.

### 📄 Third-Party Licenses

**txeo** depends on third-party libraries that have their own licenses:

- **TensorFlow C++** - Licensed under **Apache License 2.0**  
  - 📜 [TensorFlow License](third_party/tensorflow/LICENSE)
  - 🔗 [TensorFlow GitHub](https://github.com/tensorflow/tensorflow)
- **Protobuf** - Licensed under **BSD 3-Clause**  
  - 📜 [Protobuf License](https://github.com/protocolbuffers/protobuf/blob/main/LICENSE)

> **📌 Note:** The precompiled binaries of TensorFlow and Protobuf provided in the releases section **are unmodified versions** of the official source code.
