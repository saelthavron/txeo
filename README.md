# Txeo: a Developer-Friendly TensorFlow C++ Wrapper

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## 📝 Overview

**Txeo** is an intuitive and light **C++ wrapper for TensorFlow 2.0**, designed to simplify the TensorFlow C++ API while maintaining **high performance and flexibility**. Developed purely in Modern C++, **Txeo** enables developers to work with TensorFlow as easily as in Python, without the steep learning curve of TensorFlow’s low-level C++ API.

## ✨ Features

- 📦 **Easy-to-use API** – Designed for a clean and readable C++ interface.
- 🔧 **Tensor abstraction** – Simplifies TensorFlow tensor creation and manipulation.
- 💾 **Tensor IO** – Reading and writing tensors from and to text files.
- 🏗 **Model wrapper** – Streamlined saved model loading and inference.
- 💻 **XLA Enabling** - Enabling/disabling of XLA features.

## ⚡ Installation

### **Requirements**

- **Build Essentials** - Essential tools for c/c++ development.
- **Compilers** – clang v19+ or gcc v13+.
- **TensorFlow 2.18.0** - <https://github.com/tensorflow/tensorflow>
- **Protobuf 3.21.9** - <https://github.com/protocolbuffers/protobuf>

### **Steps with TensorFlow binaries (fastest way)**

#### **1️⃣ Download and install binaries**

Choosing clang generated binaries for tensorflow and installing them in `/opt`:

```sh
wget https://github.com/rdabra/txeo-tf/releases/download/v1.0.0/libtensorflow-2.18-clang19-linux-x64-cpu.tar.gz
sudo tar -xzf libtensorflow-2.18-clang19-linux-x64-cpu.tar.gz -C /opt/
echo "export TensorFlow_ROOT_DIR=/opt/tensorflow" >> ~/.bashrc
source ~/.bashrc
```

💡 **Important Note** : The TensorFlow source code used for the provided binaries **was not modified** in any way. These binaries are **only provided to simplify installation** for Txeo users.

📁 Other assets are available <https://github.com/rdabra/txeo-tf/releases/tag/v1.0.0>

#### **2️⃣ Clone and install Txeo**

Txeo is also instaled in `/opt` by default.

```sh
git clone https://github.com/rdabra/txeo-tf.git
cd txeo-tf
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### **Steps with TensorFlow built from source (may take a very long time)**

#### **1️⃣ Clone and install Tensorflow**

Bazel must be installed previously (see <https://github.com/bazelbuild/bazelisk>). During the config phase, choose not to compile for gpu (without cuda support). This build was tested in clang v19 and gcc v13 (without the `-std=gnu2x` key).

```sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout refs/tags/v2.18.0
./configure
bazel build -c opt --copt=-std=gnu2x --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.1 --copt=-msse4.2 //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //tensorflow:install_headers
```

Copy generated libraries and includes to `/opt`

```sh
cd bazel-bin
sudo mkdir /opt/tensorflow
sudo cp -r tensorflow/include /opt/tensorflow
sudo mkdir /opt/tensorflow/lib
sudo cp -r tensorflow/*.so* /opt/tensorflow/lib
echo "export TensorFlow_ROOT_DIR=/opt/tensorflow" >> ~/.bashrc
source ~/.bashrc 
```

#### **3️⃣ Clone and install Txeo**

**Txeo** is also instaled in `/opt` by default.

```sh
git clone https://github.com/rdabra/txeo-tf.git
cd txeo-tf
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

## 🚗 Basic Usage

### **A simple CMakeLists.txt**

Considering that TensorFlow and **Txeo** are installed in `/opt`:

```cmake title="CMakeLists.txt"
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
set(TENSORFLOW_LIBRARY "/opt/tensorflow/lib/libtensorflow_cc.so")

# Create an executable
add_executable(hello_txeo main.cpp)

# Include directories for Txeo and TensorFlow
target_include_directories(hello_txeo PRIVATE ${TXEO_INCLUDE_DIR} ${TENSORFLOW_INCLUDE_DIR})

# Link Txeo and TensorFlow manually
target_link_libraries(hello_txeo PRIVATE ${TXEO_LIBRARY} ${TENSORFLOW_LIBRARY})

# Optionally set rpath for runtime library search
set_target_properties(hello_txeo PROPERTIES INSTALL_RPATH "/opt/txeo/lib;/opt/tensorflow/lib")
```

Here is a code sample where a 3x3 `txeo::Tensor` is defined, written to a file and then another instance is created from the saved file.

```cpp title="main.cpp"
#include "txeo/Tensor.h"
#include "txeo/TensorIO.h"
#include <iostream>

int main() {

  // 3x3 tensor created from a list of float values in row major scheme
  txeo::Tensor<double> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

  txeo::TensorIO::write_textfile(tensor, "tensor.txt");

  auto loaded_tensor = txeo::TensorIO::read_textfile<double>("tensor.txt");

  std::cout << loaded_tensor << std::endl;

  return 0;
}
```

📁 For more samples, please visit the [examples folder](https://github.com/rdabra/txeo-tf/tree/main/examples).

## 📜 License

Txeo is licensed under the Apache License 2.0.

### Third-Party Licenses

Txeo uses TensorFlow C++ (licensed under Apache 2.0).  
TensorFlow’s original license is included in [`third_party/tensorflow/LICENSE`](third_party/tensorflow/LICENSE).  
For more details, visit [TensorFlow GitHub](https://github.com/tensorflow/tensorflow).
