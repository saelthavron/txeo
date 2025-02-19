# Txeo: a Developer-Friendly TensorFlow C++ Wrapper

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## 📝 Overview

**Txeo** is a lightweight and intuitive **C++ wrapper for TensorFlow 2.0**, designed to **simplify TensorFlow C++ development** while preserving **high performance and flexibility**. Built entirely with **Modern C++**, **Txeo** allows developers to use TensorFlow with the ease of a high-level API, eliminating the complexity of its low-level C++ interface.

## ✨ Features

- 📦 **Easy-to-use API** – Designed for a clean and readable C++ interface.
- 🔧 **Tensor abstraction** – Simplifies TensorFlow tensor creation and manipulation.
- 💾 **Tensor IO** – Reading and writing tensors from and to text files.
- 🏗 **Model wrapper** – Streamlined saved model loading and inference.
- 💻 **XLA Enabling** - Enabling/disabling of XLA features.
- 🐚 **TensorFlow Encapsulation** - TensorFlow is hidden from **Txeo** users.

## ⚡ Installation

### **Requirements**

- **Linux** - Tested in Ubuntu and Manjaro. Not yet available for other platforms.
- **Build Essentials** - Essential tools for C/C++ development.
- **CMake** - Built with v3.25.
- **Compilers**
  - clang (tested with v19.1.0)
  - gcc (tested with v13.2.0)
  - Most support C++20 features: concepts, ranges, etc.
- **TensorFlow 2.18.0** - <https://github.com/tensorflow/tensorflow>
- **Protobuf 3.21.9** - <https://github.com/protocolbuffers/protobuf>

### **Steps with Protobuf and TensorFlow binaries (fastest way)**

In the following sections, a bash shell is assumed and all the resources are installed in directory `/opt`.

#### **1️⃣ Download and install binaries**

Installing Protobuf binaries:

```sh
wget https://github.com/rdabra/txeo-tf/releases/download/v1.0.0/libprotobuf-3.21.9-linux-x64.tar.gz
sudo tar -xzf libprotobuf-3.21.9-linux-x64.tar.gz -C /opt/
echo "export Protobuf_ROOT_DIR=/opt/protobuf" >> ~/.bashrc
source ~/.bashrc
```

For TensorFlow, we provide the binaries

- Without CPU optimizations: [libtensorflow-2.18-linux-x64-cpu.tar.gz](https://github.com/rdabra/txeo-tf/releases/download/v1.0.0/libtensorflow-2.18-linux-x64-cpu.tar.gz)
- With CPU optimizations: [libtensorflow-2.18-linux-x64-cpu-opt.tar.gz](https://github.com/rdabra/txeo-tf/releases/download/v1.0.0/libtensorflow-2.18-linux-x64-cpu-opt.tar.gz)
- With GPU support: [libtensorflow-2.18-linux-x64-gpu.tar.gz](https://github.com/rdabra/txeo-tf/releases/download/v1.0.0/libtensorflow-2.18-linux-x64-gpu.tar.gz)

💡 **Important Note** : The Protobuf and TensorFlow source codes used for compiling the binaries above **were not modified** in any way. These assets are **only provided to simplify installation** for **Txeo** users.

Installing TensorFlow binaries:

```sh
wget https://github.com/rdabra/txeo-tf/releases/download/v1.0.0/libtensorflow-2.18-linux-x64-cpu.tar.gz
sudo tar -xzf libtensorflow-2.18-linux-x64-cpu.tar.gz -C /opt/
echo "export TensorFlow_ROOT_DIR=/opt/tensorflow" >> ~/.bashrc
source ~/.bashrc
```

#### **2️⃣ Clone and install Txeo**

Installing **Txeo** and making libraries visible via library path:

```sh
git clone https://github.com/rdabra/txeo-tf.git
cd txeo-tf
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
echo "export LD_LIBRARY_PATH=/opt/tensorflow/lib:/opt/txeo/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
```

### **Steps with Protobuf and TensorFlow built from source (may take a long time)**

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

Bazel must be installed previously (see <https://github.com/bazelbuild/bazelisk>). For the gcc compiler, key `-std=gnu2x` must be removed.

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

#### **3️⃣ Clone and install Txeo**

```sh
git clone https://github.com/rdabra/txeo-tf.git
cd txeo-tf
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
echo "export LD_LIBRARY_PATH=/opt/tensorflow/lib:/opt/txeo/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
```

## 🚗 Basic Usage

### **A Simple CMakeLists.txt**

Considering that TensorFlow and **Txeo** are properly installed:

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

### **Two Simple Examples**

Here is a code sample where a 3x3 `txeo::Tensor` is defined, written to a file and then another instance is created from the saved file.

```cpp
//main.cpp
#include "txeo/Tensor.h"
#include "txeo/TensorIO.h"
#include <iostream>

using namespace txeo;
using namespace std;

int main() {

  // 3x3 tensor created from a list of float values in row major scheme
  Tensor<double> tensor({3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

  TensorIO::write_textfile(tensor, "tensor.txt");

  auto loaded_tensor = TensorIO::read_textfile<double>("tensor.txt");

  cout << loaded_tensor << endl;

  return 0;
}
```

By providing a directory path containing a properly saved model (`.pb`), input data and output info, inference is intuitive and straightforward with `txeo::Predictor` :

```cpp
//main.cpp
#include "txeo/Predictor.h"
#include "txeo/Tensor.h"
#include "txeo/TensorIO.h"

using namespace txeo;
using namespace std;

int main() {

  string model_path{"path/to/saved_model"};
  string input_path{"path/to/input_tensor.txt"};
  string output_path{"path/to/output_tensor.txt"};

  Predictor<float> predictor{model_path};
  auto input = TensorIO::read_textfile<float>(input_path);
  auto output = predictor.predict(input);
  TensorIO::write_textfile(output, output_path);

  return 0;
}
```

📁 For more samples, please visit the [examples folder](https://github.com/rdabra/txeo-tf/tree/main/examples).

## 📜 License

**Txeo** is licensed under the Apache License 2.0.

### Third-Party Licenses

**Txeo** uses TensorFlow C++ (licensed under Apache 2.0).  
TensorFlow’s original license is included in [`third_party/tensorflow/LICENSE`](third_party/tensorflow/LICENSE).  
For more details, visit [TensorFlow GitHub](https://github.com/tensorflow/tensorflow).
