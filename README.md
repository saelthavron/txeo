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

### **Steps with TensorFlow and Protobuf binaries (fastest way)**

#### **1️⃣ Download and install binaries**

Choosing clang generated binaries for protobuf and installing them in `/opt`:

```sh
wget https://github.com/rdabra/txeo-tf/releases/download/v1.0.0/libprotobuf-3.21.9-clang19-linux-x64.tar.gz
sudo tar -xzf libprotobuf-3.21.9-clang19-linux-x64.tar.gz -C /opt/
export Protobuf_ROOT_DIR=/opt/protobuf
```

Choosing clang generated binaries for tensorflow and installing them in `/opt`:

```sh
wget https://github.com/rdabra/txeo-tf/releases/download/v1.0.0/libtensorflow-2.18-clang19-linux-x64-cpu.tar.gz
sudo tar -xzf libtensorflow-2.18-clang19-linux-x64-cpu.tar.gz -C /opt/
export TensorFlow_ROOT_DIR=/opt/tensorflow
```

> **📌 Important Note:** The source codes of TensorFlow and Protobuf used for the binaries available here for download were not modified in any way, shape or form. Our intention is solely to provide the user a fast way to install **Txeo**.  

📁 Other assets are available <https://github.com/rdabra/txeo-tf/releases/tag/v1.0.0>

#### **2️⃣ Clone and install Txeo**

```sh
git clone https://github.com/rdabra/txeo-tf.git
cd txeo-tf
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### **Steps with TensorFlow and Protobuf built from source (may take a very long time)**

#### **1️⃣ Clone and install Protobuf**

```sh
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout refs/tags/v3.21.9
cmake -S. -Bcmake-out -G Ninja -DCMAKE_INSTALL_PREFIX="/opt/protobuf" -Dprotobuf_ABSL_PROVIDER=package -Dprotobuf_BUILD_TESTS=OFF
cd cmake-out
cmake --build .
sudo cmake --install .
export Protobuf_ROOT_DIR=/opt/protobuf 
```

#### **2️⃣ Clone and install Tensorflow**

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
export TensorFlow_ROOT_DIR=/opt/tensorflow 
```

#### **3️⃣ Clone and install Txeo**

```sh
git clone https://github.com/rdabra/txeo-tf.git
cd txeo-tf
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

## 📜 License

Txeo is licensed under the Apache License 2.0.

### Third-Party Licenses

Txeo uses TensorFlow C++ (licensed under Apache 2.0).  
TensorFlow’s original license is included in [`third_party/tensorflow/LICENSE`](third_party/tensorflow/LICENSE).  
For more details, visit [TensorFlow GitHub](https://github.com/tensorflow/tensorflow).
