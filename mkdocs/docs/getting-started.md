# Installation Guide

## Requirements

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

---

## Option 1: Installation Steps with Precompiled Binaries (Fastest Way)

This method **installs TensorFlow and Protobuf binaries** into `/opt/`.

### **1️⃣ Download and install Protobuf**

```sh
wget https://github.com/rdabra/txeo/releases/download/v1.0.0/libprotobuf-3.21.9-linux-x64.tar.gz
sudo tar -xzf libprotobuf-3.21.9-linux-x64.tar.gz -C /opt/
echo "export Protobuf_ROOT_DIR=/opt/protobuf" >> ~/.bashrc
source ~/.bashrc
```

### **2️⃣ Download and install TensorFlow**

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

### **3️⃣ Clone and install txeo**

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

---

## Option 2: Installation Steps with Protobuf and TensorFlow built from source (may take a long time)

### **1️⃣ Clone and install Protobuf**

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

### **2️⃣ Clone and install Tensorflow**

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

### **3️⃣ Installing txeo**

```sh
git clone https://github.com/rdabra/txeo.git
cd txeo
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
echo "export LD_LIBRARY_PATH=/opt/tensorflow/lib:/opt/txeo/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
```