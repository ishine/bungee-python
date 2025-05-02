# Bungee Python Bindings

[![PyPI version](https://badge.fury.io/py/bungee-python.svg)](https://badge.fury.io/py/bungee-python)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

该项目提供了 [Bungee](https://github.com/bungee-audio-stretch/bungee) C++ 库的 Python 绑定，允许您在 Python 中轻松实现高质量的实时音频时间拉伸和变调。

This project provides Python bindings for the [Bungee](https://github.com/bungee-audio-stretch/bungee) C++ library, enabling high-quality real-time audio time-stretching and pitch-shifting in Python.

---

## 特性 Features

- 高质量的音频时间拉伸和变调  
  High-quality audio time-stretching and pitch-shifting
- 支持实时处理，低延迟  
  Real-time processing with low latency
- 允许连续改变播放速度和音高，支持平滑搓碟和变速播放  
  Continuous speed/pitch change, smooth scratching, and variable playback
- 基于频域相位声码器算法  
  Frequency-domain phase vocoder algorithm
- 使用现代 C++ 编写，代码清晰健壮  
  Modern C++ implementation, robust and clean code
- 采用 MPL-2.0 宽松许可证  
  Licensed under MPL-2.0

---

## 安装 Installation

您可以通过 pip 从 PyPI 安装 `bungee-python`:

You can install `bungee-python` from PyPI via pip:

```bash
pip install bungee-python
```

---

## 使用示例 Example

下面是一个简单的示例，演示如何使用 `bungee-python` 将音频速度减慢一半：

Here is a simple example showing how to slow down audio by half using `bungee-python`:

```python
import numpy as np
from bungee_python import bungee

sample_rate = 44100
channels = 1
duration_seconds = 5
frequency = 440

t = np.linspace(0., duration_seconds, int(sample_rate * duration_seconds))
input_audio = 0.5 * np.sin(2. * np.pi * frequency * t)
input_audio = input_audio.astype(np.float32)
input_audio = input_audio[:, np.newaxis]  # (frames, channels)

stretcher = bungee.Bungee(sample_rate=sample_rate, channels=channels)
stretcher.set_speed(0.5)  # 慢放 slow down
stretcher.set_pitch(1.0)  # 音高不变 keep pitch

output_audio = stretcher.process(input_audio)
print(f"Output shape: {output_audio.shape}")
```

---

## 从源码构建 Build from Source

1. **克隆仓库 (包括子模块) / Clone repository (with submodules):**

    ```bash
    git clone --recurse-submodules https://github.com/longredzhong/bungee-python.git
    cd bungee-python
    ```

2. **安装构建依赖 / Install build dependencies:**

    - C++ 编译器 (支持 C++17) / C++17 compiler
    - CMake (>= 3.15)
    - Ninja (推荐 recommended)
    - Python (>= 3.8) 和开发头文件 / Python dev headers

3. **运行构建脚本 / Run build script:**

    ```bash
    ./scripts/build.sh
    ```

    编译后的 Python 扩展模块会位于 `build` 目录中。  
    The built Python extension module will be in the `build` directory.

4. **安装或测试 / Install or test:**

    可以用 `pip install .` 在本地安装。  
    You can install locally with `pip install .`.

---

## 依赖 Dependencies

- **运行时 Runtime:**
  - Python (>= 3.8)
  - NumPy
- **构建时 Build:**
  - [bungee-core](https://github.com/bungee-audio-stretch/bungee) (as submodule)
    - Eigen
    - KissFFT
  - [pybind11](https://github.com/pybind/pybind11) (as submodule)
  - CMake (>= 3.15)
  - C++17 compiler
  - Ninja (optional)

---

## 许可证 License

本项目采用 [MPL-2.0](https://opensource.org/licenses/MPL-2.0) 许可证，与 `bungee-core` 保持一致。  
This project is licensed under [MPL-2.0](https://opensource.org/licenses/MPL-2.0), same as `bungee-core`.

---

## 致谢 Acknowledgements

- 感谢 [Parabola Research Limited](https://parabolaresearch.com/) 开发了优秀的 [Bungee](https://github.com/bungee-audio-stretch/bungee) 库。  
  Thanks to [Parabola Research Limited](https://parabolaresearch.com/) for developing the excellent Bungee library.
- 感谢 [pybind11](https://github.com/pybind/pybind11) 团队提供了方便易用的 C++/Python 绑定工具。  
  Thanks to the [pybind11](https://github.com/pybind/pybind11) team for their great C++/Python binding tool.

---