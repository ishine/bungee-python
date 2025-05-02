# Bungee Python Bindings

[![PyPI version](https://badge.fury.io/py/bungee-python.svg)](https://badge.fury.io/py/bungee-python)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

该项目提供了 [Bungee](https://github.com/bungee-audio-stretch/bungee) C++ 库的 Python 绑定，允许您在 Python 中轻松实现高质量的实时音频时间拉伸和变调。

`bungee-core` 库 ([bungee-core/README.md](bungee-core/README.md)) 是一个现代化的 C++ 库，用于音频处理，具有以下特点：

* 高质量的音频时间拉伸和变调。
* 支持实时处理，低延迟。
* 允许连续改变播放速度和音高，支持平滑搓碟和变速播放。
* 基于频域相位声码器算法。
* 使用现代 C++ 编写，代码清晰健壮。
* 采用 MPL-2.0 宽松许可证。

本项目使用 [pybind11](https://github.com/pybind/pybind11) ([extern/pybind11](extern/pybind11)) 将 Bungee 的核心功能暴露给 Python。

## 安装

您可以通过 pip 从 PyPI 安装 `bungee-python` (发布后):

```bash
pip install bungee-python
```

## 使用示例

下面是一个简单的示例，演示如何使用 `bungee-python` 将音频速度减慢一半：

```python
# filepath: example.py
import numpy as np
# 假设您已经安装了 bungee-python 包
from bungee_python import bungee # 导入方式可能需要根据您的打包结构调整

# --- 准备输入音频 ---
sample_rate = 44100
channels = 1
duration_seconds = 5
frequency = 440 # A4 音符

# 生成一个简单的正弦波
t = np.linspace(0., duration_seconds, int(sample_rate * duration_seconds))
input_audio = 0.5 * np.sin(2. * np.pi * frequency * t)

# 确保是 float32 类型
input_audio = input_audio.astype(np.float32)

# Bungee 需要 (frames, channels) 的形状
if channels == 1:
    input_audio = input_audio[:, np.newaxis]
elif channels == 2:
    # 如果是立体声，需要堆叠
    input_audio = np.stack([input_audio, input_audio], axis=-1)

print(f"Input shape: {input_audio.shape}") # 输出: Input shape: (220500, 1)

# --- 使用 Bungee 处理 ---

# 1. 初始化 Bungee Stretcher
#    需要提供采样率和通道数
stretcher = bungee.Bungee(sample_rate=sample_rate, channels=channels)

# 2. 设置处理参数 (可选，默认为 1.0)
stretcher.set_speed(0.5) # 将速度设置为 0.5 倍 (慢放)
stretcher.set_pitch(1.0) # 音高保持不变

# 3. 处理音频
#    输入和输出都是 float32 NumPy 数组，形状为 (frames, channels)
output_audio = stretcher.process(input_audio)

print(f"Output shape: {output_audio.shape}") # 输出: Output shape: (大约 441000, 1)

# --- 后续处理 ---
# 您可以将 output_audio 保存到文件或进行其他处理
# 例如使用 soundfile 库保存:
# import soundfile as sf
# sf.write("output_slow.wav", output_audio, sample_rate)

print("处理完成!")
```

有关 Bungee C++ API 的更详细信息，请参阅 [bungee-core/bungee/Bungee.h](bungee-core/bungee/Bungee.h) 和 [bungee-core/README.md](bungee-core/README.md)。

## 从源码构建

如果您想从源代码构建 `bungee-python`，请按照以下步骤操作：

1. **克隆仓库 (包括子模块):**

    ```bash
    git clone --recurse-submodules https://github.com/longredzhong/bungee-python.git
    cd bungee-python
    ```

    (如果已经克隆但没有更新子模块，请运行 `git submodule update --init --recursive`)

2. **安装构建依赖:**

    * C++ 编译器 (支持 C++17)
    * CMake (版本 3.15 或更高)
    * Ninja (推荐，用于加速构建)
    * Python (3.8 或更高) 和开发头文件 (例如 `python3-dev` on Debian/Ubuntu)

3. **运行构建脚本:**
    项目提供了一个构建脚本来简化流程：

    ```bash
    ./scripts/build.sh
    ```

    该脚本 ([scripts/build.sh](scripts/build.sh)) 会创建 `build` 目录，运行 CMake 配置，并使用 Ninja 进行编译。编译后的 Python 扩展模块 (`.so` 或 `.pyd`) 会位于 `build` 目录中。

4. **安装或测试:**
    您可以将构建目录中的模块复制到您的项目中，或者使用 `pip install .` 在本地安装（需要调整 `pyproject.toml` 或 `setup.py` 以正确构建和打包）。

## 依赖

* **运行时:**
  * Python (>= 3.8)
  * NumPy
* **构建时:**
  * [bungee-core](https://github.com/bungee-audio-stretch/bungee) (作为 Git 子模块)
    * Eigen
    * KissFFT
  * [pybind11](https://github.com/pybind/pybind11) (作为 Git 子模块)
  * CMake (>= 3.15)
  * C++17 兼容编译器
  * Ninja (可选但推荐)

## 许可证

本项目采用 [MPL-2.0](https://opensource.org/licenses/MPL-2.0) 许可证，与 `bungee-core` 保持一致。

## 致谢

* 感谢 [Parabola Research Limited](https://parabolaresearch.com/) 开发了优秀的 [Bungee](https://github.com/bungee-audio-stretch/bungee) 库。
* 感谢 [pybind11](https://github.com/pybind/pybind11) 团队提供了方便易用的 C++/Python 绑定工具。
