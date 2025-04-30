#!/bin/bash

# 当任何命令失败时立即退出
set -e

# 获取脚本所在的目录，以便我们知道项目根目录在哪里
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="$SCRIPT_DIR/.." # 项目根目录是脚本目录的上一级

# 定义构建目录
BUILD_DIR="$PROJECT_ROOT/build"

# --- 新增：初始化和更新 Git 子模块 ---
echo "Initializing and updating Git submodules..."
cd "$PROJECT_ROOT" # 确保在项目根目录执行
git submodule update --init --recursive # 使用 --recursive 确保子模块的子模块也被更新
# --- 结束新增部分 ---

# 清理旧的构建目录（可选）
# echo "Cleaning previous build directory..."
# rm -rf "$BUILD_DIR"

# 创建构建目录
echo "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

# 进入构建目录
cd "$BUILD_DIR"

# 运行 CMake 配置
# -S 指定源码目录 (项目根目录)
# -B 指定构建目录 (当前目录)
# -G Ninja 使用 Ninja 构建系统 (通常比 Make 快)
# -DCMAKE_BUILD_TYPE=Release 配置为发布模式以获得优化
echo "Configuring project with CMake..."
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release

# 运行构建命令
echo "Building project with Ninja..."
ninja

# 可选：运行安装命令 (如果 CMakeLists.txt 中定义了安装规则)
# echo "Installing project..."
# ninja install

# 返回到项目根目录
cd "$PROJECT_ROOT"

echo "Build completed successfully!"
echo "The Python module should be located in: $BUILD_DIR"