# 手写数字识别小项目 — README

本仓库演示了 “用 Python 从零训练一层隐藏网络 → 用 C++/OpenCL 端极速推理” 的完整闭环。下面依次说明 项目用途、两种部署方式、文件结构、如何切换推理模式，以及参考输出示例

## 1. 简介

- **训练端**：`NeuralNetworkFromScratch/train.py` 以纯 NumPy 方式在 MNIST 数据集上训练一张「784 → 20 → 10」的全连接网络，并把权重保存为 `model.npz`
- **推理端** (`test/`)
  - CPU 版本 `main.cpp` —— 朴素实现
  - GPU 版本 `NeuralNetworkOpenCL.cpp` + `neural_network_kernels.cl` —— 用 OpenCL 并行计算，两层 FC + Sigmoid，每张图片仅需亚毫秒级。

用户可以在同一套源码中自由切换 “纯 C++” 与 “OpenCL 加速” 两种模式，对比性能与实现差异。

## 2. 部署方式

### 2.1 训练端部署（Python）

1. 安装 Python 依赖
    只需要在终端中运行以下命令来安装所有依赖：

   ```
   cd NeuralNetworkFromScratch
   pip install -r requirements.txt
   ```

   这将自动安装项目所需的 Python 包，确保 `train.py` 能顺利运行

2. 运行训练代码
    在命令行中运行以下命令来训练神经网络并保存模型：

   ```
   python train.py
   ```

   训练完成后，权重文件将被保存为 `model.npz`，测试数据在 `mnist.npz` 之中

### 2.2 推理端部署（C++）

1. 构建推理代码

   进入 `digitInfer` 文件夹，使用 CMake 构建项目，需要使用 MSYS32 shell 配置环境：

   - 安装 ninja 和 cmake

   ```
   pacman -Syu
   pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
   ```

   - 配置与编译

   ```
   cd cmake-build-debug
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ..
   ninja
   ```

2. 切换推理模式

   在 `digitInfer/CMakeLists.txt` 文件中，切换推理模式只需要注释或取消注释一行代码：

   - **CPU 模式**（不使用 OpenCL）：

     ```
     add_executable(test
             nets.hpp
             main.cpp)
     #        NeuralNetworkOpenCL.cpp)
     ```

   - **OpenCL 模式**（使用 OpenCL 加速）：

     ```
     add_executable(test
             nets.hpp
     #        main.cpp)
             NeuralNetworkOpenCL.cpp)
     ```

3. **运行推理代码**

   编译完成后，你可以运行推理程序：

   ```
   ./test.exe
   ```

   你将看到如下输出：

   ```
   复制初始化 OpenCL 神经网络...
   数据导入完成
   模型加载完成！输入编号 (0-9999), 负数退出
   index = 0
   预测结果: 7
   实际结果: 7
   ##### ASCII 图像在此处省略 #####
   index = 1
   预测结果: 2
   实际结果: 2
   ...
   ```

------

**注意：**

- **OpenCL 依赖**：确保你的显卡驱动包含 OpenCL 组件，或者安装适配 CPU 的 OpenCL 运行时。

## 3. 文件说明

```
.
├── NeuralNetworkFromScratch
    ├── train.py					# 训练
    ├── model.npz					# 权重保存
    ├── data
        ├── mnist.npz				# 数据集
└── main.cpp                  		# 纯 CPU 推理
└── NeuralNetworkOpenCL.cpp   		# OpenCL 推理
└── neural_network_kernels.cl 		# 两个 kernel: FC + argmax
└── nets.hpp						# 头文件，定义 C++ 侧网络结构与工具函数
└── CMakeLists.txt
```

## 4. 运行结果

| 模式                   | 每秒处理图片数 |
| ---------------------- | -------------- |
| OpenCL（无优化）       | ≈ 16004        |
| OpenCL（分块矩阵优化） | ≈ 17189        |
