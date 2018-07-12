# YOLO_v3_PyTorch

### 背景知识
+ 卷积神经网络的工作原理，包括残差块、跳过连接和上采样；
+ 目标检测、边界框回归、IoU 和非极大值抑制（NMS）；
+ 基础的 PyTorch 使用，会创建简单的神经网络；
+ 阅读 YOLO 三篇论文，了解 YOLO 的工作原理。

### 代码实现
1. 创建 YOLO 网络层级
2. 实现网络的前向传播
3. objectness 置信度阈值和非极大值抑制
4. 设计输入和输出管道

### 版本条件
* Ubuntu 18.04LTS(64-bit)
* Python 3.6.5(pip3)
* torch 0.4.0(cpu)
* OpenCV 3.4.2
