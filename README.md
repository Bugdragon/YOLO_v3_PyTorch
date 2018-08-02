# YOLO_v3_PyTorch
YOLOv3目标检测器的**输出样例**：
![Image text](https://raw.githubusercontent.com/Bugdragon/YOLO_v3_PyTorch/master/det/det_%E5%B7%A5%E4%BD%9C%E7%BB%86%E8%83%9E1.png)

### 代码实现
1. 创建 YOLOv3 网络层级☑
2. 实现网络的前向传播☑
3. objectness 置信度阈值和非极大值抑制☑
4. 设计输入和输出管道☑
5. 在视频/网络摄像头上运行检测器☑

### 背景知识
+ 卷积神经网络的工作原理，包括残差块、跳过连接和上采样；
+ 目标检测、边界框回归、IoU 和非极大值抑制（NMS）；
+ 基础的 PyTorch 使用，会创建简单的神经网络；
+ 阅读 YOLO 三篇论文，了解 YOLO 的工作原理。

### 版本条件
* Ubuntu 18.04LTS(64-bit)
* Python 3.6.5(pip3)
* torch 0.4.0(cpu)
* OpenCV 3.4.2

### 安装指南
* git clone https://github.com/Bugdragon/YOLO_v3_PyTorch.git
* cd YOLO_v3_PyTorch
* wget https://pjreddie.com/media/files/yolov3.weights
* python detect.py

#### tips
1. 提前将需要检测的图片放入 imgs 文件夹下
2. 检测结果图片将被保存在 det 文件夹下
