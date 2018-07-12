from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

# 把检测特征图转换成二维张量，张量的每一行对应边界框的属性，5个参数：输出，输入图像的维度
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    # 锚点的维度与net块的h和w属性一致，输入图像的维度和检测图的维度之商就是步长，用检测特征图的步长分割锚点
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # 对（x，y）坐标和objectness分数执行Sigmoid函数操作
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])