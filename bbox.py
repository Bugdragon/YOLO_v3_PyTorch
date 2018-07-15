from __future__ import division

import torch
import random
import numpy as np
import cv2

# 计算两个边界框的IoU
def bbox_iou(box1, box2):
    # 获取边框的坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # 获取交叉矩形的坐标
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # 交叉面积
    inter_area = (inter_rect_x2 - inter_rect_x1) * (inter_rect_y2 - inter_rect_y1)
    #if torch.cuda.is_available():
    #        inter_area = torch.max(inter_rect_x2 - inter_rect_x1,torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1, torch.zeros(inter_rect_x2.shape).cuda())
    #else:
    #        inter_area = torch.max(inter_rect_x2 - inter_rect_x1,torch.zeros(inter_rect_x2.shape))*torch.max(inter_rect_y2 - inter_rect_y1, torch.zeros(inter_rect_x2.shape))
    
    # 合并面积
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    # IoU
    iou = inter_area / union_area

    return iou