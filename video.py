# 在视频/网络摄像头上运行检测器
# 不在batch上迭代，而是在视频的帧上迭代

from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

# 命令行参数
def arg_parse():
   
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    # images（用于指定输入图像或图像目录）
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    # det（保存检测结果的目录）
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    # batch大小
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    # objectness置信度
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    # NMS阈值
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    # cfg（替代配置文件）
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    # reso（输入图像的分辨率，可用于在速度与准确度之间的权衡）
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)

    return parser.parse_args()

if __name__ == '__main__':

    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    num_classes = 80 # COCO数据集中目标的名称
    classes = load_classes("data/coco.names")
        
    # 初始化网络，加载权重
    print("正在加载网络QAQ")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("网络加载成功QvQ")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # GPU加速
    if CUDA:
        model.cuda()
    
    # 模型评估
    model.eval()

    # 绘制边界框:从colors中随机选颜色绘制矩形框
    # 边界框左上角创建一个填充后的矩形，写入该框位置检测到的目标的类别
    def write(x, results):

        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results # 仅处理一帧
        cls = int(x[-1])
        color = random.choice(colors)
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1) # -1表示填充的矩形
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,225,225], 1)

        return img

    # 检测阶段
    videofile = "workingcell.mp4" # 加载视频文件路径
    cap = cv2.VideoCapture(videofile) # 用OpenCV打开视频/相机流
    #assert cap.isOpened(), '未找到需要检测视频TAT'
    
    frames = 0 # 帧的数量
    start = time.time()
    # 在帧上迭代，一次处理一帧
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            img = prep_image(frame, inp_dim)
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1,2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            output = model(Variable(img, volatile=True), CUDA)
            output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("视频的FPS为 {:5.4f}".format(frames / (time.time() - start)))
                # 使用cv2.imshow展示画有边界框的帧
                cv2.imshow("帧", frame)
                key = cv2.waitKey(1)
                # 用户按q，就会终止视频(代码中断循环)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))
            im_dim = im_dim.repeat(output.size(0), 1)/inp_dim
            output[:,1:5] *= im_dim

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            list(map(lambda x: write(x, frame), output))

            cv2.imshow("帧", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                    break
            frames += 1
            print(time.time() - start)
            print("视频的FPS为 {:5.4f}".format(frames / (time.time() - start)))
        
        else:
            break




