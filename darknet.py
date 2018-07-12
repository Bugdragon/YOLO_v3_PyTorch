from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *

def parse_cfg(cfgfile):
  """
  Takes a cfg file,returns a list of blocks. 
  """
  file = open(cfgfile, 'r')
  lines = file.read().split('\n') # store lines in a list
  lines = [x for x in lines if len(x)>0] # get rid of empty lines
  lines = [x for x in lines if x[0] != '#'] # get rid of comments
  lines = [x.rstrip().lstrip() for x in lines] # get rid of whitespaces
  
  block = {}
  blocks = []

  for line in lines:
    if line[0] == "[": # a new block
      if len(block) != 0: # not empty
        blocks.append(block) # add blocks list
        block = {} # init blocks
      block["type"] = line[1:-1].rstrip()
    else:
      key, value = line.split("=")
      block[key.rstrip()] = value.lstrip()
  blocks.append(block)

  return blocks

# 空层
class EmptyLayer(nn.Module):
  def __init__(self):
    super(EmptyLayer, self).__init__() # 调用父类方法

# 定义一个新的Detectinet_infLayer保存用于检测边界框的锚点
class DetectionLayer(nn.Module):
  def __init__(self, anchors):
    super(DetectionLayer, self).__init__()
    self.anchors = anchors

def create_modules(blocks):
  net_info = blocks[0] # input and pre-processing
  module_list = nn.ModuleList()
  prev_filters = 3 # depth of last conv
  output_filters = [] # number of output conv kernel，输出通道数量序列
  
  for index, x in enumerate(blocks[1:]):
    module = nn.Sequential()
    # convolutional模块有卷积层、批量归一化层和leaky ReLU激活层
    if (x["type"] == "convolutional"):
      # get layer info
      activation = x["activation"]
      try:
        batch_normalize = int(x["batch_normalize"])
        bias = False
      except:
        batch_normalize = 0
        bias = True

      filters = int(x["filters"]) # 卷积数量
      padding = int(x["pad"]) # 填充数量
      kernel_size = int(x["size"]) # 卷积核大小
      stride= int(x["stride"]) # 步长

      if padding:
        pad = (kernel_size - 1) // 2 # 运算后，宽度和高度不变
      else:
        pad = 0
      
      # Add conv layer
      conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
      module.add_mjiasuqianxiangchuanboodule("conv_{0}".format(index), conv)

      # Add batch norm layer
      if batch_normalize:
        bn = nn.BatchNorm2d(filters)
        module.add_module("batch_norm_{0}".format(index), bn)
      
      # Check activation
      if activation == "leaky":
        activn = nn.LeakyReLU(0.1, inplace = True) # 斜率0.1
        module.add_module("leaky_{0}".format(index), activn)

    # upsample上采样层
    elif (x["type"] == "upsample"):
      stride = int(x["stride"])
      upsample = nn.Upsample(scale_factor = 2, mode = "nearest") # 或者mode="bilinear"
      module.add_module("upsample_module_list{}".format(index), upsample)

    # route路由层，路由层是获取之前层的拼接
    elif (x["type"] == "route"):
      x["layers"] = x["layers"].split(",") # 保存start和end层号
      # Start of a route
      start = int(x["layers"][0])
      # end, if there exists one
      try:
        end = int(x["layers"][1])
      except:
        end = 0 # 没有end
      route = EmptyLayer() # 创建空层
      module.add_module("route_{0}".format(index), route)
      if end < index:
        # 计算卷积数量，即两层叠加
        filters = output_filters[start] + output_filters[end]
      else:
        filters = output_filters[start] 

    # shortcut捷径层（跳过连接），捷径层是将前一层的特征图添加到后面的层上
    elif (x["type"] == "shortcut"):
      shortcut = EmptyLayer()
      module.add_module("shortcut_{}".format(index), shortcut)

    # yolo层，检测层
    elif (x["type"] == "yolo"):
      # 保存mask序号
      mask = x["mask"].split(",")
      mask = [int(x) for x in mask]

      # 保存anchors box
      anchors = x["anchors"].split(",")
      anchors = [int(a) for a in anchors]
      # 两个一组，还ge和宽
      anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
      # 选取mask序号对应的anchors box，一般为3个
      anchors = [anchors[i] for i in mask]
      
      detection = DetectionLayer(anchors)
      module.add_module("Detection_{}".format(index), detection)
    
    module_list.append(module)
    prev_filters = filters
    output_filters.append(filters)

  return (net_info, module_list)

# blocks = parse_cfg("cfg/yolov3.cfg")
# print(create_modules(blocks))

class Darknet(nn.Module):
  # 用net_info和module_list对网络进行初始化
  def __init__(self, cfgfile):
    super(Darknet, self).__init__()
    self.blocks = parse_cfg(cfgfile)
    self.net_info, self.module_list = create_modules(self.blocks)

  # CUDA为true，则用GPU加速前向传播
  def forward(self, x, CUDA):
    # delf.blocks第一个元素是net块
    modules = self.blocks[1:]
    # 缓存每个层的输出特征图，以备route层和shortcut层使用
    outputs = {}

    write = 0
    for i, module in enumerate(modules):
      module_type = (module["type"])

      if module_type == "convolutional" or module_type == "upsample":
        x = self.module_list[i](x)

      elif module_type == "route":
        layers = module["layers"]
        layers = [int(a) for a in layers]

        if layers[0] > 0:
          layers[0] = layers[0] - i

        if len(layers) == 1:
          x = outputs[i + layers[0]]

        else:
          if layers[1] > 0:
            layers[1] = layers[1] - i

          map1 = outputs[i + layers[0]]
          map2 = outputs[i + layers[1]]
          x = torch.cat((map1, map2), 1)

      elif module_type == "shortcut":
        from_ = int(module["from"])
        x = outputs[i-1] + outputs[i+from_]

      elif module_type == "yolo":
        anchors = self.module_list[i][0].anchors
        # input dimensions
        inp_dim = int(self.net_info["height"])
        
        # number of classes
        num_classes = int(module["classes"])

        # transform
        x = x.data
        x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
        if not write:
          detections = x
          write = 1

        else:
          detections = torch.cat((detections, x), 1)
      
      outputs[i] = x

    return detections

  






  