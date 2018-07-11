from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
  else
   key,value = line.split("=")
   block[key.rstrip()] = value.lstrip()
 blocks.append(block)

 return blocks

def create_modules(blocks):
 net_info = blocks[0] # input and pre-processing
 module_list = nn.ModuleList()
 prev_filters = 3 # depth of last conv
 output_filters = [] # number of output conv kernel
 
 for index, x in enumerate(blocks[1:]):
  module = nn.Sequential()
   fdf defkkk