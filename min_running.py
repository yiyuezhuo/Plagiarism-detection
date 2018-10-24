# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 00:38:55 2018

@author: yiyuezhuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from CNN_Text import CNN_Text


class args():
    embed_num = 259922
    embed_dim = 300
    class_num = 2
    kernel_num = 100 # default value in https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/main.py
    kernel_sizes = [3,4,5]
    dropout = 0.5
    lr = 0.001
    
model = CNN_Text(args)
model.load_state_dict(torch.load("model_cache/model"))

print(model(torch.LongTensor([[1,2,3,4,5,6,7,8,8]])))