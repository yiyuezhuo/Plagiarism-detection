# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:38:55 2018

@author: yiyuezhuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CNN1DText
#import json
import random
import os
from data_process import EmbeddingManager


#import os

from evaluation_tool import EvaluationManager

print("creating model")
used_embedding = 'sgns.zhihu.word'
embedding_dir = 'embedding_tensor'
load_cache_model = "model_cache/model"

embedding_manager = EmbeddingManager(used_embedding, embedding_dir)
    

model = CNN1DText(embed_num = 259922, embed_dim = 300, class_num = 2, kernel_num = 100, kernel_sizes = (3,4,5),
                dropout = 0.5)

embedding_obj = torch.load(os.path.join(embedding_dir, used_embedding))
model.use_pretrained_embedding(embedding_obj['tensor'], non_trainable = True)

x0 = torch.tensor([1,2,3,4,5,6,7,8,9])
x1 = model.embed(x0)
x2 = x1.transpose(0,1)
x3 = torch.unsqueeze(x2,0)
x4 = [conv(x3) for conv in model.convs]
x5 = [F.pad(torch.squeeze(x4[i], 0), model.pad_sizes[i]) for i in range(len(x4))]
x6 = torch.cat(x5,0) # (len(kernel_sizes)*kernel_num, L')
x7 = x6.transpose(0,1) # (L', len(kernel_sizes)*kernel_num)
x8 = model.dropout(x7)
x9 = model.fc(x8) # (L', class_num)
