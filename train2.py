# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:50:49 2018

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


used_embedding = 'sgns.zhihu.word'
embedding_dir = 'embedding_tensor'
load_cache_model = "model_cache/model"

embedding_manager = EmbeddingManager(used_embedding, embedding_dir)
    

model = CNN1DText(embed_num = 259922, embed_dim = 300, class_num = 2, kernel_num = 100, kernel_sizes = (3,4,5),
                dropout = 0.5)

embedding_obj = torch.load(os.path.join(embedding_dir, used_embedding))
model.use_pretrained_embedding(embedding_obj['tensor'])

output = model(torch.LongTensor([[1, 2, 3,4,5,6,7,8,9,0]]))
predicted = F.softmax(output,dim = 1)

