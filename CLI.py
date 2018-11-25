# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:32:30 2018

@author: yiyuezhuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#import json
import random
import os
from data_process import EmbeddingManager

from evaluation_tool import EvaluationManager

print("creating model")
used_embedding = 'sgns.zhihu.word'
embedding_dir = 'embedding_tensor'
load_cache_model = "model_cache/model"

embedding_manager = EmbeddingManager(used_embedding, embedding_dir)

import models