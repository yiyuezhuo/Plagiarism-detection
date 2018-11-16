# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:49:48 2018

@author: yiyuezhuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CNN_Text
#import json
import random
from data_process import EmbeddingManager


#import os

from evaluation_tool import EvaluationManager


used_embedding = 'sgns.zhihu.word'
embedding_dir = 'embedding_tensor'
load_cache_model = "model_cache/model"

embedding_manager = EmbeddingManager(used_embedding, embedding_dir)
    
class args():
    embed_num = embedding_manager.spec['num_word']
    embed_dim = embedding_manager.spec['dim_word']
    class_num = 2
    kernel_num = 100 # default value in https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/main.py
    kernel_sizes = [3,4,5]
    dropout = 0.5
    lr = 0.001
    
model = CNN_Text(args)
'''
network without pretrained embedding with training weight on embedding layer(20000):
    train acc 0.9050952303434804
tensor([[4588,  213],
        [ 370,  972]])
test acc 0.7679738562091504
tensor([[216,  26],
        [ 45,  19]])
'''
'''
model.load_state_dict(torch.load(load_cache_model))
print(f'load {load_cache_model}')
'''

feature_list, target_list = embedding_manager.load_data('traning_data.json')
feature_list_test, target_list_test = embedding_manager.load_data('testing_data.json')

cuda = True
if cuda:

    feature_list = [feature.cuda() for feature in feature_list]
    target_list = [target.cuda() for target in target_list]
    feature_list_test = [feature.cuda() for feature in feature_list_test]
    target_list_test = [target.cuda() for target in target_list_test]

    model.cuda()
    

#for i in range(100000):
len_data = len(feature_list)
n_iter = 20000

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
loss_cache = []
for i in range(n_iter):
    idx = random.randint(0,len_data-1)
    feature, target = feature_list[idx], target_list[idx]
    
    optimizer.zero_grad()
    out = model(feature) # before softmax(log_softmax)
    loss = F.cross_entropy(out, target, size_average=False)
    loss_cache.append(loss.item())
    loss.backward()
    optimizer.step()
    
    if len(loss_cache) == 1000:
        loss_est = sum(loss_cache)/len(loss_cache)
        loss_cache = []
        print(f'loss: {loss_est} {i}/{n_iter}')
        
import datetime
timestamp = str(datetime.datetime.now()).replace(':','-')
cache_path = "model_cache/model"+timestamp
torch.save(model.state_dict(), cache_path)
print(f'save cache model {cache_path}')


model.train(False)

evaluation_manager = EvaluationManager(model, embedding_manager, cuda = cuda)
get_acc = evaluation_manager.get_acc
confuse_matrix = evaluation_manager.confuse_matrix
analysis_doc = evaluation_manager.analysis_doc


print(f'train acc {get_acc(feature_list, target_list)}') # 0.94
print(confuse_matrix(feature_list, target_list))

print(f'test acc {get_acc(feature_list_test, target_list_test)}') #
print(confuse_matrix(feature_list_test, target_list_test))

    
good_list = analysis_doc('extracted_text/AI报告_杨帆_117106010714.docx.txt')
bad_list = analysis_doc('extracted_text/AI报告_张佳洛_117106021976.docx.txt')

import matplotlib.pyplot as plt
import numpy as np

plt.plot(good_list)
plt.title(np.mean(good_list))
plt.show()
plt.plot(bad_list)
plt.title(np.mean(bad_list))
plt.show(np.mean(bad_list))