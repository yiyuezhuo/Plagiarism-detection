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

print("creating model")
used_embedding = 'sgns.zhihu.word'
embedding_dir = 'embedding_tensor'
load_cache_model = "model_cache/model"

embedding_manager = EmbeddingManager(used_embedding, embedding_dir)
    

model = CNN1DText(embed_num = 259922, embed_dim = 300, class_num = 2, kernel_num = 100, kernel_sizes = (3,4,5),
                dropout = 0.5)

embedding_obj = torch.load(os.path.join(embedding_dir, used_embedding))
model.use_pretrained_embedding(embedding_obj['tensor'], non_trainable = True)

#output = model(torch.LongTensor([[1, 2, 3,4,5,6,7,8,9,0]]))
#predicted = F.softmax(output,dim = 1)


# prepare data
print("loading data")
feature_list, target_list = embedding_manager.load_data('traning_data.json')
feature_list_test, target_list_test = embedding_manager.load_data('testing_data.json')

cuda = True
if cuda:

    feature_list = [feature.cuda() for feature in feature_list]
    target_list = [target.cuda() for target in target_list]
    feature_list_test = [feature.cuda() for feature in feature_list_test]
    target_list_test = [target.cuda() for target in target_list_test]

    model.cuda()
    


# training
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
len_data = len(feature_list)
n_iter = 20000

print("start training")
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
