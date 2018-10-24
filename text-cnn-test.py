# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:16:38 2018

@author: yiyuezhuo
"""

used_embedding = 'sgns.zhihu.word'
embedding_dir = 'embedding_tensor'

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CNN_Text
import json
import random

import os

embedding_obj = torch.load(os.path.join(embedding_dir, used_embedding))

word2idx_map = {word:idx for idx,word in enumerate(embedding_obj['word_list'])}

def word2idx(word_vector):
    words = []
    for word in word_vector:
        if word in word2idx_map:
            words.append(word2idx_map[word])
    return torch.tensor([words])

def create_emb_layer(weights_matrix, non_trainable=False):
    # https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

    


embedding_layer, num_embeddings, embedding_dim = create_emb_layer(embedding_obj['tensor'], non_trainable = True)

class args():
    embed_num = embedding_obj['num_word']
    embed_dim = embedding_obj['dim_word']
    class_num = 2
    kernel_num = 100 # default value in https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/main.py
    kernel_sizes = [3,4,5]
    dropout = 0.5
    lr = 0.001



cnn = CNN_Text(args, embedding_layer)
output = cnn(torch.LongTensor([[1, 2, 3,4,5,6,7,8,9,0]]))
predicted = F.softmax(output,dim = 1)
assert F.nll_loss(F.log_softmax(output, dim=1),torch.tensor([1])).item() == F.cross_entropy(output, torch.tensor([1])).item()

with open('traning_data.json',encoding='utf8') as f:
    data = json.load(f)
    
word_list = data['word_list']
label_list = data['label_list']

cnn(word2idx(word_list[0]))
predicted = F.softmax(output,dim = 1)

feature_list = []
target_list = []
for i in range(len(word_list)):
    section, label = word_list[i], label_list[i]
    idxs = word2idx(section)
    if len(idxs[0]) >= 5: # min size for current cnn
        feature_list.append(idxs)
        target_list.append(torch.LongTensor([label]))
print(f'size {len(word_list)} -> {len(feature_list)}')

#feature_list = [word2idx(section) for section in word_list]
#target_list = [torch.LongTensor([label]) for label in label_list]

    

cuda = True
if cuda:
    '''
    Why model.cuda() will move all parameters to gpu but tensor.cuda() not?
    for feature in feature_list:
        feature.cuda()
    for target in target_list:
        target.cuda()
    '''
    feature_list = [feature.cuda() for feature in feature_list]
    target_list = [target.cuda() for target in target_list]
    cnn.cuda()

#for i in range(100000):
len_data = len(feature_list)
n_iter = 10000

optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
loss_cache = []
for i in range(n_iter):
    idx = random.randint(0,len_data-1)
    feature, target = feature_list[idx], target_list[idx]
    
    optimizer.zero_grad()
    out = cnn(feature) # before softmax(log_softmax)
    loss = F.cross_entropy(out, target, size_average=False)
    loss_cache.append(loss.item())
    loss.backward()
    optimizer.step()
    
    if i % 1000 == 0:
        loss_est = sum(loss_cache)/len(loss_cache)
        loss_cache = []
        print(f'loss: {loss_est} {i}/{n_iter}')
        
torch.save(cnn.state_dict(), "model_cache/model")
#cnn = CNN_Text(args, **kwargs)
#cnn.load_state_dict(torch.load("model_cache/model"))

correct = 0
for feature, target in zip(feature_list, target_list):
    out = cnn(feature)
    if out[0,0].item() > out[0,1].item():
        if target[0].item() == 0:
            correct += 1
    else:
        if target[0].item() == 1:
            correct += 1
print(f'acc {correct/len(feature_list)}') # 0.94