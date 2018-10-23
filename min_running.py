# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 00:38:55 2018

@author: yiyuezhuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):
    # https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
    def __init__(self, args, embed = None):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        
        if embed is None:
            self.embed = nn.Embedding(V, D)
        else:
            self.embed = embed
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
    
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