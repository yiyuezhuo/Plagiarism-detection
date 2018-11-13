# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:57:06 2018

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

class CNNText1D(nn.Module):
    # Here, I will use standard conv1d instead of conv2d and its lame appearance
    def __init__(self, embed_num, embed_dim, class_num = 2, 
                 kernel_num = 100, kernel_sizes = (3,4,5),
                 dropout = 0.5, pool_size = 3):
        super(CNNText1D, self).__init__()
        
        self.pool_size = pool_size
        
        self.embed = nn.Embedding(embed_num, embed_dim)
        # Conv1d 
        # Input: (N, C_{in}, L_{in})`  
        # Output: (N, C_{out}, L_{out})`
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, kernel_num, size) for size in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        #self.fc1 = nn.Linear
        self.conv_single = nn.Conv1d(kernel_num * len(kernel_sizes), class_num, 1)
    def forward(self, x):
        # x: (L, embed_num) 
        x = self.embed(x)  # x: (L, embed_dim)
        x = torch.transpose(x,0,1) # x: (embed_dim, L)
        x = x.unsqueeze(0) # x: (N, embed_dim, L) N = 1
        x = [F.relu(conv(x)) for conv in self.convs] #[(N, num_kernel, L-d),...]
        x = [F.max_pool1d(y, self.pool_size) for y in x] # [N, num_kernel, L-d']
        x = torch.cat(x, 1) 
        
        
        
        
    
class CNN1DText(nn.Module):
    def __init__(self, embed_num = 259922, embed_dim = 300, class_num = 2, kernel_num = 100, kernel_sizes = (3,4,5),
                dropout = 0.5):
        super(CNN1DText, self).__init__()
        
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        
        min_size = max(kernel_sizes)
        self.pad_sizes = [((size - min_size)//2 + (size -min_size)%2 , (size -min_size)//2)  for size in kernel_sizes]
        
        self.embed = nn.Embedding(embed_num, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, kernel_num, kernel_size) for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(kernel_num * len(kernel_sizes), class_num) # fake fc
    def use_pretrained_embedding(self, weights_matrix, non_trainable = True):
        self.embed.load_state_dict({'weight': weights_matrix})
        self.weight.requires_grad = not non_trainable
    def batch_forward(self, x):
        # x (L)
        x = self.embed(x) # (L, embed_dim)
        x = torch.unsqueeze(x.transpose(0,1),0) # (1, embed_dim, L)
        x = [conv(x) for conv in self.convs] # [(1, kernel_num, L'), (1, kernel_num, L''), ...]
        x = [F.pad(torch.squeeze(x[i], 0), self.pad_sizes[i]) for i in range(len(x))] # [(kernel_num, L'),...]
        x = torch.cat(x,0) # (len(kernel_sizes)*kernel_num, L')
        x = x.transpose(0,1) # (L', len(kernel_sizes)*kernel_num)
        x = self.dropout(x)
        x = self.fc(x) # (L', class_num)
        return x # logit. softmax(x) = probability output 
    def forward(self, x):
        # add and remove the extra N=1 dummy dimention to leverage other procedure which can only handle such form, 
        # such as those traning algorithm.
        # (1, L)
        x = self.batch_forward(torch.squeeze(x,0))
        return torch.unsqueeze(torch.max(x,0)[0],0) #(1, class_num)
