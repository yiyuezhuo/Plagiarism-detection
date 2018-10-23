# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:58:52 2018

@author: yiyuezhuo
"""

import json
import os
import numpy as np

import torch

def to_dict(fname, root_dir='embedding', target_dir='embedding_tensor'):
    word_list = []
    word_mat = None
    
    source_path = os.path.join(root_dir, fname)
    target_path = os.path.join(target_dir, fname)
    
    with open(source_path, encoding='utf8') as f:
        spec = f.readline().split(' ')
        num_word, dim_word = int(spec[0]),int(spec[1])
        word_mat = np.empty((num_word, dim_word))
        for i,line in enumerate(f):
            line = line.split(' ')
            word_list.append(line[0])
            word_mat[i,:] = np.array(line[1:-1]) # -1 for removing /n
    
    word_mat_tensor = torch.tensor(word_mat)
    
    word_pt = {'name':fname,
           'tensor':word_mat_tensor,
           'num_word':num_word,
           'dim_word':dim_word,
           'word_list':word_list}
    
    torch.save(word_pt, target_path)