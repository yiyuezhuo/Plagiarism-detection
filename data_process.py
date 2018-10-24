# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:05:58 2018

@author: yiyuezhuo
"""

import torch
import os
import json

class EmbeddingManager:
    def __init__(self, used_embedding, embedding_dir = 'embedding_tensor',
                 min_length = 5):
        self.used_embedding = used_embedding
        self.embedding_dir = embedding_dir
        self.min_length = min_length
        self.obj = None
        self.spec = torch.load(os.path.join(embedding_dir, used_embedding)+'.spec')
        self.word2idx_map = {word:idx for idx,word in enumerate(self.spec['word_list'])}
    def load_matrix(self):
        if self.obj is None:
            self.obj = torch.load(os.path.join(self.used_embedding, self.embedding_dir))
        return self.obj['tensor']
    def word2idx(self, word_vector):
        words = []
        for word in word_vector:
            if word in self.word2idx_map:
                words.append(self.word2idx_map[word])
        #if len(words) == 0:
        if len(words) < self.min_length:
            return None
        return torch.tensor([words])
    def load_data(self, data_path, verbose = True):
        with open(data_path, encoding='utf8') as f:
            data = json.load(f)
            
        word_list = data['word_list']
        label_list = data['label_list']
        
        feature_list = []
        target_list = []
        for i in range(len(word_list)):
            section, label = word_list[i], label_list[i]
            idxs = self.word2idx(section)
            #if idxs is not None and len(idxs[0]) >= self.min_length: # min size for current cnn
            if idxs is not None: # min size for current cnn
                feature_list.append(idxs)
                target_list.append(torch.LongTensor([label]))
        if verbose:
            print(f'size {len(word_list)} -> {len(feature_list)}')
            
        return feature_list, target_list



        