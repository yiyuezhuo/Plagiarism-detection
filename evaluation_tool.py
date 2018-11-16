# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:27:17 2018

@author: yiyuezhuo
"""

import torch
from utils import generate_doc, generate_doc_flatten
import torch.nn.functional as F

class EvaluationManager:
    def __init__(self, model, embedding_manager, cuda = True):
        self.model = model
        self.embedding_manager = embedding_manager
        self.cuda = cuda

    def get_acc(self, feature_list, target_list):
        model = self.model
    
        correct = 0
        for feature, target in zip(feature_list, target_list):
            out = model(feature)
            if out[0,0].item() > out[0,1].item():
                if target[0].item() == 0:
                    correct += 1
            else:
                if target[0].item() == 1:
                    correct += 1
        return correct/len(feature_list)
    
    def confuse_matrix(self, feature_list, target_list):
        model = self.model
        
        mat = torch.zeros(2,2,dtype=torch.int64)
        for feature, target in zip(feature_list, target_list):
            out = model(feature)
            if target[0].item() == 0:
                if out[0,0].item() > out[0,1].item():
                    mat[0,0] += 1
                else:
                    mat[0,1] += 1
            else:
                if out[0,0].item() > out[0,1].item():
                    mat[1,0] += 1
                else:
                    mat[1,1] += 1
        return mat
    
    def analysis_doc(self, doc_path):
        embedding_manager = self.embedding_manager
        model = self.model
        
        doc = generate_doc(doc_path)
        doc_section = ([embedding_manager.word2idx(section) for section in doc])
        if self.cuda:
            doc_section = [t.cuda() if t is not None else None for t in doc_section]
        print(f'none count: {len([t for t in doc_section if t is None])} / {len(doc_section)}')
        
        plag_prob_list = []
        for section in doc_section:
            if section is not None:
                pred_plag_prob = F.softmax(model(section),1)[0,1].item()
                plag_prob_list.append(pred_plag_prob)
            else:
                plag_prob_list.append(0)
        return plag_prob_list
    
    def analysis_doc_batch(self, doc_path):
        embedding_manager = self.embedding_manager
        model = self.model
        
        doc = generate_doc_flatten(doc_path)
        doc_idxs = embedding_manager.word2idx(doc)
        if self.cuda:
            doc_idxs = doc_idxs.cuda()
        
        
        res = model(doc_idxs)
        
        print(f'origin:{len(doc)} encoded: {doc_idxs.size()} output: {res.size()}')

        
        return F.softmax(res, 1)[0,1,:]
        
