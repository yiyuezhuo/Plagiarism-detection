# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 07:21:37 2018

@author: yiyuezhuo
"""

import os
import json


def generate_index():
    os.makedirs('index', exist_ok=True)
    for root,dir_list,file_list in os.walk('stat'):
        for fname in file_list:
            #source_path = os.path.join(root, fname)
            target_path = os.path.join('index', fname)
            with open(target_path, 'w') as _:
                pass

def convert_stat_to_data(stat_folder, target_path, verbose=True):
    record_list = []
    name_list = []
    for root, folders, fnames in os.walk(stat_folder):
        for fname in fnames:
            path = os.path.join(root, fname)
            with open(path, encoding='utf8') as f:
                record = json.load(f)
                name_list.append(path)
            record_list.append(record)
    
    count_list = [len([True for s in p if s['count'] > 0]) for p in record_list]
    
    import jieba
    
    word_list = [list(jieba.cut(s['text'])) for p in record_list for s in p]
    label_list = [s['count']>0 for p in record_list for s in p]
    
    if verbose:
        print(f'positive percent: {sum(label_list)/len(label_list)}')
        
    with open(target_path,'w',encoding='utf8') as f:
        json.dump({'word_list': word_list, 
                   'label_list': label_list,
                   'count_list': count_list,
                   'name_list': name_list}, f)
        
    if verbose:
        print(f'{stat_folder} => {target_path}')

def generate_doc(doc_path):
    import jieba
    
    with open(doc_path,encoding='utf8') as f:
        doc = f.read()
    len_doc = len(doc)
    word_list = []
    for i in range(len_doc - 25):
        text = doc[i:i+25]
        word_list.append(list(jieba.cut(text)))
    return word_list
    
def generate_doc_flatten(doc_path):
    import jieba
    
    with open(doc_path,encoding='utf8') as f:
        doc = f.read()
    return list(jieba.cut(doc))