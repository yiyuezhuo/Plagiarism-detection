# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 07:21:37 2018

@author: yiyuezhuo
"""

import os

def generate_index():
    os.makedirs('index', exist_ok=True)
    for root,dir_list,file_list in os.walk('stat'):
        for fname in file_list:
            #source_path = os.path.join(root, fname)
            target_path = os.path.join('index', fname)
            with open(target_path, 'w') as f:
                pass