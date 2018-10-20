# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 23:21:50 2018

@author: yiyuezhuo
"""

import os
import textract
import fnmatch, os, sys, win32com.client



#import win32com.client


class DummyException(Exception):
    pass

class TypeMismatch(Exception):
    pass

source_root = 'data'
target_root = 'extracted_text'


class DocExtractor:
    '''
    Holy crap, There's someone still using doc file.
    The antiword used by textract seems not working in windows, this 
    function can be refer to Python Cookbook:
    https://www.oreilly.com/library/view/python-cookbook-2nd/0596007973/ch02s28.html
    '''

    def __init__(self, cache_name = 'temp_buffer.txt'):
        self.wordapp = None
        self.is_setup = False
        self.cache_name = cache_name
        
    def setup(self):
        self.wordapp = win32com.client.gencache.EnsureDispatch("Word.Application")
        #self.FileFormat = win32com.client.constants.wdFormatText
        self.is_setup = True
    def close(self):
        if not self.is_setup:
            return
        self.wordapp.Quit()
        
    def process(self, path):
        if not self.is_setup:
            self.setup()
            
        path = os.path.abspath(path)
        cache_name = os.path.abspath(self.cache_name)
        
        self.wordapp.Documents.Open(path)
        #print(cache_name)
        self.wordapp.ActiveDocument.SaveAs(cache_name,
            FileFormat = win32com.client.constants.wdFormatText)
        self.wordapp.ActiveDocument.Close()
        
        with open(cache_name, 'rb') as f:
            txt = f.read()
        with open(cache_name, 'wb') as f:
            f.write(b'')
        return txt    
    
doc_extractor = DocExtractor()

def extract(path):
    if path.endswith('.doc'):
        text = doc_extractor.process(path)
    elif path.endswith('.zip') or path.endswith('.rar'):
        print(f'{path} is archive file, be sure that they are extracted before procssing')
        raise TypeMismatch
    else:
        text = textract.process(path)
    return text

# extract all archived file firstly
import zipfile 
for root, dir_names, file_names in os.walk(source_root):
    for file_name in file_names:
        path = os.path.join(root, file_name)
        if path.endswith('.zip'):
            zip_ref = zipfile.ZipFile(path, 'r')
            zip_ref.extractall(root)
            zip_ref.close()

for root, dir_names, file_names in os.walk(source_root):
    for file_name in file_names:
        path = os.path.join(root, file_name)
        target_path = os.path.join(target_root, file_name) + '.txt'
        try:
            text = extract(path)
        except TypeMismatch:
            print(f'skip {path}')
            continue
        except Exception:
            print(f'fail to extract {path}')
            continue
        try:
            text = text.decode()
        except:
            text = text.decode('gbk')
        with open(target_path, 'w', encoding = 'utf8') as f:
            f.write(text)