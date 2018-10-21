# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:13:18 2018

@author: yiyuezhuo
"""

    
import random
import requests
import webbrowser
from bs4 import BeautifulSoup
import time
import re
import os
import json


url = 'https://www.sogou.com/web'

'''
query: 绘画，油画等。图像可以记录、保存在纸质媒介、胶片等等对光信号
ie: utf8
_ast: 1540023552
_asf: null
w: 01029901
p: 40040100
dp: 1
cid: 
s_from: result_up
sut: 4870
sst0: 1540023670874
lkt: 0,0,0
sugsuv: 000417922495DCA55BB59C9FE7F9D285
sugtime: 1540023670874
'''

def fff(res):
    if isinstance(res, str):
        content = res.encode('utf8')
    elif isinstance(res, bytes):
        content = res
    else:
        content = res.content
    with open('temp.html', 'wb') as f:
        f.write(content)
    webbrowser.open('temp.html')



'''
test_path = 'extracted_text/AI 报告_许杰_117106010696.docx.txt'

with open(test_path, encoding='utf8') as f:
    text = f.read()


res = requests.get(url, params = {'query': '绘画，油画等。图像可以记录、保存在纸质媒介、胶片等等对光信号'})
soup = BeautifulSoup(s,'lxml')
'''

def preprocess(text):
    return text.replace(' ','')

class Downloader:
    def get(self, query):
        raise NotImplementedError

class RequestsDownloader(Downloader):
    def get(self, query):
        res = requests.get(url, {'query': query})
        page = res.content.decode()
        soup = BeautifulSoup(page, 'lxml')
        result = soup.select_one('.results').text

        return result
    
class SeleniumDownloader(Downloader):
    def __init__(self, delay = 5.1):
        self.is_setup = False
        self.driver = None
        self.delay = delay
    def setup(self):
        from selenium import webdriver
        #from selenium.webdriver.common.keys import Keys

        self.driver = webdriver.Chrome()
        self.is_setup = True
    def get(self, text):
        if not self.is_setup:
            self.setup()
        
        self.driver.get(self.host_url)
        query = self.driver.find_element_by_id(self.text_box_id)
        query.send_keys(text)
        stb = self.driver.find_element_by_id(self.click_id)
        stb.click()
        '''
        self.driver.implicitly_wait(self.delay) # seconds
        results = self.driver.find_element_by_class_name('results')
        return results.text
        '''
        return self.get_text()
    def get_text(self):
        raise NotImplemented
    
class SogouDownloader(SeleniumDownloader):
    host_url = 'https://www.sogou.com/'
    text_box_id = 'query'
    click_id = 'stb'
    def get_text(self):
        #self.driver.implicitly_wait(self.delay) # seconds
        time.sleep(self.delay)
        
        results = self.driver.find_element_by_class_name('results')
        return results.text

class BaiduDownloader(SeleniumDownloader):
    host_url = 'https://www.baidu.com/'
    text_box_id = 'kw'
    click_id = 'su'
    def get_text(self):
        #self.driver.implicitly_wait(self.delay) # seconds
        time.sleep(self.delay)
        
        page = self.driver.page_source
        soup = BeautifulSoup(page, 'lxml')
        content_list = []
        for div in soup.select('.result'):
            content_list.append(div.text)
        result = ''.join(content_list)
        return result

requests_downloader = RequestsDownloader()
sogou_downloader = SogouDownloader()
baidu_downloader = BaiduDownloader()

def check_doc(doc, check_iter = 30, check_text_size = 25, delay_base = 0.5, 
              delay_scale = 0.5, verbose = True):
    length = len(doc)
    check_list = []
    for i in range(check_iter):

        idx = random.randint(0,length - check_text_size)
        text = doc[idx:idx+check_text_size]
        if verbose:
            print(f"testing {text}")
        #legacy requests implementation
        #res = requests.get(url, {'query': text})
        #page = res.content.decode()
        #Wraped requests implementation
        #page = requests_downloader.get(text)
        
        # Wrap selenium implementation
        #result = baidu_downloader.get(text)
        #result = sogou_downloader.get(text)
        result = requests_downloader.get(text)
        #soup = BeautifulSoup(page, 'lxml')
        #result = soup.select_one('.results').text

        count = len(re.findall(re.escape(preprocess(text)), preprocess(result)))
        est = dict(text = text, result = result, count = count, check_text_size = check_text_size)
        check_list.append(est)
        if verbose:
            print(f'count: {count} {i+1}/{check_iter}')
        time.sleep(delay_base + delay_scale * random.random())
    return check_list

def stat_check(check_list):
    matched_num = len([check for check in check_list if check['count'] > 0])
    len_check_list = len(check_list) 
    print(f'matched: {matched_num}/{len_check_list}')

verbose = True

for root,folder_list,file_list in os.walk('extracted_text'):
    for fname in file_list:
        source_path = os.path.join(root, fname)
        target_path = os.path.join('stat', fname+'.json')
        index_path = os.path.join('index', fname+'.json')
        if os.path.isfile(target_path) or os.path.isfile(index_path):
            if verbose:
                print(f'skip {source_path}')
            continue
        if verbose:
            print(f"start checking {source_path}")
        with open(source_path, encoding='utf8') as f:
            doc = f.read()
            
        check_list = check_doc(doc, check_iter = 40)
        
        with open(target_path, 'w') as f:
            json.dump(check_list, f)
        
        if verbose:
            print(f'{source_path} -> {target_path}')