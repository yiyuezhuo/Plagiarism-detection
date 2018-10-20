# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:53:58 2018

@author: yiyuezhuo
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome()

'''
driver.get("http://www.baidu.com")
kw = driver.find_element_by_id('kw')
kw.send_keys('中国语')
su = driver.find_element_by_id('su')
su.click()
#driver.close()
'''

driver.get('https://www.sogou.com/')
query = driver.find_element_by_id('query')
query.send_keys('测试文本')
stb = driver.find_element_by_id('stb')
stb.click()
#driver.close()