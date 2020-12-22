# -*- coding: utf-8 -*- 
# @File gen_synonyms.py
# @Time 2020/12/7 16:49
# @Author wcy
# @Software: PyCharm
# @Site
import os
from pprint import pprint

from simbert import gen_synonyms
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    a = gen_synonyms("训练是不是只用了seq2seq mask这种啊", 10, 10)
    print(a)
    pprint(gen_synonyms("夏天是什么样的季节？", 30, 10, topp=0.8))