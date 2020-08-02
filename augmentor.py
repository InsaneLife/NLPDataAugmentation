#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/06/14 17:50:03
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
todo: 将各中增强方法在此汇合和使用，输入文件，输出为各中方法增强后的结果。
'''

# here put the import lib
import argparse
from eda import *


class Augmentor(object):
    def __init__(self,):
        pass

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha", required=False, type=float, help="percent of words in each sentence to be changed")
args = ap.parse_args()


