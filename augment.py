#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/06/14 17:50:03
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
'''

# here put the import lib




from eda import *
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha", required=False, type=float, help="percent of words in each sentence to be changed")
args = ap.parse_args()
