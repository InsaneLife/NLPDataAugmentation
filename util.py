#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/07/10 23:31:30
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
'''

# here put the import lib

def read_file(file_p):
    out_arr = []
    with open(file_p) as f:
        out_arr = [x.strip() for x in f.readlines()]
    return out_arr

