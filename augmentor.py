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
from email.policy import default
import util
import eda_chinese as eda
import bert_main as bert
import back_translate as bt




class Augmentor(object):
    def __init__(self, model_dir:str):
        # model_dir: bert 预训练模型地址，中文bert下载链接：https://github.com/InsaneLife/ChineseNLPCorpus#%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%8D%E5%90%91%E9%87%8For%E6%A8%A1%E5%9E%8B
        self.mask_model = bert.BertAugmentor(model_dir)
        pass
    
    def bert_augment(self, file_:str):
        """ 
        file_: 输入文件，每行是一个query
        """
        queries = util.read_file(file_)
        # 随机替换：通过随机mask掉词语，预测可能的值。
        replace_result = self.mask_model.replace_word2queries(queries, beam_size=20)
        with open(file_ + ".augment.bert_replace", 'w', encoding='utf-8') as out:
            for query, v in replace_result.items():
                out.write("{}\t{}\n".format(query, ';'.join(v)))
        # 随机插入：通过随机插入mask，预测可能的词语
        insert_result = self.mask_model.insert_word2queries(queries, beam_size=20)
        print("Augmentor's result:", insert_result)
        # 写出到文件
        with open(file_ + ".augment.bert_insert", 'w', encoding='utf-8') as out:
            for query, v in insert_result.items():
                out.write("{}\t{}\n".format(query, ';'.join(v)))


    def augment(self, file_):
        # ead
        eda.augment(file_)
        # bert
        self.bert_augment(file_)
        # back translate
        bt.augment(file_)
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data/input", type=str, help="input file of unaugmented data")
    ap.add_argument("--bert_dir", default="/Volumes/HddData/ProjectData/NLP/bert/chinese_L-12_H-768_A-12/", type=str, help="input file of unaugmented data")
    ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
    ap.add_argument("--alpha", required=False, type=float, help="percent of words in each sentence to be changed")
    args = ap.parse_args()
    augmentor = Augmentor(args.bert_dir)
    # 数据增强
    augmentor.augment(args.input)
    pass



