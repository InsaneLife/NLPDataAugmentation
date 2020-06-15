#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/06/14 22:52:45
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
1. 使用词向量做相似词语的召回，丰富ontology。
2. 使用wordNet做近义词召回。
'''

# here put the import lib
import gensim
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet
import time
import nltk
nltk.download('omw')

def load_w2v_fi():
    start = time.time()
    w2v_file = "/Volumes/HddData/ProjectData/NLP/wordEmbedding/cn_bi_fastnlp_100d/cn_bi_fastnlp_100d.txt"
    w2v_model = KeyedVectors.load_word2vec_format(w2v_file)
    print("load w2v file time cost", time.time() - start)
    return w2v_model

if __name__ == '__main__':
    
    word = "空调"
    # 词向量召回
    w2v_model = load_w2v_fi()
    print(w2v_model.similar_by_word(word)[:10])
    # [('冷气', 0.832690954208374), ('暖气', 0.7806607484817505), ('电扇', 0.7694630026817322), ('电热', 0.7415034174919128), ('风扇', 0.7370954751968384), ('供暖', 0.7363734841346741), ('采暖', 0.7239724397659302), ('电暖', 0.7215089797973633), ('通风', 0.7174738645553589), ('隔音', 0.7118726968765259)]
    # wordNet 召回近义词
    print(wordnet.synsets(word, lang='cmn'))

