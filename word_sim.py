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
import synonyms
import time
import nltk
# nltk.download('omw')

def load_w2v_fi():
    start = time.time()
    w2v_file = "/Volumes/HddData/ProjectData/NLP/wordEmbedding/cn_bi_fastnlp_100d/cn_bi_fastnlp_100d.txt"
    w2v_model = KeyedVectors.load_word2vec_format(w2v_file)
    print("load w2v file time cost", time.time() - start)
    return w2v_model

if __name__ == '__main__':
    
    word = "空调"
    # 词向量召回，词向量下载： https://docs.qq.com/sheet/DVnpkTnF6VW9UeXdh?tab=BB08J2
    w2v_model = load_w2v_fi()
    print(w2v_model.similar_by_word(word)[:10])
    # [('冷气', 0.832690954208374), ('暖气', 0.7806607484817505), ('电扇', 0.7694630026817322), ('电热', 0.7415034174919128), ('风扇', 0.7370954751968384), ('供暖', 0.7363734841346741), ('采暖', 0.7239724397659302), ('电暖', 0.7215089797973633), ('通风', 0.7174738645553589), ('隔音', 0.7118726968765259)]
    # wordNet 召回近义词
    for each in wordnet.synsets(word, lang='cmn'):
        print(each.lemma_names('cmn'), )
    # ['冷气机', '空调', '空调器', '空调装置', '空调设备']
    print(synonyms.nearby(word))
    # (['空调', '冷气', '空调设备', '空调系统', '波箱', '用车', '制冷', '空调机', '空气调节', '巴士在'], [1.0, 0.75175405, 0.7452018, 0.6877022, 0.6544307, 0.62812567, 0.62259305, 0.59779996, 0.57414114, 0.5611771])

