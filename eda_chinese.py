#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/02/31 23:54:50
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   来源于：https://github.com/jasonwei20/eda_nlp, 原版本适用于英文，这里根据中文做了修改。修改如下：
1. 停用词使用中文。停用词来源：https://github.com/goto456/stopwords
2. 随机替换使用word_net/synonyms/词向量召回
3. 更多的同义词召回方法: word_net/synonyms/词向量召回

'''

# here put the import lib

# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
import jieba
from random import shuffle
import util
random.seed(1)

#使用中文的停用词，这里使用百度的，更多见 "./data/stopwords/"
_stop_words = util.read_file("./data/stopwords/baidu_stopwords.txt")

#cleaning up text
import re
def get_only_chars(line):
    # 英文的清洗逻辑，这块看自己需求用或者不用吧，但是中文肯定是不能用的。
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
# 同义词替换，原版本用的wordnet，也有中文的，第一次需要下载词库
########################################################################

import synonyms
import nltk
# 第一次使用请打开下面这一行，下载中文的wordnet
nltk.download('omw')
from nltk.corpus import wordnet 

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in _stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    # 这里使用了word_net + synonyms, 将两者的同义词召回做合并
    synonyms_word = set()
    for syn in wordnet.synsets(word, lang='cmn'):
        synonyms_word = set(syn.lemma_names('cmn'))
    for w in synonyms.nearby(word)[0]:
        synonyms_word.add(w)
    return list(synonyms_word)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    words = jieba.cut(sentence)
    words = " ".join(words)
    words = list(words.split())
    num_words = len(words)
    
    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1
    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    #sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(''.join(a_words))

    #ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(''.join(a_words))

    #rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(''.join(a_words))

    #rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(''.join(a_words))

    # augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    #trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    #append the original sentence
    augmented_sentences.append(sentence)

    return augmented_sentences

def augment(file_):
    """
    file_: 输入文件，每行是一个query
    """
    queries = util.read_file(file_)
    result = {}
    for query in queries:
        result[query] = eda(query)
    # 写出到文件
    with open(file_ + ".augment.eda", 'w', encoding='utf-8') as out:
        for query, v in result.items():
            out.write("{}\t{}\n".format(query, ';'.join(v)))
    pass

if __name__ == "__main__":
    # 打开文件
    queries = augment('./data/input')
