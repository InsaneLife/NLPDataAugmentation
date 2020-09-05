#!/usr/bin/env python
# encoding=utf-8
'''
@Time    :   2020/06/14 17:45:13
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
1. 随机插入mask，使用bert来生成 mask 的内容，来丰富句子
2. 随机将某些词语mask，使用bert来生成 mask 的内容。
    - 使用贪心算法，每次最优。
    - beam search方法，每次保留最优的前n个，最多num_beams个句子。(注意句子数据大于num_beams个时候，剔除概率最低的，防止内存溢出)。
'''

# here put the import lib
import nltk
import tensorflow as tf
# from transformers import *
import heapq
from zhon.hanzi import punctuation
import string
import jieba
import numpy as np
from util import read_file
from bert_modify import modeling as modeling, tokenization, optimization
from collections import defaultdict
print(tf.__version__)
punc = string.punctuation + punctuation


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

    return logits


class BertAugmentor(object):
    def __init__(self, model_dir, beam_size=5):
        self.beam_size = beam_size    # 每个带mask的句子最多生成 beam_size 个。
        # bert的配置文件
        self.bert_config_file = model_dir + 'bert_config.json'
        self.init_checkpoint = model_dir + 'bert_model.ckpt'
        # init_checkpoint = model_dir
        self.bert_vocab_file = model_dir + 'vocab.txt'
        self.bert_config = modeling.BertConfig.from_json_file(
            self.bert_config_file)
        # token策略，由于是中文，使用了token分割，同时对于数字和英文使用char分割。
        self.token = tokenization.CharTokenizer(vocab_file=self.bert_vocab_file)
        self.mask_token = "[MASK]"
        self.mask_id = self.token.convert_tokens_to_ids([self.mask_token])[0]
        self.cls_token = "[CLS]"
        self.cls_id = self.token.convert_tokens_to_ids([self.cls_token])[0]
        self.sep_token = "[SEP]"
        self.sep_id = self.token.convert_tokens_to_ids([self.sep_token])[0]
        # 构图
        self.build()
        # sess init
        self.build_sess()

    def __del__(self):
        # 析构函数
        self.close_sess()

    def build(self):
        # placeholder
        self.input_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(
            tf.int32, shape=[None, None], name='input_masks')
        self.segment_ids = tf.placeholder(
            tf.int32, shape=[None, None], name='segment_ids')
        self.masked_lm_positions = tf.placeholder(
            tf.int32, shape=[None, None], name='masked_lm_positions')

        # 初始化BERT
        self.model = modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        self.masked_logits = get_masked_lm_output(
            self.bert_config, self.model.get_sequence_output(), self.model.get_embedding_table(),
            self.masked_lm_positions)
        self.predict_prob = tf.nn.softmax(self.masked_logits, axis=-1)

        # 加载bert模型
        tvars = tf.trainable_variables()
        (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment)

    def build_sess(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def close_sess(self):
        self.sess.close()

    def predict_single_mask(self, word_ids:list, mask_index:int, prob:float=None):
        """输入一个句子token id list，对其中第mask_index个的mask的可能内容，返回 self.beam_size 个候选词语，以及prob"""
        word_ids_out = []
        word_mask = [1] * len(word_ids)
        word_segment_ids = [0] * len(word_ids)
        fd = {self.input_ids: [word_ids], self.input_mask: [word_mask], self.segment_ids: [
            word_segment_ids], self.masked_lm_positions: [[mask_index]]}
        mask_probs = self.sess.run(self.predict_prob, feed_dict=fd)
        for mask_prob in mask_probs:
            mask_prob = mask_prob.tolist()
            max_num_index_list = map(mask_prob.index, heapq.nlargest(self.beam_size, mask_prob))
            for i in max_num_index_list:
                if prob and mask_prob[i] < prob:
                    continue
                cur_word_ids = word_ids.copy()
                cur_word_ids[mask_index] = i
                word_ids_out.append([cur_word_ids, mask_prob[i]])
        return word_ids_out
    
    def predict_batch_mask(self, query_ids:list, mask_indexes:int, prob:float=0.5):
        """输入多个token id list，对其中第mask_index个的mask的可能内容，返回 self.beam_size 个候选词语，以及prob
        word_ids: [word_ids1:list, ], shape=[batch, query_lenght]
        mask_indexes: query要预测的mask_id, [[mask_id], ...], shape=[batch, 1, 1]
        """
        word_ids_out = []
        word_mask = [[1] * len(x) for x in query_ids]
        word_segment_ids = [[1] * len(x) for x in query_ids]
        fd = {self.input_ids: query_ids, self.input_mask: word_mask, self.segment_ids: 
              word_segment_ids, self.masked_lm_positions: mask_indexes}
        mask_probs = self.sess.run(self.predict_prob, feed_dict=fd)
        for mask_prob, word_ids_, mask_index in zip(mask_probs, query_ids, mask_indexes):
            # each query of batch
            cur_out = []
            mask_prob = mask_prob.tolist()
            max_num_index_list = map(mask_prob.index, heapq.nlargest(self.n_best, mask_prob))
            for i in max_num_index_list:
                cur_word_ids = word_ids_.copy()
                cur_word_ids[mask_index[0]] = i
                cur_out.append([cur_word_ids, mask_prob[i]])
            word_ids_out.append(cur_out)
        return word_ids_out

    def gen_sen(self, word_ids:list, indexes:list):
        """
        输入是一个word id list, 其中包含mask，对mask生产对应的词语。
        因为每个query的mask数量不一致，预测测试不一致，需要单独预测
        """
        out_arr = []
        for i, index_ in enumerate(indexes):
            if i == 0:
                out_arr = self.predict_single_mask(word_ids, index_)
            else:
                tmp_arr = out_arr.copy()
                out_arr = []
                for word_ids_, prob in tmp_arr:
                    cur_arr = self.predict_single_mask(word_ids_, index_)
                    cur_arr = [[x[0], x[1] * prob] for x in cur_arr]
                    out_arr.extend(cur_arr)
                # 筛选前beam size个
                out_arr = sorted(out_arr, key=lambda x: x[1], reverse=True)[:self.beam_size]
        for i, (each, _) in enumerate(out_arr):
            query_ = [self.token.id2vocab[x] for x in each]
            out_arr[i][0] = query_
        return out_arr

    def word_insert(self, query):
        """随机将某些词语mask，使用bert来生成 mask 的内容。
        
        max_query： 所有query最多生成的个数。
        """
        out_arr = []
        seg_list = jieba.cut(query, cut_all=False)
        # 随机选择非停用词mask。
        i, index_arr = 1, [1]
        for each in seg_list:
            i += len(each)
            index_arr.append(i)
        # query转id
        split_tokens = self.token.tokenize(query)
        word_ids = self.token.convert_tokens_to_ids(split_tokens)
        word_ids.insert(0, self.cls_id)
        word_ids.append(self.sep_id)
        word_ids_arr, word_index_arr = [], []
        # 随机insert n 个字符, 1<=n<=3
        for index_ in index_arr:
            insert_num = np.random.randint(1, 4)
            word_ids_ = word_ids.copy()
            word_index = []
            for i in range(insert_num):
                word_ids_.insert(index_, self.mask_id)
                word_index.append(index_ + i)
            word_ids_arr.append(word_ids_)
            word_index_arr.append(word_index)
        for word_ids, word_index in zip(word_ids_arr, word_index_arr):
            arr_ = self.gen_sen(word_ids, indexes=word_index)
            out_arr.extend(arr_)
            pass
        # 这个是所有生成的句子中，筛选出前 beam size 个。
        out_arr = sorted(out_arr, key=lambda x: x[1], reverse=True)
        out_arr = ["".join(x[0][1:-1]) for x in out_arr[:self.beam_size]]
        return out_arr
    
    def word_replace(self, query):
        """随机将某些词语mask，使用bert来生成 mask 的内容。"""
        out_arr = []
        seg_list = jieba.cut(query, cut_all=False)
        # 随机选择非停用词mask。
        i, index_map = 1, {}
        for each in seg_list:
            index_map[i] = len(each)
            i += len(each)
        # query转id
        split_tokens = self.token.tokenize(query)
        word_ids = self.token.convert_tokens_to_ids(split_tokens)
        word_ids.insert(0, self.cls_id)
        word_ids.append(self.sep_id)
        word_ids_arr, word_index_arr = [], []
        # 依次mask词语，
        for index_, word_len in index_map.items():
            word_ids_ = word_ids.copy()
            word_index = []
            for i in range(word_len):
                word_ids_[index_ + i] = self.mask_id
                word_index.append(index_ + i)
            word_ids_arr.append(word_ids_)
            word_index_arr.append(word_index)
        for word_ids, word_index in zip(word_ids_arr, word_index_arr):
            arr_ = self.gen_sen(word_ids, indexes=word_index)
            out_arr.extend(arr_)
            pass
        out_arr = sorted(out_arr, key=lambda x: x[1], reverse=True)
        out_arr = ["".join(x[0][1:-1]) for x in out_arr[:self.beam_size]]
        return out_arr

    def insert_word2queries(self, queries:list, beam_size=10):
        self.beam_size = beam_size
        out_map = defaultdict(list)
        for query in queries:
            out_map[query] = self.word_insert(query)
        return out_map

    def replace_word2queries(self, queries:list, beam_size=10):
        self.beam_size = beam_size
        out_map = defaultdict(list)
        for query in queries:
            out_map[query] = self.word_replace(query)
        return out_map

    def predict(self, query_arr, beam_size=None):
        """
        query_arr: ["w1", "w2", "[MASK]", ...], shape=[word_len]
        每个query_arr, 都会返回beam_size个
        """
        self.beam_size = beam_size if beam_size else self.beam_size
        word_ids, indexes = self.token.convert_tokens_to_ids(query_arr), [x[0] for x in filter(lambda x: x[1] == self.mask_token, enumerate(query_arr))]
        out_queries = self.gen_sen(word_ids, indexes)
        out_queries = [["".join(x[0]), x[1]] for x in out_queries]
        return out_queries


if __name__ == "__main__":
    # bert 模型下载地址，中文bert下载链接：https://github.com/InsaneLife/ChineseNLPCorpus#%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%8D%E5%90%91%E9%87%8For%E6%A8%A1%E5%9E%8B
    model_dir = '/Volumes/HddData/ProjectData/NLP/bert/chinese_L-12_H-768_A-12/'
    # query输入文件，每个query一行
    queries = read_file("data/input")
    mask_model = BertAugmentor(model_dir)
    # bert 预测 mask
    out_queries = mask_model.predict(["[MASK]", "[MASK]", "卖", "账", "号", "吗"], beam_size=5)
    # 随机替换：通过随机mask掉词语，预测可能的值。
    replace_result = mask_model.replace_word2queries(queries, beam_size=20)
    with open("data/bert_replace", 'w', encoding='utf-8') as out:
        for query, v in replace_result.items():
            out.write("{}\t{}\n".format(query, ';'.join(v)))
    # 随机插入：通过随机插入mask，预测可能的词语, todo: 将随机插入变为beam search
    insert_result = mask_model.insert_word2queries(queries, beam_size=20)
    print("Augmentor's result:", insert_result)
    # 写出到文件
    with open("data/bert_insert", 'w', encoding='utf-8') as out:
        for query, v in insert_result.items():
            out.write("{}\t{}\n".format(query, ';'.join(v)))
