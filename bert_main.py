#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2020/06/14 17:45:13
@Author  :   zhiyang.zzy 
@Contact :   zhiyangchou@gmail.com
@Desc    :   
'''

# here put the import lib
import nltk
import tensorflow as tf
# from transformers import *
import heapq
from zhon.hanzi import punctuation
import string
import jieba
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
    def __init__(self, model_dir, n_best=3):
        self.n_best = n_best
        self.bert_config_file = model_dir + 'bert_config.json'
        self.init_checkpoint = model_dir + 'bert_model.ckpt'
        # init_checkpoint = model_dir
        self.bert_vocab_file = model_dir + 'vocab.txt'
        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
        self.token = tokenization.CharTokenizer(vocab_file=self.bert_vocab_file)
        self.build()
        self.build_sess()

    def __del__(self):
        self.close_sess()
    
    def build(self):
        # placeholder
        self.input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
        self.masked_lm_positions = tf.placeholder(tf.int32, shape=[None, None], name='masked_lm_positions')

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
        (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment)

    def build_sess(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def close_sess(self):
        self.sess.close()

    def predict(self, queries):
        out_map = defaultdict(list)
        for query in queries:
            split_tokens = self.token.tokenize(query)
            word_ids = self.token.convert_tokens_to_ids(split_tokens)
            word_ids.insert(0, self.token.convert_tokens_to_ids(["[CLS]"])[0])
            word_ids.append(self.token.convert_tokens_to_ids(["[SEP]"])[0])

            # 分词
            index_arr = [0]
            seg_list = jieba.cut(query, cut_all=False)
            i = 0
            for each in seg_list:
                i += len(each)
                index_arr.append(i)

            # 插入字符
            for i in index_arr:
                insert_index = i + 1
                # print("index", i)
                word_ids.insert(insert_index, self.token.convert_tokens_to_ids(["[MASK]"])[0])
                mask_lm_position = [insert_index]
                word_mask = [1] * len(word_ids)
                word_segment_ids = [0] * len(word_ids)
                fd = {self.input_ids: [word_ids], self.input_mask: [word_mask], self.segment_ids: [word_segment_ids], self.masked_lm_positions:[mask_lm_position]}
                mask_probs = self.sess.run(self.predict_prob, feed_dict=fd)
                for mask_prob in mask_probs:
                    mask_prob = mask_prob.tolist()
                    max_num_index_list = map(mask_prob.index, heapq.nlargest(self.n_best, mask_prob))
                    for i in max_num_index_list:
                        words = self.token.id2vocab[i]
                        if words in punc:
                            continue
                        new_query = [x for x in query]
                        new_query.insert(insert_index-1, words)
                        # print("".join(new_query))
                        out_map[query].append("".join(new_query))
            pass
        return out_map

if __name__ == "__main__":
    # bert 模型下载地址
    model_dir = '/Volumes/HddData/ProjectData/NLP/bert/chinese_L-12_H-768_A-12/'
    # query输入文件，每个query一行
    queries = read_file("data/input")
    mask_model = BertAugmentor(model_dir)
    result = mask_model.predict(queries)
    print("Augmentor's result:", result)
    # 写出到文件
    with open("data/bert_output", 'w', encoding='utf-8') as out:
        for query, v in result.items():
            out.write("{}\t{}\n".format(query, ';'.join(v)))

