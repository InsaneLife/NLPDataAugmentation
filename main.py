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

from bert_modify import modeling as modeling, tokenization, optimization

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


# nltk.download('wordnet')
if __name__ == "__main__":
    # 配置文件
    
    data_root = '/Volumes/HddData/ProjectData/NLP/bert/chinese_L-12_H-768_A-12/'
    # data_root = "/mnt/nlp/bert/chinese_wwm_L-12_H-768_A-12/"
    # data_root = "/mnt/nlp/bert/ERNIE_stable-1.0.1/checkpoints/"
    # data_root = '/mnt/nlp/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/'
    # data_root = '/mnt/nlp/bert/os_nlu_bert_base_100k/'
    # data_root = '/mnt/nlp/bert/RoBERTa-large-clue/'
    bert_config_file = data_root + 'bert_config.json'
    init_checkpoint = data_root + 'bert_model.ckpt'
    # init_checkpoint = data_root
    bert_vocab_file = data_root + 'vocab.txt'
    # bert_vocab_En_file = 'weight/uncased_L-12_H-768_A-12/vocab.txt'
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    # graph
    input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
    segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
    masked_lm_positions = tf.placeholder(tf.int32, shape=[None, None], name='masked_lm_positions')

    # 初始化BERT
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)


    masked_logits = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions)
    predict_prob = tf.nn.softmax(masked_logits, axis=-1)

     # 加载bert模型
    tvars = tf.trainable_variables()
    (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment)

    # # 获取最后一层和倒数第二层。

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
        query = '帮我查一下航班信息'
        query = '查一下航班信息'
        query = '附近有什么好玩的'
        split_tokens = token.tokenize(query)
        word_ids = token.convert_tokens_to_ids(split_tokens)
        word_ids.insert(0, token.convert_tokens_to_ids(["[CLS]"])[0])
        word_ids.append(token.convert_tokens_to_ids(["[SEP]"])[0])

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
            print("index", i)
            word_ids.insert(insert_index, token.convert_tokens_to_ids(["[MASK]"])[0])
            mask_lm_position = [insert_index]
            word_mask = [1] * len(word_ids)
            word_segment_ids = [0] * len(word_ids)
            fd = {input_ids: [word_ids], input_mask: [word_mask], segment_ids: [word_segment_ids], masked_lm_positions:[mask_lm_position]}
            mask_probs, last2 = sess.run([predict_prob, masked_logits], feed_dict=fd)
            for mask_prob in mask_probs:
                mask_prob = mask_prob.tolist()
                max_num_index_list = map(mask_prob.index, heapq.nlargest(3, mask_prob))
                for i in max_num_index_list:
                    words = token.id2vocab[i]
                    if words in punc:
                        continue
                    new_query = [x for x in query]
                    new_query.insert(insert_index-1, words)
                    print("".join(new_query))
                    # break
            print('-' * 50)
        pass
        
        # # 替换字符
        # for i in range(len(query)):
        #     insert_index = i + 1
        #     print("index", insert_index)
        #     word_ids[insert_index] = token.convert_tokens_to_ids(["[MASK]"])[0]
        #     mask_lm_position = [insert_index]
        #     word_mask = [1] * len(word_ids)
        #     word_segment_ids = [0] * len(word_ids)
        #     fd = {input_ids: [word_ids], input_mask: [word_mask], segment_ids: [word_segment_ids], masked_lm_positions:[mask_lm_position]}
        #     mask_probs, last2 = sess.run([predict_prob, masked_logits], feed_dict=fd)
        #     for mask_prob in mask_probs:
        #         mask_prob = mask_prob.tolist()
        #         max_num_index_list = map(mask_prob.index, heapq.nlargest(5, mask_prob))
        #         for i in max_num_index_list:
        #             words = token.id2vocab[i]
        #             if words in punc or words in ['[UNK]']:
        #                 continue
        #             new_query = [x for x in query]
        #             new_query[insert_index-1] = words
        #             print("".join(new_query))
        #             break
        #     print('-' * 50)
        #     pass
