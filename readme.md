# Data Augmentation
对SLU数据做数据增强，主要包括意图分类数据、槽位填充的数据。理论上分类数据也行。

TODO: 将[eda](https://arxiv.org/abs/1901.11196)中方法使用语言模型做增强，而非随机替换
- synonym replacement(SR)：随机选取句子中n个非停用词的词语。对于每个词语随机选取它的一个同义词替换该词语。
- random insertion(RI)：随机选取句子中的一个非停用词的词语，随机选取这个词语的一个近义词，将近义词随机插入到句子中，做n次。
- random swap(RS)：随机选取两个词语，交换他们的位置，做n次。
- random deletion(RD)：对于句子中的每个词语，以概率p选择删除。


# reference
- [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)
- https://github.com/jasonwei20/eda_nlp