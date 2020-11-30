# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:32:32 2020

@author: hanyl
"""
import os
import time
from Sentence import Sentence
from Data import Data
from FilterDev import EntitySpliter, DictRecorder, SizingSystem, POSStater
from System import System, DictLookuper, Filter
from collections import Counter

'==1. 从带有Annotation的train数据集出发得到Dict_positive, Dict_negative, (4,27), Set_pos, 并部署给Filter应用。=='

s = System()

before = time.time()
d = Data(keys = ['train'], system = s)
print(time.time() - before) # ~2min
train_titles = d.getDataDict('train', 'T')
train_abstracts = d.getDataDict('train', 'A')

before = time.time()
es = EntitySpliter(False)
es.DevSplit(train_titles.values())
es.DevSplit(train_abstracts.values())
print(time.time() - before) # ~3min
# counter = es.getCounter()  # es.counter

counter_annotation = Counter()
for key in es.counter.keys():
    if key != 'COMMON':
        counter_annotation += es.counter[key]
counter_common = es.counter['COMMON']

dr = DictRecorder()
before = time.time()
dr.record(es.counter, [ key for key in es.counter.keys() if key != 'COMMON'], ['COMMON'],
            POSITIVE_COUNT = 5, NEGATIVE_COUNT = 5,
            RATIO_POS2NEG = 4, RATIO_NEG2POS = 16,
            POSITIVE_COVER_RATIO = 0.9, NEGATIVE_COVER_RATIO = 0.5, 
            VERBOSE = 0)
print(time.time() - before) # ~1.5s
# dr.dict_positive
# dr.dict_negative

ss = SizingSystem()
ss.sizing(counter_annotation)
ss.thresholdSetting(RATIO = 0.8, MIN_THRESHOLD = 4, MARGINAL_BENEFIT = 0.003, VERBOSE = 0)
# ss.thresholdSet

ps2 = POSStater()
ps2.stat(counter_annotation, counter_common, COVER_RATIO = 0.9, VERBOSE = 0)
# ps2.list_pos

'==2. 从FilterDev给到的结果出发，对新句子（这个是train中的句子，所以表现比较好...）进行探究=='
dl2 = DictLookuper(dr.dict_negative)
dl2.add_keywords_dict(dr.dict_positive)
esu = Filter(max_distinct_word = 3, pos_list = ps2.list_pos, thresholdSetting = ss.thresholdSet)

example_sentence = Sentence(train_abstracts['22616559'].text)
example_sentence.pos(system = s)
# example_sentence.POS_char
dl2.lookup([example_sentence])
# example_sentence.entities
esu.filt([example_sentence])
# example_sentence.entities


'==3. 从有标注的数据集出发，得到NER的训练数据集=='
dl = DictLookuper(dr.dict_negative)
dl.add_keywords_dict(dr.dict_positive)
esu = Filter(max_distinct_word = 3, pos_list = ps2.list_pos, thresholdSetting = ss.thresholdSet)

sentences = train_abstracts.values()
dl.lookup(sentences)
esu.filt(sentences)
# X, Y = generateTrainingData(list(sentences)[:3])

list(sentences)[3].entities
'==自贴 train-train=='
from TrainingData import *


X, Y = generateTrainingData(sentences, pos_list = ps2.list_pos, thresholdSetting = ss.thresholdSet)
saveTrainingData(X, Y, 'train-train')
saveTrainingDataCategory(X, Y, ['Chem', 'COMMON'])
strset_X = set([i for x in X
             for i in x])
len(strset_X)
strset_Xpost = set([i if i in all_letters else '*'
                    for x in X 
                        for i in x ])
len(X)
len(Y)
sum([ y == 1 for y in Y])
# 8848/15379
max([len(x) for x in X])
min([len(x) for x in X])

X[:5]
Y[:5]



'==互贴=='
# Sentence 和 dict 由互不相交的语料集构建。






