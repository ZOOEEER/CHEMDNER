# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:55:25 2020

@author: hanyl
"""

# import os
import time
# from Sentence import Sentence
from Data import Data
from FilterDev import FilterDev
from System import System, Filter
# from collections import Counter

'==1. 从带有Annotation的train数据集出发得到Dict_positive, Dict_negative, (4,27), Set_pos, 并部署给Filter应用。=='

s = System()
fd = FilterDev()

before = time.time()
d = Data(keys = ['train'], system = s)
print(time.time() - before) # ~2min
train_titles = d.getDataDict('train', 'T')
train_abstracts = d.getDataDict('train', 'A')
sentences = list(train_titles.values()) + list(train_abstracts.values())

before = time.time()
fd.tune(sentences)
print(time.time() - before) # ~3min
# counter = es.getCounter()  # es.counter

dicts, list_pos, thresholdSetting = fd.getFilterSettings()
# dicts = fd.dictRecorder.dicts
# list_pos = fd.posStater.list_pos
# thresholdSetting = fd.sizingSystem.thresholdSet


'==2. 从FilterDev给到的结果出发，对新句子（这个是train中的句子，所以表现比较好...）进行探究=='
d2 = Data(keys = ['dev'], system = s)
dev_sens = list(d2.getDataDict('dev', 'T').values()) + list(d2.getDataDict('dev', 'A').values())
f = Filter(*fd.getFilterSettings())
# f = Filter(dicts, list_pos, thresholdSetting)
f.filt(dev_sens)
# dev_sens

'==3. 从有标注的数据集出发，得到NER的训练数据集=='
from TrainingData import getTrainingData, saveTrainingData
categories = getTrainingData(dev_sens, False, thresholdSetting)
saveTrainingData(categories)

'==2. 从FilterDev给到的结果出发，对新句子（这个是train中的句子，所以表现比较好...）进行探究=='
d3 = Data(keys = ['eva'], system = s)
test_sens = list(d3.getDataDict('eva', 'T').values()) + list(d3.getDataDict('eva', 'A').values())
# f = Filter(*fd.getFilterSettings())
# f = Filter(dicts, list_pos, thresholdSetting)
f.filt(test_sens)
# dev_sens

'==3. 从有标注的数据集出发，得到NER的训练数据集=='
# from TrainingData import getTrainingData, saveTrainingData
categories = getTrainingData(test_sens, False, thresholdSetting)
saveTrainingData(categories)
