# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:36:02 2020
一个交叉Filt，然后将数据整理成训练格式，供model学习的示例。
@author: hanyl
"""


from Sentence import EmptyEntities
from Data import Data
from System import System, Filter
from FilterDev import FilterDev
from TrainingData import TrainingData
from Classifier import ClassifierDev, Classifier
from Model import saveModel, loadModel

'===== 1. Data ====='
s = System()
d = Data(keys = ['train', 'dev'], system = s)
train = d.getSentences('train')
backoff_0 = None
train_for_td = EmptyEntities(train) if not backoff_0 else EmptyEntities(backoff_0)
backoff_0 = EmptyEntities(train_for_td)

'===== 2. FilterDev ====='
# len(train) # 7000 前一半是标题，后一半是摘要
names = ['Split2\\train1', 'Split2\\train2']
fds = [ FilterDev(name = name) for name in names ]
fds[0].tune(train[:1750] + train[3500:3500+1750])
fds[1].tune(train[1750:3500] + train[3500+1750:])
for fd in fds:
    fd.writeToFile()
'===== 3. Filter ====='
fs = [Filter.readFromFile(path = name) for name in names]
train_for_tds = [ train_for_td[:1750] + train_for_td[3500:3500+1750], 
                  train_for_td[1750:3500] + train_for_td[3500+1750:] 
                  ]
for i in range(len(train_for_tds)):
    fs[i].filt(train_for_tds[i])

'=====4. TrainingData====='
tds = [ TrainingData(DirPath = name) for name in names ]
for i in range(len(train_for_tds)):
    tds[i].getTrainingData(train_for_tds[i])
for i in range(len(train_for_tds)):    
    tds[i].saveTrainingData()
    

'=====5. ClassifierDev======'
model_name = 'GRU_128_3-30'
new_model_name = 'GRU_128_3-30-Split2_10_10'
cd = ClassifierDev(name = new_model_name, model_name = model_name)
cd.setData(names[0])
cd.train()
cd.evaluation_training_process()
cd.setData(names[1])
cd.train()
cd.evaluation_training_process()
saveModel(cd.model, cd.name)

'=====6. Classifier / Evaluation on Dev======'
"--For Data--"
dev = d.getSentences('dev')
backoff_d = None
pred = EmptyEntities(dev) if not backoff_0 else EmptyEntities(backoff_d)
backoff_d = EmptyEntities(pred)
"--For Filter--"
filter_name = 'train-POS5_Star-20201122'
f = Filter.SettingFile(filter_name)
'--Evaluation--'
f.filt(pred)
from Evaluation import evaluation_sens, causeFN
conf = evaluation_sens(dev, pred)

import copy
backoff_1 = copy.deepcopy(pred)
'''
# With Refiner/POSGatherStar
      Pred-Pos | Pred-Neg | Total
Pos     26829      2697     29526
Neg   137469         0    137469
Total   164298      2697    166995
90.9%
16.3%
'''
efn = causeFN(conf, f)
# '''
# # With Refiner/POSGatherStar
#         cA  664 2.2%
#         Ac  994 3.4%
#   FalsePOS 1035 3.5%
#       len4  308 1.0%
#      len27  320 1.1%
#     InDict  409 1.4%
#      Other  191 0.6%
# Stat of Pure Causes for FN:
#         cA  255
#         Ac  408
#   FalsePOS  608
#       len4   56
#      len27    0
#     InDict  265
#      Other  121
#      Total 1713

# '''
# stat_enum(entitiesFN = efn, myfilter = f, num_samples = 20, key = 'len4')

'----dev----'

# cd = ClassiferDev()
# cd.name = 'GRU_128_3-30'
# cd.setModel(loadModel(cd.name))

'====prediction===='
# from Classifier import ClassifierDev
# cd = ClassifierDev()
# cd.trainingdata.all_categories
# from Classifier import Mapper
model_name = 'GRU_128_3-30'
model_name = 'GRU_128_3-30-Split2_10_10'
classifier = Classifier(category_names = ['chem', 'COMMON'], model = loadModel(model_name))
# classifer.model.to('cpu')
classifier.mentionToType('apple')

pred = copy.deepcopy(backoff_d)
classifier.predict(pred)
conf = evaluation_sens(dev, pred)
'''
No map.
      Pred-Pos | Pred-Neg | Total
Pos     23866      5660     29526
Neg    34165         0     34165
Total    58031      5660     63691
80.8%
41.1%
'''
backoff_2 = copy.deepcopy(pred)

from Evaluation import causeFP
cfp = causeFP(conf)
'''
       out 24528 42.3%
      tptp   56 0.1%
      tppt 2746 4.7%
       btp  459 0.8%
       bpt 3037 5.2%
      ptpt   76 0.1%
      pttp  173 0.3%
       ptb  399 0.7%
       tpb 2865 4.9%
Stat of Pure Causes for FN:
       out 24528
      tptp   38
      tppt 2735
       btp  331
       bpt 3025
      ptpt   52
      pttp  124
       ptb  302
       tpb 2862
     Total 33997
'''

from Evaluation import evaluationChar_sens
conf_star = evaluationChar_sens(dev, pred)
'''
# Char-level Evaluation.
tp_ratio:87.0% = 257158/295520
fp_ratio:68.0% = 257158/378299
fp2_ratio:49.1% = 257158/524229
'''




'----检查Mapper----'
# classifer.mapper = Mapper(dict_point = 3, pred_point = 1, threshold = 0.01)
# # pred = copy.deepcopy(backoff_1)
# example_sentence = [copy.deepcopy(backoff_1[a])]
# classifer.predict(example_sentence)
# for t in [0.01, 0.16, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.66, 0.75, 0.8, 0.83, 1]:
#     print('threshold:{}'.format(t))
#     classifer.mapper = Mapper(dict_point = 3, pred_point = 1, threshold = t)
#     pred = copy.deepcopy(backoff_1)
#     classifer.predict(pred)
#     conf = evaluation_sens(dev, pred)

# from Classifier import MapperNone
# classifier.mapper = MapperNone()
'----换两个模型----'


