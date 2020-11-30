# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:36:02 2020

@author: hanyl
"""


from Sentence import EmptyEntities
from Data import Data
from System import System, Filter
from Evaluation import evaluation_sen, evaluation_sens, causeFN, stat_enum
from Model import loadModel
from Classifier import ClassifierDev

"==For Data=="
s = System()
d = Data(keys = ['dev'], system = s)

dev = d.getSentences('dev')
backoff_0 = None
pred = EmptyEntities(dev) if not backoff_0 else EmptyEntities(backoff_0)
backoff_0 = EmptyEntities(pred)


"==For Filter=="
filter_name = 'train-POS5_Star-20201122'
f = Filter.SettingFile(filter_name)


'====Evaluation===='
f.filt(pred)
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
from Classifier import Classifier
# from Classifier import Mapper
model_name = 'GRU_128_3-30'
classifier = Classifier(category_names = ['chem', 'COMMON'], model = loadModel(model_name))
# classifer.model.to('cpu')
classifier.mentionToType('ethanol')

# for t in [0.01, 0.16, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.66, 0.75, 0.8, 0.83, 1]:
#     print('threshold:{}'.format(t))
#     classifer.mapper = Mapper(dict_point = 3, pred_point = 1, threshold = t)
#     pred = copy.deepcopy(backoff_1)
#     classifer.predict(pred)
#     conf = evaluation_sens(dev, pred)

from Classifier import MapperNone
classifier.mapper = MapperNone()

pred = copy.deepcopy(backoff_1)

classifier.predict(pred)

conf = evaluation_sens(dev, pred)

'''
(dict_point = 3, pred_point = 1, threshold = 0.01)
      Pred-Pos | Pred-Neg | Total
Pos     26776      2750     29526
Neg   132616         0    132616
Total   159392      2750    162142
90.7%
16.8%
'''


'''
(dict_point = 3, pred_point = 1, threshold = 0.5)
      Pred-Pos | Pred-Neg | Total
Pos     25692      3834     29526
Neg   111445         0    111445
Total   137137      3834    140971
87.0%
18.7%
'''

backoff_2 = copy.deepcopy(pred)
'''
No map.
      Pred-Pos | Pred-Neg | Total
Pos     23866      5660     29526
Neg    34165         0     34165
Total    58031      5660     63691
80.8%
41.1%
'''
 
a = 3911
dev[a].entities
pred[a].entities
backoff_1[a].entities



'----检查Mapper----'
# classifer.mapper = Mapper(dict_point = 3, pred_point = 1, threshold = 0.01)
# # pred = copy.deepcopy(backoff_1)
# example_sentence = [copy.deepcopy(backoff_1[a])]
# classifer.predict(example_sentence)












