# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:57:02 2020

@author: hanyl
"""

# import time

from Sentence import EmptyEntities
from Data import Data
from System import System, Filter
from Evaluation import evaluation_sens, causeFN, stat_enum
from FilterDev import FilterDev



"==For Data=="
s = System()
d = Data(keys = ['train', 'dev'], system = s)
train = d.getSentences('train')
dev = d.getSentences('dev')
backoff_1 = None
pred = EmptyEntities(train) if not backoff_1 else EmptyEntities(backoff_1)
backoff_1 = EmptyEntities(train)
backoff_0 = None
pred = EmptyEntities(dev) if not backoff_0 else EmptyEntities(backoff_0)
backoff_0 = EmptyEntities(pred)


# # For eva data
# d.readData(['eva'], s)
# eva = list(d.getDataDict('eva', 'T').values()) + list(d.getDataDict('eva', 'A').values())
# eva_pred = EmptyEntities(eva)

"==For Filter=="
filter_name = 'train-POS5_Star-20201122'
f = Filter.SettingFile(filter_name)



"==Evaluation-train=="

train_pred = EmptyEntities(train)
f.filt(train_pred)
conf_train = evaluation_sens(train, train_pred)
efn_train = causeFN(conf_train, f)

'''
      Pred-Pos | Pred-Neg | Total
Pos     28983       495     29478
Neg    41423         0     41423
Total    70406       495     70901
98.3%
41.2%
        cA  125 0.4%
        Ac  343 1.2%
  FalsePOS   46 0.2%
      len4  211 0.7%
     len27    5 0.0%
    InDict   21 0.1%
     Other   27 0.1%
Stat of Pure Causes for FN:
        cA   38
        Ac  152
  FalsePOS   10
      len4   46
     len27    0
    InDict    0
     Other   27
     Total  273
'''

'====trainingData===='
from TrainingData import TrainingData
td = TrainingData()
td.getTrainingData(sentences=train_pred)
td.saveTrainingData()

td2 = TrainingData()
td2.loadTrainingData()
# sample(td2.categories['chem'],3)
from TrainingData import TrainingData
td3 = TrainingData()
td3.loadTrainingData()
for key in td3.all_categories:
    td3.categories[key] = td3.categories[key][:10]
td3.loadDataLoaders()
sample_lines = td3.sample('train', 20)
sample_lines

td3.vectorizer.lineToTensor(sample_lines[0][0]).size()
x_tensor, y = td3.sampleTensors(32)
x_tensor.size()
y.size()

from classifier import classiferDev
cd = classiferDev()
cd.batch_size = 128
cd.num_epochs = 4
cd.lr = 0.0001
cd.train()
from TrainingData import saveModel, loadModel
saveModel(cd.model, 'rnn2-128-train-train-5')

cd.model = loadModel('rnn2-128-train-train-5')
saveModel(cd.model, 'rnn2-128-train-train-10')

from Model import LSTM
cd2 = classiferDev()
cd2.model = LSTM(input_size = 72, hidden_size = 256, output_size = 2, num_layers = 1)
cd2.batch_size = 128
cd2.num_epochs = 1
cd2.lr = 1
cd2.train()

cd3 = classiferDev()
cd3.model = LSTM(input_size = 72, hidden_size = 128, output_size = 2, num_layers = 1)
cd3.batch_size = 128
cd3.num_epochs = 4
cd3.lr = 0.00001
cd3.train()

cd4 = classiferDev()
cd4.model = LSTM(input_size = 72, hidden_size = 128, output_size = 2, num_layers = 2)
cd4.batch_size = 128
cd4.num_epochs = 1
cd4.lr = 1
cd4.train()

import torch.nn as nn
from Model import GRU # rnn
cd5 = classiferDev()
cd5.setModel(GRU(input_size = 72, hidden_size = 128, output_size = 2, num_layers = 1))
cd5.batch_size = 128
cd5.num_epochs = 1
cd5.lr = 0.0001
cd5.train()



cd6 = classiferDev()
cd6.model
import torch.nn as nn
cd6.model.relu = nn.Tanh()
cd6.train()

from classifier import classiferDev
from Model import RNN2torch
cd7 = classiferDev()
cd7.model = RNN2torch(input_size = 72, hidden_size = 128, output_size = 2)
cd7.lr = 0.0001
cd7.train()

from Model import RNN2_nolsm
cd8 = classiferDev()
# cd8.device = 'cpu'
cd8.setModel(RNN2_nolsm(input_size = 72, hidden_size = 128, output_size = 2))
cd5.setModel(GRU(input_size = 72, hidden_size = 128, output_size = 2, num_layers = 1))

cd8.lr = 0.0001
cd8.train()











'===草稿==='
fp = conf_train['fp']
len(conf_train['fp'])
from random import sample
examples = sample(fp, 1)
example = examples[0]
print(example[0])
print(example[1].text)
print('# Truth:')
for e in example[1].entities:
    print(e)
print('# Pred:')
for e in example[2].entities:
    print(e, e.pos)

f.dictLookuper.word('Octanal')
from Sentence import Sentence
sample_sentence = Sentence('A new and efficient synthesis of a naturally occurring amide alkaloid, N-isobutyl-4,5-epoxy-2(E)-decenamide isolated from the roots of Piper nigrum has been described involving a total of nine steps. Octanal and 2-bromoacetic acid have been used as the starting materials.')
sample_sentence.pos(s)
f.bl_sifter = True
f.sifter.bl_list_pos_star = False
f.bl_refiner = True

f.filt([sample_sentence])


for e in sample_sentence.entities:
    print(e)

# def smash(sentences):
#     for sentence in sentences:
#         _smash(sentence, )


'====Evaluation-dev===='
# f.filt(pred)
# conf = evaluation_sens(dev, pred)

# '''
# # With Refiner/POSGatherStar
#       Pred-Pos | Pred-Neg | Total
# Pos     26829      2697     29526
# Neg   137469         0    137469
# Total   164298      2697    166995
# 90.9%
# 16.3%

# '''
# efn = causeFN(conf, f)
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




    
