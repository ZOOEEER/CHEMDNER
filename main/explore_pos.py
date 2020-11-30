# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:25:58 2020

@author: hanyl
"""

# import time

from Sentence import EmptyEntities
from Data import Data
from System import System, Filter
from Evaluation import evaluation_sens, causeFN, stat_enum
from FilterDev import FilterDev



"==For Data=="
# POS 的方法是一个全局设定 嗷
s = System()
d = Data(keys = ['train', 'dev'], system = s)
train = d.getSentences('train')
dev = d.getSentences('dev')
backoff_0 = None
pred = EmptyEntities(dev) if not backoff_0 else EmptyEntities(backoff_0)
backoff_0 = EmptyEntities(pred)


# # For eva data
# d.readData(['eva'], s)
# eva = list(d.getDataDict('eva', 'T').values()) + list(d.getDataDict('eva', 'A').values())
# eva_pred = EmptyEntities(eva)

"==For FilterDev=="
name = 'train-POS5_Star-20201122'
fd = FilterDev(name = name)
# fd.paras['COVERAGE'] = 0.95
# fd.paras['VERBOSE'] = 3
# fd._posstat_star(train)
fd.tune(train) # 改名字
# dict是最耗时间的嗷...
fd.writeToFile()

"==For Filter=="
filter_name = name
f = Filter.SettingFile(filter_name)

# from System import readFile
# filter_name = 'train-POS2'
# f = Filter(*readFile(filter_name))


'====Evaluation===='
f.filt(pred)
conf = evaluation_sens(dev, pred)

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
'''
# With Refiner/POSGatherStar
        cA  664 2.2%
        Ac  994 3.4%
  FalsePOS 1035 3.5%
      len4  308 1.0%
     len27  320 1.1%
    InDict  409 1.4%
     Other  191 0.6%
Stat of Pure Causes for FN:
        cA  255
        Ac  408
  FalsePOS  608
      len4   56
     len27    0
    InDict  265
     Other  121
     Total 1713

'''
stat_enum(entitiesFN = efn, myfilter = f, num_samples = 20, key = 'len4')

'----POS----'
posp =  ('JJ', ' ', 'CD')
print(fd.posStater.counter_negative[posp], fd.posStater.counter_positive[posp])



posp = ('()', '', 'NNP')
posp = ('[]', '', 'NNP')
posp = ('{}', '', 'NNP')

'====COMMON===='



'===草稿==='


    
