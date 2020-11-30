# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:25:58 2020

@author: hanyl
"""

# import time

from Sentence import EmptyEntities
from Data import Data
from System import System, Filter
# from Model import RNN, loadModel
# from TrainingData import lineToTensor
from Evaluation import evaluation_sen, evaluation_sens, causeTN, stat_enum

'=================配置===============' '最佳实践{？}' 
'3-50 优化的比例'
'POS仍然严格'
'refiner得到很多'
'max_ratio_common 控制比例'
'min_ratio_common 开白名单'

"==For Data=="
s = System()
d = Data(keys = ['dev'], system = s)

dev = list(d.getDataDict('dev', 'T').values()) + list(d.getDataDict('dev', 'A').values()) # 做成Data的接口
backoff_0 = None
pred = EmptyEntities(dev) if not backoff_0 else EmptyEntities(backoff_0)
backoff_0 = EmptyEntities(pred)


# # For eva data
# d.readData(['eva'], s)
# eva = list(d.getDataDict('eva', 'T').values()) + list(d.getDataDict('eva', 'A').values())
# eva_pred = EmptyEntities(eva)

"==For FilterDev=="
from FilterDev import FilterDev
key = 'train'
d.readData([key], s)
train = d.getSentences(key)



fd = FilterDev(name = 'train-POS2')
fd.paras['COVERAGE'] = 0.95
# fd.paras['VERBOSE'] = 3
fd.tune(train) # 改名字

from FilterDev import POSStater, priorityItemInPOSS2
import math
def priorityItemInPOSS3(item):
    return 1 + 5 * math.log10(item.positive_count) - \
        0.5 * item.negative_count / item.positive_count - item.pos_length


def priorityItemInPOSSH(item):
    ratio =  max(0, item.negative_count / item.positive_count - 10)
    length = max(0, item.pos_length - 4)
    return - math.log10(1+ratio) - length


# fd._size()
# fd._accastat(train)
fd.posStater = POSStater(priorityItemInPOSS = priorityItemInPOSSH)
fd._posstat()

fd.writeToFile()



"==For Filter=="
# filter_name = 'train'
filter_name = 'train-POS2'
f = Filter.SettingFile(filter_name)
# f.sifter.min_length = 3
# f.sifter.min_common_tolerated = 3



'====Evaluation===='
f.filt(pred)
conf = evaluation_sens(dev, pred)

'''
      Positive | Negative | Total
True     26381      3145     29526
False   134222         0    134222
Total   160603      3145    163748
89.3%
16.4%
'''

etn = causeTN(conf, f)
stat_enum(entitiesTN = etn, myfilter = f, num_samples = 20, key = 'Ac')
'''
        cA  664 2.2%
        Ac  994 3.4%
  FalsePOS 1035 3.5%
      len4  308 1.0%
     len27  320 1.1%
    InDict  409 1.4%
     Other  191 0.6%
Stat of Pure Causes for TN:
        cA  255
        Ac  408
  FalsePOS  608
      len4   56
     len27    0
    InDict  265
     Other  121
     Total 1713
'''


'----POS----'
posp =  ('3', ' ', 'NNP')
fd.posStater.counter_negative[posp]
fd.posStater.counter_positive[posp]


posp = ('()', '', 'NNP')
posp = ('[]', '', 'NNP')
posp = ('{}', '', 'NNP')

'====COMMON===='
'''
如何搞COMMON方法？通过chem 和 COMMON 的区别来定位单词?
'''




'=======函数定义-模型使用========'
# # For NERer
# ner_name = 'rnn-dev-train-100000'
# model = loadModel(ner_name)

# def predict_tensor(model, line_tensor):
#     hidden = model.initHidden()
    
#     for i in range(line_tensor.size()[0]):
#         output, hidden = model(line_tensor[i], hidden)
#     return output

# categories = ['Chem', 'COMMON']
# def predict_mention(model, mention):
#     tensor = lineToTensor(mention)
#     output = predict_tensor(model, tensor)
#     index_category = output.topk(1).indices.item()
#     category = categories[index_category]
#     return category

# def predict(model, sentences):
#     for sen in sentences:
#         for entity in sen.entities[::-1]: # 不知道其实现原理，反着删比较好{。}
#             if entity.type_cem == '':
#                 type_cem = predict_mention(model, entity.mention)
#                 if type_cem != 'COMMON':
#                     entity.type_cem = type_cem
#                 else:
#                     sen.entities.remove(entity)




'============Evaluation==============' 'train数据Filter,dev - NERer, dev数据测试'

    
