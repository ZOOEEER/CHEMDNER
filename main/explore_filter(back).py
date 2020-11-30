# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:25:58 2020

@author: hanyl
"""

# import time


from Sentence import EmptyEntities
from Data import Data
# from FilterDev import FilterDev
from System import System, Filter
from Model import RNN, loadModel
from TrainingData import lineToTensor
# from NERer import evaluate

'=================配置===============' '最佳实践{？}' 
'3-50 优化的比例'
'POS仍然严格'
'refiner得到很多'
'max_ratio_common 控制比例'
'min_ratio_common 开白名单'
# For Filter
filter_name = 'train'
f = Filter.SettingFile(filter_name)
f.sifter.min_length = 3
f.sifter.max_length = 50
f.sifter.min_ratio_common = -0.1
# f.sifter.max_ratio_common = 1.01
# # For NERer
# ner_name = 'rnn-dev-train-100000'
# model = loadModel(ner_name)

# For Data
s = System()
d = Data(keys = ['dev'], system = s)
# d.readData(['train'], s)

dev_sens = list(d.getDataDict('dev', 'T').values()) + list(d.getDataDict('dev', 'A').values())
dev_sens_no_entity = EmptyEntities(dev_sens)
# # For eva data
# d.readData(['eva'], s)
# eva_sens = list(d.getDataDict('eva', 'T').values()) + list(d.getDataDict('eva', 'A').values())
# eva_sens_no_entity = EmptyEntities(eva_sens)

# f.filt(dev_sens_no_entity)
# conf1 = evaluation_sens(dev_sens , dev_sens_no_entity)
# evaluation_stat(conf1)

#       Positive | Negative | Total
# True     25337      4189     29526
# False    84764         0     84764
# Total   110101      4189    114290
# 85.8%

# confusion_tn, stat_tn = analyze_tn_auto(conf1)
# stat_pure(confusion_tn)
# stat_enum(confusion_tn)




'=======函数定义-模型使用========'

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

# '=======函数定义-模型评估========'

# def evaluation_sen(sen_target, sen_pred):
#     confusion = {}
#     se_target = set( (entity.start, entity.end) for entity in sen_target.entities)
#     se_pred = set( (entity.start, entity.end) for entity in sen_pred.entities)
#     confusion['tp'] = [ (entity, sen_target)
#              for entity in sen_target.entities
#                  if (entity.start, entity.end) in se_pred
#         ]
#     confusion['tn'] = [ (entity, sen_target)
#              for entity in sen_target.entities
#                  if not (entity.start, entity.end) in se_pred
#         ]
#     confusion['fp'] = [ (entity, sen_target)
#              for entity in sen_pred.entities
#                  if not (entity.start, entity.end) in se_target
#         ]
#     confusion['fn'] = []
#     return confusion

# def evaluation_sens(sens_target, sens_pred):
#     confusion = {'tp':[],
#                  'tn':[],
#                  'fp':[],
#                  'fn':[]}
#     for sen_target, sen_pred in zip(sens_target, sens_pred):
#         sen_conf = evaluation_sen(sen_target, sen_pred)
#         for key in confusion.keys():
#             confusion[key].extend(sen_conf[key])
#     return confusion

# def evaluation_stat(confusion):
#     tps = len(confusion['tp'])
#     tns = len(confusion['tn'])
#     fps = len(confusion['fp'])
#     fns = len(confusion['fn'])
#     print(' '*6 + 'Positive | Negative | Total')
#     print('True  {:8}  {:8}  {:8}'.format(tps, tns, tps+tns))    
#     print('False {:8}  {:8}  {:8}'.format(fps, fns, fps+fns))
#     print('Total {:8}  {:8}  {:8}'.format(tps+fps, tns+fns, tps+tns+fps+fns))



# from Sentence import posGather
# from FilterDev import EntityInD
# from collections import namedtuple
# c_tn = namedtuple('entityTN', ['Entity', 'Sentence', 'pos', 'cA', 'Ac', 'FalsePOS', 'len4', 'len27', 'InDict', 'Other'])
# def analyze_tn_auto(confusion):
#     count_truth = len(confusion['tp']) + len(confusion['tn'])
#     c = []
#     for item in confusion['tn']:
#         entity = item[0]
#         sentence = item[1]
#         # -A
#         b3 = (entity.start - 1 >= 0 and sentence.text[entity.start-1] != ' ')
#         b4 = (entity.end + 1 < len(sentence.text) and sentence.text[entity.end] not in ' .,')
#         _, pos = posGather(sentence.POS_char[entity.start:entity.end]) 
#         b5 = pos not in f.sifter.pos_list
#         b6 = len(entity.mention) < f.sifter.min_length
#         b6_2 = len(entity.mention) >= f.sifter.max_length
#         b7 = EntityInD(entity.mention, pos) in f.dictLookuper.dict.keys() and \
#                 f.dictLookuper.dict[EntityInD(entity.mention, pos)] == 'COMMON'
#         b8 = not any([b3, b4, b5, b6, b7])
#         c.append(c_tn(entity, sentence, pos, b3, b4, b5, b6, b6_2, b7, b8))
#     records = [(c_tn._fields[i], sum([item[i] for item in c]), sum([item[i] for item in c])/count_truth)
#                    for i in range(3,len(c[0]))]
#     for record in records:    
#         print('{} {} {}'.format(*record))
#     return c


# def stat_pure(confusion_tn):
#     print('cA', sum([ 1 for c_tn_he in confusion_tn if c_tn_he.cA and not c_tn_he.FalsePOS \
#                and not c_tn_he.len4 and not c_tn_he.len27 and not c_tn_he.InDict]))
#     print('Ac', sum([ 1 for c_tn_he in confusion_tn if c_tn_he.Ac and not c_tn_he.FalsePOS \
#                and not c_tn_he.len4 and not c_tn_he.len27 and not c_tn_he.InDict]))
#     print('FalsePOS', sum([ 1 for c_tn_he in confusion_tn if c_tn_he.FalsePOS and not c_tn_he.cA \
#                and not c_tn_he.Ac and not c_tn_he.len4  and not c_tn_he.len27 and not c_tn_he.InDict]))
#     print('len27', sum([ 1 for c_tn_he in confusion_tn if c_tn_he.len27 and not c_tn_he.FalsePOS \
#                and not c_tn_he.InDict and not c_tn_he.cA and not c_tn_he.Ac]))
#     print('len4', sum([ 1 for c_tn_he in confusion_tn if c_tn_he.len4 and not c_tn_he.FalsePOS \
#                and not c_tn_he.InDict and not c_tn_he.cA and not c_tn_he.Ac]))    
#     print('Indict',sum([ 1 for c_tn_he in confusion_tn if c_tn_he.InDict and not c_tn_he.FalsePOS \
#                and not c_tn_he.len4 and not c_tn_he.len27 and not c_tn_he.cA and not c_tn_he.Ac]))

# def stat_enum(confusion_tn, key = 'Ac'):
#     a = 0
#     for i, c_tn_he in enumerate(confusion_tn[a:200]):
#         entity = c_tn_he.Entity
#         sentence = c_tn_he.Sentence
#         if key == 'FalsePOS' and c_tn_he.FalsePOS and not c_tn_he.cA and not c_tn_he.Ac \
#                     and not c_tn_he.len4  and not c_tn_he.len27 and not c_tn_he.InDict:
#             sli = slice(max(entity.start - 5, 0), min(entity.end + 5, len(sentence.text)))
#             print(i)
#             print(entity)
#             print(c_tn_he.pos)
#             print('     {}     '.format(entity.mention))
#             print(sentence.text[sli])
#             print(sentence.POS_char[sli])
        
#         if key == 'InDict' and c_tn_he.InDict and not c_tn_he.FalsePOS \
#                     and not c_tn_he.len4 and not c_tn_he.len27 and not c_tn_he.cA and not c_tn_he.Ac:
#             print(EntityInD(entity.mention, c_tn_he.pos),
#                   f.dictLookuper.dict[EntityInD(entity.mention, c_tn_he.pos)])
#         # Ac
#         if key == 'Ac' and c_tn_he.Ac and not c_tn_he.FalsePOS and not c_tn_he.len4 \
#             and not c_tn_he.len27 and not c_tn_he.InDict: # 394
#         # - / 后面跟一堆东西，新的Chem或者是indu, base, simu 等
#         # 粘太多东西，分不开了...(1. 修改数据集/2.把粘的东西用规则打开/3.char-level的对齐预测。)
#             sli = slice(max(entity.start - 5, 0), min(entity.end + 20, len(sentence.text)))
#             print(i)
#             print(entity)
#             print('     {}     '.format(entity.mention))
#             print(sentence.text[sli])
#             print(sentence.POS_char[sli])
#             for index in range(entity.end, len(sentence.text)):
#                 if sentence.text[index] == ' ':
#                     break
#             substr = sentence.text[entity.end:index]
#             print('len:{}'.format(index - entity.start))
#             print('验证：{}'.format(substr in f.refiner.lists['Ac'] or substr in f.refiner.lists['endswith']))
#         # # cA
#         if key == 'cA' and c_tn_he.cA and not c_tn_he.FalsePOS and not c_tn_he.len4 \
#             and not c_tn_he.len27 and not c_tn_he.InDict: # 394
#         # 粘太多东西，分不开了...(1. 修改数据集/2.把粘的东西用规则打开/3.char-level的对齐预测。)    
#             sli = slice(max(entity.start - 5, 0), min(entity.end + 5, len(sentence.text)))
#             print(i)
#             print('     {}     '.format(entity.mention))
#             print(sentence.text[sli])
#             print(sentence.POS_char[sli])

    

'=========探索=========='
# # f.sifter.max_distinct_word = 4
# a = 1
# a+= 4500
# f.sifter._filt(EmptyEntities([confusion_tn[a].Sentence])[0])

# from Sentence import Sentence
# sen_example = Sentence('Di 2-ethyl hexyl phthalate affects differentiation and matrix mineralization of rat calvarial osteoblasts--in vitro.')
# sen_example.pos(s)
# f.sifter._filt(sen_example)
# f.sifter.filt([sen_example])
# f.dictLookuper.lookup([sen_example])
# f.sifter.min_length = 1
# f.sifter.ratio_common = 0.2
# f.filt([sen_example])
# sen_example.entities


# sen_example = EmptyEntities([confusion_tn[a].Sentence])[0]
# f.filt([sen_example])
# conf_example = evaluation_sen(confusion_tn[a].Sentence, sen_example)
# for key in conf_example.keys():
#     print(key)
#     for items in conf_example[key]:
#         print(items[0])
# # example_sentence.entities
# predict_mention(model, '')
# # for item in confusion_tn.items():
# #     print(item[0], len(item[1]))
    

# from flashtext import KeywordProcessor
# kp = KeywordProcessor()
# kp.add_keyword('AMP')

# example = 'Phillyrin attenuates high glucose-induced lipid accumulation in human HepG2 hepatocytes through the activation of LKB1/AMP-activated protein kinase-dependent signalling.'
# kp.extract_keywords(example)
# kp.add_keyword('O(3)')
# example = 'one (O(3)) and'
# kp.extract_keywords(example)

'============AccA开发=============='
# from copy import deepcopy
# backoff_counters = deepcopy(accas.counters)

# train_sens = list(d.getDataDict('train', 'T').values()) + list(d.getDataDict('train', 'A').values())
# from FilterDev import AccAStater
# accas = AccAStater()
# accas.counters = deepcopy(backoff_counters)

# accas.lists = {}
# for key in accas.counters.keys():
#     accas.lists[key] = [] 
# accas.genlist(POSITIVE_COVER_RATIO = 0.9, VERBOSE = 2)

# accas.lists['startswith']

# for key in accas.counters.keys():
#     print(len(accas.counters[key]))
#     print(sum(accas.counters[key].values()))
#     for i, item in enumerate(accas.counters[key].most_common(400)):
#         if item[1] >= 3:
#             print(item)

# for i, item in enumerate(accas.counters['Ac'].most_common(400)):
#     print('{} {:<20} {:3}/{:3}'.format(i, *item, accas.counters['endswith'][item[0]]))
# for i, item in enumerate(accas.counters['cA'].most_common(400)):
#     print('{} {:>20} {:3}/{:3}'.format(i, *item, accas.counters['startswith'][item[0]]))

# accas.counters['endswith']
# accas.counters['cA']
# accas.counters['startswith']


# # No entity
# dev_sens_no_entitym
# # After Filt
# dev_sens_no_entitymm

# dev_sens_no_entity6 = deepcopy(dev_sens_no_entitym)
# f.sifter.ratio_common = -0.1
# f.filt(dev_sens_no_entity6)


# dev_sens_no_entity6 = deepcopy(dev_sens_no_entitymm)
# from System import Refiner
# refiner = Refiner(accas.lists)
# refiner.refine(dev_sens_no_entity6)
# conf16 = evaluation_sens(dev_sens , dev_sens_no_entity6)
# evaluation_stat(conf16)
# # refiner
# #       Positive | Negative | Total
# # True     25009      4517     29526
# # False    86841         0     86841
# # Total   111850      4517    116367
# # 84.70%
# confusion_tn6, stat_tn = analyze_tn_auto(conf16)
# stat_pure(confusion_tn6)
# stat_enum(confusion_tn6, 'Ac')
# # cA 421
# # Ac 584
# # FalsePOS 1425
# # len27 308
# # len4 79
# # Indict 265

# from Sentence import Sentence
# sen_example = Sentence('Phillyrin attenuates high glucose-induced lipid accumulation in human HepG2 hepatocytes through the activation of LKB1/AMP-activated protein kinase-dependent signalling.')
# sen_example.pos(s)
# f.filt([sen_example])
# sen_example.entities
# refiner.refine([sen_example])
# sen_example.entities

# key = '-peptide'
# key = '-contaminated'
# key = '/Si/Cr'
# accas.counters['Ac'][key]
# key in refiner.lists['Ac']

# '====AccA的保存===='
# from FilterDev import writeFile
# writeFile(d, l, t, refiner.lists, 'train')
# from FilterDev import readFile
# d, l, t, rl = readFile('train')
# # rl == refiner.lists

'============Filter参数变化=============='
# '======min_length======'
# # dev_sens_no_entity2 = EmptyEntities(dev_sens)
# # f.sifter.min_length = 4
# # f.filt(dev_sens_no_entity2)
# # conf12 = evaluation_sens(dev_sens , dev_sens_no_entity2)
# # evaluation_stat(conf12)
# # #       Positive | Negative | Total
# # # True     24209      5317     29526
# # # False    80869         0     80869
# # # Total   105078      5317    110395
# '====平均0.166(3) -> 81.99%的检出率 ！！！===='
# # #       Positive | Negative | Total
# # # True     24316      5210     29526
# # # False    84593         0     84593
# # # Total   108909      5210    114119
# # # 平均0.119的回报率(2) -> 82.3%的检出率 ！！！
# # #       Positive | Negative | Total
# # # True     24405      5121     29526
# # # False    94724         0     94724
# # # Total   119129      5121    124250
# # # 平均0.0657的回报(1) -> 82.6% 的检出率
# # #       Positive | Negative | Total
# # # True     24810      4716     29526
# # # False   489779         0    489779
# # # Total   514589      4716    519305
# # # 平均0.00356的回报(0)

# # confusion_tn, stat_tn = analyze_tn_auto(conf12)
# # # f.sifter.min_length = 3
# # # [('cA', 965, 0.03268305899884847),
# # #   ('Ac', 1908, 0.06462101198943304),
# # #   ('FalsePOS', 2531, 0.08572105940526993),
# # #   ('len4', 1096, 0.037119826593510805),
# # #   ('len27', 1140, 0.03861003861003861),
# # #   ('InDict', 417, 0.014123145702093071),
# # #   ('Other', 424, 0.014360224886540675)]
# # stat_pure(confusion_tn)
# # # cA 375
# # # Ac 829
# # # len27 308
# # # len4 82
# '======max_length======'
# # dev_sens_no_entity3 = EmptyEntities(dev_sens)
# # f.sifter.max_length = 27
# # f.filt(dev_sens_no_entity3)
# # conf13 = evaluation_sens(dev_sens , dev_sens_no_entity3)
# # evaluation_stat(conf13)
# # #       Positive | Negative | Total
# # # True     23253      6273     29526
# # # False    75697         0     75697
# # # Total    98950      6273    105223
# # # 平均0.1167的回报(35) -> 检出率提高0.01
# # #       Positive | Negative | Total
# # # True     23337      6189     29526
# # # False    76050         0     76050
# # # Total    99387      6189    105576
# # # 平均0.1353的回报(50) -> 检出率提高0.01

'======min_ratio_common======'
# dev_sens_no_entitym = EmptyEntities(dev_sens)

# dev_sens_no_entity4 = EmptyEntities(dev_sens_no_entitym)
# f = Filter(*readFile(filter_name))
# f.sifter.min_length = 3
# f.sifter.ratio_common = 0.2
# f.filt(dev_sens_no_entity4)
# conf14 = evaluation_sens(dev_sens , dev_sens_no_entity4)
# evaluation_stat(conf14)
# # f.sifter.ratio_common = 0.1
# #       Positive | Negative | Total
# # True     24727      4799     29526
# # False   139964         0    139964
# # Total   164691      4799    169490
# # 83.74% 0.0087的回报率 （没有正筛选了.../边界情况？）
# # f.sifter.ratio_common = 0.15
# #       Positive | Negative | Total
# # True     24832      4694     29526
# # False   148101         0    148101
# # Total   172933      4694    177627
# # f.sifter.ratio_common = 0.2
# #       Positive | Negative | Total
# # True     24916      4610     29526
# # False   152145         0    152145
# # Total   177061      4610    181671
# # 84.38% 0.00992的回报率
# # f.sifter.ratio_common = 0.5
# #       Positive | Negative | Total
# # True     25088      4438     29526
# # False   498367         0    498367
# # Total   523455      4438    527893
# # 84.97%
# # f.sifter.ratio_common = 0.5
# #       Positive | Negative | Total
# # True     25088      4438     29526
# # False   498367         0    498367
# # Total   523455      4438    527893
# # 84.97%
# confusion_tn, stat_tn = analyze_tn_auto(conf14)
# stat_pure(confusion_tn)
# stat_enum(confusion_tn)
# # cA 528
# # Ac 1171 
# # FalsePOS 768
# # len27 308
# # len4 79
# # Indict 265

'======distinct word======'
# dev_sens_no_entity5 = EmptyEntities(dev_sens_no_entitym)
# f = Filter(*readFile(filter_name))
# f.sifter.min_length = 3
# f.sifter.ratio_common = 0.2
# f.sifter.max_distinct_word = 4
# f.filt(dev_sens_no_entity5)
# conf15 = evaluation_sens(dev_sens , dev_sens_no_entity5)
# evaluation_stat(conf15)

# confusion_tn5, stat_tn5 = analyze_tn_auto(conf15)
# stat_pure(confusion_tn5)
# stat_enum(confusion_tn5)
# # f.sifter.max_distinct_word = 6
# #       Positive | Negative | Total
# # True     25170      4356     29526
# # False   171644         0    171644
# # Total   196814      4356    201170

# # f.sifter.max_distinct_word = 10
# #       Positive | Negative | Total
# # True     25228      4298     29526
# # False   182165         0    182165
# # Total   207393      4298    211691
# # 85.44%



'============Evaluation==============' 'train数据Filter,dev - NERer, dev数据测试'
# # dev_sens_no_entity = EmptyEntities(dev_sens)
# f.sifter.ratio_common = -0.1
# f.filt(dev_sens_no_entity)
# conf1 = evaluation_sens(dev_sens , dev_sens_no_entity)
# evaluation_stat(conf1)
# # train-dev dev
# # f.sifter.min_length = 3
# #       Positive | Negative | Total
# # True     24209      5317     29526
# # False    80869         0     80869
# # Total   105078      5317    110395

# # train-dev dev
# # f.sifter.min_length = 4
#       Positive | Negative | Total
# True     23024      6502     29526
# False    73736         0     73736
# Total    96760      6502    103262
# # 75%的检出率；
# # predict(model, dev_sens_no_entity)
# # conf2 = evaluation_sens(dev_sens , dev_sens_no_entity)
# # evaluation_stat(conf2)
# # # train-dev dev
# # #       Positive | Negative | Total
# # # True     19457     10069     29526
# # # False    13104         0     13104
# # # Total    32561     10069     42630
# # # 66%的检出率；

# confusion_tn, stat_tn = analyze_tn_auto(conf1)
# # sum([s[1] for s in stat_tn ])
# # confusion['tn']的统计
# # cA 1259 0.042
# # Ac 2216 0.0755
# # FalsePOS 2531 0.0857
# # len4 2187 0.1126
# # len27 1140 0.0386
# # InDict 417 0.014
# # Other 188 0.0091
# # Total 10744 
# stat_pure(confusion_tn)
# # cA 394
# # Ac 850 粘连的问题解决一下 -> +3%的检出率。
# # FalsePOS 1419 # Big Problem... 换一些POS Statter的策略。
# # len27 308 -> 1% 检出率
# # len4 861 -> 在4->3之后有有效地降低... -> +2-5%的检出率。
# # Indict 144 小问题... 数据标注的问题比较多...


# # # confusion_tn, stat_tn = analyze_tn_auto(conf12)
# # # f.sifter.min_length = 3
# # # [('cA', 965, 0.03268305899884847),
# # #  ('Ac', 1908, 0.06462101198943304),
# # #  ('FalsePOS', 2531, 0.08572105940526993),
# # #  ('len4', 1096, 0.037119826593510805),
# # #  ('len27', 1140, 0.03861003861003861),
# # #  ('InDict', 417, 0.014123145702093071),
# # #  ('Other', 424, 0.014360224886540675)]
# # # stat_pure(confusion_tn)
# # # cA 375
# # # Ac 829
# # # len27 308
# # # len4 82


# # a = 4001
# # slice_here = slice(a, a+1)
# # evaluate_check_by_human(dev_sens[slice_here] , dev_sens_no_entity[slice_here])
# # # evaluate_check_by_human(dev_sens[slice_here] , dev_sens_no_pred[slice_here])
# # a += 1
# # '============Evaluation=============='   'train数据Filter,dev - NERer, eva数据测试'
# # f.filt(eva_sens_no_entity)
# # evaluate_auto(eva_sens , eva_sens_no_entity)
# # # train-dev eva
# # #       Positive | Negative | Total
# # # True     19745      5606     25351
# # # False    60575         0     60575
# # # Total    80320      5606     85926
# # predict(eva_sens_no_entity)
# # evaluate_auto(eva_sens , eva_sens_no_entity)
# # # train-dev eva
# # #       Positive | Negative | Total
# # # True     16800      8551     25351
# # # False    10096         0     10096
# # # Total    26896      8551     35447
# # # 与 dev类似。
    

    
