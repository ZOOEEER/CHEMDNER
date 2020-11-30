# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:25:58 2020

@author: hanyl
"""
from collections import namedtuple
from random import sample

from Sentence import posGather
from FilterDev import EntityInD

__all__ = [
    "evaluation_sen",
    "evaluation_sens",
    "causeFN",
    "stat_enum"
]

'====evaluation===='
def evaluation_sen(sen_target, sen_pred):
    confusion = {}
    se_target = set( (entity.start, entity.end) for entity in sen_target.entities)
    se_pred = set( (entity.start, entity.end) for entity in sen_pred.entities)
    confusion['tp'] = [ (entity, sen_target)
             for entity in sen_target.entities
                 if (entity.start, entity.end) in se_pred
        ]
    confusion['fn'] = [ (entity, sen_pred)
             for entity in sen_target.entities
                 if not (entity.start, entity.end) in se_pred
        ]
    confusion['fp'] = [ (entity, sen_target, sen_pred)
             for entity in sen_pred.entities
                 if not (entity.start, entity.end) in se_target
        ]
    confusion['tn'] = []
    return confusion

def evaluation_sens(sens_target, sens_pred, bl_stat = True):
    confusion = {'tp':[],
                 'fn':[],
                 'fp':[],
                 'tn':[]}
    for sen_target, sen_pred in zip(sens_target, sens_pred):
        sen_conf = evaluation_sen(sen_target, sen_pred)
        for key in confusion.keys():
            confusion[key].extend(sen_conf[key])
    if bl_stat:
        evaluation_stat(confusion)
    return confusion

'=====分析====='
def evaluation_stat(confusion):
    tps = len(confusion['tp'])
    fns = len(confusion['fn'])
    fps = len(confusion['fp'])
    tns = len(confusion['tn'])
    print(' '*6 + 'Pred-Pos | Pred-Neg | Total')
    print('Pos  {:8}  {:8}  {:8}'.format(tps, fns, tps+fns))    
    print('Neg {:8}  {:8}  {:8}'.format(fps, tns, fps+tns))
    print('Total {:8}  {:8}  {:8}'.format(tps+fps, fns+tns, tps+fns+fps+tns))
    print('{:.1%}'.format(tps/(tps+fns)))
    print('{:.1%}'.format(tps/(tps+fps)))

# def overlapFN(confusion):
#     ofn = {'ppAA':[],
#            'pApA':[],
#            'pAAp':[],
#            'AppA':[],
#            'ApAp':[],
#            'pAAp':[]}
#     overlaps, keys = _overlap()
#     for key in keys:
#         ofn[keys]
#     return ofn



entityFN = namedtuple('entityFN', ['Entity', 'Sentence', 'pos', 'cA', 'Ac', 'FalsePOS', 'len4', 'len27', 'InDict', 'Other'])
def causeFN(confusion, myfilter, bl_pure = True):
    count_truth = len(confusion['tp']) + len(confusion['fn'])
    entitiesFN = []
    for item in confusion['fn']:
        entity = item[0]
        sentence = item[1]
        _, pos = posGather(sentence.POS_char[entity.start:entity.end]) 
        # -A
        b3 = (entity.start - 1 >= 0 and sentence.text[entity.start-1] != ' ')
        b4 = (entity.end + 1 < len(sentence.text) and sentence.text[entity.end] not in ' .,')
        b5 = pos not in myfilter.sifter.list_pos
        b6 = len(entity.mention) < myfilter.sifter.min_length
        b6_2 = len(entity.mention) >= myfilter.sifter.max_length
        b7 = EntityInD(entity.mention, pos) in myfilter.dictLookuper.dict.keys() and \
                myfilter.dictLookuper.dict[EntityInD(entity.mention, pos)] == 'COMMON'
        b8 = not any([b3, b4, b5, b6, b7])
        entitiesFN.append(entityFN(entity, sentence, pos, b3, b4, b5, b6, b6_2, b7, b8))
    records = []
    for i in range(3,len(entitiesFN[0])):
        records.append((entityFN._fields[i], sum([item[i] for item in entitiesFN]), sum([item[i] for item in entitiesFN])/count_truth))
    for record in records:    
        print('{:>10} {:4} {:.1%}'.format(*record))
    if bl_pure:
        print('Stat of Pure Causes for FN:')
        stat_pureFN(entitiesFN)
    return entitiesFN

def stat_pureFN(entitiesFN):
    msg ='{:>10} {:4}'
    total = 0
    for key in entityFN._fields[3:]:
        sum_key = sum([ entityfn._asdict()[key] for entityfn in entitiesFN if sum(entityfn[3:]) == 1])
        print(msg.format(key,sum_key))
        total += sum_key
    print(msg.format('Total', total))

def stat_enum(entitiesFN, myfilter, num_samples = 20, key = 'Ac'):
    records = []
    for index, entityfn in enumerate(entitiesFN):
        if entityfn._asdict()[key] and sum(entityfn[3:]) == 1:
            records.append((index, entityfn))
    for index, entityfn in sample(records, min(len(records), num_samples)):
        entity = entityfn.Entity
        sentence = entityfn.Sentence
        space_counts = 5
        sli = slice(max(entity.start - space_counts, 0), min(entity.end + space_counts, len(sentence.text)))
        print(index)
        print(entity)
        if key in ['FalsePOS']:
            print(entityfn.pos)
            print(sentence.POS_char[sli])
        if key in ['Ac','cA']:
            print((' '*min(space_counts, entity.start)+'{}').format(entity.mention))
            print(sentence.text[sli])
        if key in ['InDict']:
            print(EntityInD(entity.mention, entityfn.pos),
                      myfilter.dictLookuper.dict[EntityInD(entity.mention, entityfn.pos)])
        if key in ['len4', 'len27']:
            print('len:{}'.format(entity.end - entity.start))
