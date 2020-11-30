# -*- coding: utf-8 -*-
"""
2020/11/05 15:55
explore data of CHEMDNER
整理一个data class出来
"""

import collections
import os
import matplotlib.pyplot as plt
import re
import nltk
from flashtext import KeywordProcessor

Abstract = collections.namedtuple('Abstract', ['PMID', 'title', 'abstract'])
Annotation = collections.namedtuple('Annotation', ['PMID', 'type_text', 
                                                   'start', 'end', 'mention', 'type_cem'])
class Data:
    def __init__(self, path = '.\\chemdner_corpus'):
        self.path = path
        self.files = {
            'train':{'abstracts':os.path.join(self.path, 'training.abstracts.txt'),
                     'annotations':os.path.join(self.path, 'training.annotations.txt')},
            'silver':{'abstracts':os.path.join(self.path, 'silver.abstracts.txt'),
                      'annotations':os.path.join(self.path, 'silver.predictions.txt')},
            'dev':{'abstracts':os.path.join(self.path, 'development.abstracts.txt'),
                   'annotations':os.path.join(self.path, 'development.annotations.txt')},
            'eva':{'abstracts':os.path.join(self.path, 'evaluation.abstracts.txt'),
                   'annotations':os.path.join(self.path, 'evaluation.annotations.txt')},
            'test':{'cdi':os.path.join(self.path, 'cdi_ann_test_13-09-13.txt'),
                    'cem':os.path.join(self.path, 'cem_ann_test_13-09-13.txt')},
            'aux':{'chemical':os.path.join(self.path, 'chemdner_chemical_disciplines.txt'),
                   'corpus':os.path.join(self.path, 'chemdner_abs_test_pmid_label.txt')}          
            }
        self.readData()
    
    def _preData(self, line, key2):
        if key2 == 'abstracts':
            return Abstract(*tuple(line.replace('\n','').split('\t')))
        elif key2 == 'annotations':
            return Annotation(*tuple(line.replace('\n','').split('\t')))
    
    def readData(self):
        self.data = {}
        for key1 in ['train', 'dev', 'eva']:
            self.data[key1] = {}
            for key2 in ['abstracts', 'annotations']:
                self.data[key1][key2] = []
                with open(self.files[key1][key2], 'r', encoding ='utf8') as f:
                    line = f.readline()
                    while line:
                        self.data[key1][key2].append(self._preData(line, key2))
                        line = f.readline()
    
    # def stat(self):
        # for key1 in self.data.keys():
        #     for key2 in self.data[key1].keys():
        #         print('data.{}.{}.length: {}'.format(key1, key2, len(self.data[key1][key2])))
        

if __name__ == '__main__':
    d = Data()

def selfTest(d):
    '''
    使用train, dev, eval生成的字典，交叉对数据集进行探索。
    以字典的匹配结果作为base_baseline
    '''
    kps = {} # keyword processors
    ans_pred = {} # Annotations_pred_by_dict_method key1-pred, key2-vocab(kp)
    for key1 in ['train']:
        kps[key1] = KeywordProcessor(case_sensitive = True)
        for annotation in d.data[key1]['annotations']:
            kps[key1].add_keyword(annotation.mention, (annotation.type_cem, annotation.mention))
    for keya in ['train']:
        ans_pred[keya] = {}
        for keyb in ['train']:
            ans_pred[keya][keyb] = []
            for corpus in d.data[keya]['abstracts']:
                annotations_pre = kps[keyb].extract_keywords(corpus.title, span_info=True)
                # [(('Monument', 'Taj Mahal'), 0, 9), (('Location', 'Delhi'), 16, 21)]
                if len(annotations_pre):
                    for a in annotations_pre:
                        ans_pred[keya][keyb].append(
                            Annotation(corpus.PMID, 'T', str(a[1]), str(a[2]), a[0][1], a[0][0]))
                annotations_pre = kps[keyb].extract_keywords(corpus.abstract, span_info=True)
                if len(annotations_pre):
                    for a in annotations_pre:
                        ans_pred[keya][keyb].append(
                            Annotation(corpus.PMID, 'A', str(a[1]), str(a[2]), a[0][1], a[0][0]))
    # len(ans_pred['train']['train']) # 40208
    # len(d.data['train']['annotations']) # 29478
    # 探究一下True Negtive的原因
    rm_tt = _evalAA(d.data['train']['annotations'], ans_pred['train']['train'])
    text_tn = exploreA(d.data['train']['abstracts'], rm_tt['tn'])
    len(text_tn)
    for i in text_tn:
        print('\t'.join(i))
    # 数据集本身的问题：子项的标注(多义性)问题/variance
    bt = []
    for i in text_tn:
        if len(i[-3]) and len(i[-1]):
            bt.append(re.match('[0-9a-zA-Z]', i[-3][-1]) is not None or 
                      re.match('[0-9A-z]', i[-1][0]) is not None)
    (len(bt) - sum(bt) + 3)/29478
    
    text_fp = exploreA(d.data['train']['abstracts'], rm_tt['fp'])
    len(text_fp)
    counter_keylength_fp = collections.Counter([len(i[-2]) for i in text_fp])
    x = list(counter_keylength_fp.elements())
    n_bins = max(counter_keylength_fp.keys()) - 1 # 35
    fig, ax = plt.subplots()
    ax.set_title('fp_Num')
    ax.hist(x, n_bins)
    len([i for i in x if i > 3]) # 965
    
    with open('text_fp.txt', 'w', encoding='utf8') as f:
        for i in text_fp:
            if len(i[-2]) > 3:
                f.write('\t'.join(i)+'\n')
    
    
    
    
    
    
    abstracts = d.data['train']['abstracts']
    dict_abs = { c.PMID:c.abstract for c in abstracts} # dict of abstract
    dict_title = { c.PMID:c.title for c in abstracts} # dict of title

    query = 'a nonlinear and nona'
    # query = '[xB2O3 + (1 - x)P2O5]'
    
    text = dict_abs['23261590']
    kp_temp = KeywordProcessor(case_sensitive = True)
    # kp_temp.add_keyword('aa')
    # kp_temp.add_keyword('aa bb')
    # kp_temp.add_keyword('bbaa')
    # kp_temp.add_keyword('bb')    
    # kp_temp.extract_keywords('aa bb aa bb bb aa bc', span_info = True)
    
    kp_temp.extract_keywords(text, span_info = True)
    re.search(query, text)
    kps['train'].extract_keywords(text, span_info = True)
    kps['train'].extract_keywords(dict_abs['23300000'], span_info = True)
    


# d.data['train']['annotations'][0]


# Annotation = collections.namedtuple('Annotation', ['PMID', 'type_text', 
#                                                    'start', 'end', 'mention', 'type_cem'])

def evalAA(annotations_target, annotations_pred):
    '''
    给定两个annotations, 得到一个evaluation matrix, 给出TP, TN, FP, FN; Accuracy, Recall; F1
    进阶：分类别统计：type_text? type_cem?
    调用_evalAA
    '''
    # for 
    
    
def _evalAA(annotations_target, annotations_pred):
    '''
    evalAA的核心实现，在evalAA中对数据进行分类记项讨论。
    '''
    set_target = set(annotations_target)
    set_pred = set(annotations_pred)
    refuse_matrix = {'tp':set_target & set_pred,
                     'tn':set_target - set_pred,
                     'fp':set_pred - set_target,
                     'fn':set()}
    tp = len(refuse_matrix['tp'])
    tn = len(refuse_matrix['tn'])
    fp = len(refuse_matrix['fp'])
    fn = len(refuse_matrix['fn'])
    print(' '*6 + 'Positive | Negative | Total')
    print('True  {:8}  {:8}  {:8}'.format(tp, tn, tp+tn))    
    print('False {:8}  {:8}  {:8}'.format(fp, fn, fp+fn))
    print('Total {:8}  {:8}  {:8}'.format(tp+fp, tn+fn, tp+tn+fp+fn))
    return refuse_matrix
    
def exploreA(abstracts, annotations):
    '''
    根据annotations，返回对应的摘要text及前后文信息，以\t分割，便于人类检查。
    '''
    dict_abs = { c.PMID:c.abstract for c in abstracts} # dict of abstract
    dict_title = { c.PMID:c.title for c in abstracts} # dict of title
    text_return = []
    for a in annotations:
        if a.type_text == 'A':
            text = dict_abs[a.PMID]
        elif a.type_text == 'T':
            text = dict_title[a.PMID]    
        text_return.append((a.PMID, a.type_text, a.start, a.end,
                            text[max(0, int(a.start)-30):int(a.start)],
                            text[int(a.start):int(a.end)],
                            text[int(a.end):min(len(text), int(a.end)+30)]
                        ))
    return text_return

def stat(d):
    '''
    数据探索，得到一些统计上的信息：key的长度分布，Counter(key)的分布。
    非常恶劣的长尾分布。

    '''
    print('==length==')
    for key1 in d.data.keys():
        for key2 in d.data[key1].keys():
            print('data.{}.{}.length: {}'.format(key1, key2, len(d.data[key1][key2])))

    print('==correlation==')
    cc = {} # for chemical_counter
    for key1 in d.data.keys():
        for key2 in ['annotations']:
            cc[key1] = collections.Counter([a.mention for a in d.data[key1][key2]])
    print(' '*12 + '|{:<12}|{:<12}|{:<12}'.format(*cc.keys()))
    for keya in cc.keys():
        print('{:<12}|{:<12}|{:<12}|{:<12}'.format(keya,
               *tuple(len(cc[keya].keys() & cc[keyb].keys()) for keyb in cc.keys())))

    print('==name length distribution==')
    # clc = {} # for chemical_length_count
    for key1 in cc.keys():
        x1 = [len(mention) for mention in cc[key1].keys()]
        x2 = [len(a.mention) for a in d.data[key1]['annotations']]
        n_bins = max(x1)
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].set_title(key1 + 'Key Length')
        axs[0].hist(x1, bins=n_bins)
        axs[1].hist(x2, bins=n_bins)
    
    print('==num distribution==')
    for key1 in cc.keys():
        x = [cc[key1][mention] for mention in cc[key1].keys()]
        n_bins = max(x)
        fig, ax = plt.subplots()
        ax.set_title(key1 + 'Num')
        ax.hist(x, n_bins)
    
    print('==feature==')
    dataset = 'train'
    print('for dataset = {}'.format(dataset))
    ct = cc[dataset]
    
    print('more than 50 times:',
        { key:ct[key] for key in ct.keys() if ct[key] > 50 })
    print('the number of these: ',
          len({ key:ct[key] for key in ct.keys() if ct[key] > 50 }))
    print('the mention of these: ',
        sum(ct[key] for key in ct.keys() if ct[key] > 50))
    
    print('less than 2 times:',
        collections.Counter( len(key) for key in ct.keys() if ct[key] < 2 ))
    print('the number of these: ',
          len({ key:ct[key] for key in ct.keys() if ct[key] < 2 }))
    print('the mention of these: ',
        sum([ct[key] for key in ct.keys() if ct[key] < 2]))
    
    print('Length of key longer than 50:',
        { (len(key),ct[key]) for key in ct.keys() if len(key) > 50 })
    print('the number of these: ',
          len({ key:ct[key] for key in ct.keys() if len(key) > 50 }))
    print('the mention of these: ',
        sum(ct[key] for key in ct.keys() if len(key) > 50))
    
    print('Length of key shorter than 5:',
        { (key,ct[key]) for key in ct.keys() if len(key) < 5 })
    print('the number of these: ',
          len({ key:ct[key] for key in ct.keys() if len(key) < 5 }))
    print('the mention of these: ',
        sum(ct[key] for key in ct.keys() if len(key) < 5))
    
    print('Length of key shorter than 10:')
    print('the number of these: ',
          len({ key:ct[key] for key in ct.keys() if len(key) < 10 }))
    print('the mention of these: ',
        sum(ct[key] for key in ct.keys() if len(key) < 10))
    
    
    
    # list(counter.elements())

    # kp_temp.add_keyword('aa')
    # kp_temp.add_keyword('aa bb')
    # kp_temp.add_keyword('bbaa')
    # kp_temp.add_keyword('bb')    
    # kp_temp.extract_keywords('aa bb aa bb bb aa bc', span_info = True)

# with open(os.path.join(d.path, 'training.abstracts.txt'), 'r', encoding ='utf8') as f:
#     line = f.readline()       
#     i = 1
#     while line:
#         print(line)
#         print(i)
#         i -= 1
#         if i < 0:
#             break
#         # self.data[key1][key2].append(self._preData(line, key2))
#         line = f.readline()

# [('Monument', 'Taj Mahal'), ('Location', 'Delhi')]
# NOTE: replace_keywords feature won't work with this.