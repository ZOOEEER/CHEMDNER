# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:34:48 2020

@author: hanyl
"""

import os
import collections
from Sentence import Sentence, Entity

Abstract = collections.namedtuple('Abstract', ['PMID', 'title', 'abstract'])
Annotation = collections.namedtuple('Annotation', ['PMID', 'type_text', 
                                                   'start', 'end', 'mention', 'type_cem'])

class Data: # CHEMDNER
    def __init__(self, path = os.path.join('D:\\Desktop\\Python\\CHEMDNER','chemdner_corpus'), keys = ['train'], system = None):
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
        self.data = {}
        self.dict = {}
        self.dict['T'] = {}
        self.dict['A'] = {}
        self.readData(keys, system) # 默认读入train ['train', 'dev', 'eva']
    
    def _preData(self, line, key2):
        # readline后的预处理，删除末尾空格，将双引号替换为单引号以防止POS问题。
        data = tuple(line.replace('\n','').replace('"','\'').split('\t'))
        if key2 == 'abstracts':
            return Abstract(*data)
        elif key2 == 'annotations':
            return Annotation(*data)
    
    def readData(self, keys, system):
        keys = set(keys) - set(self.data.keys()) # 除去已经加载的信息。
        for key1 in keys:
            self.data[key1] = {}
            for key2 in ['abstracts', 'annotations']:
                self.data[key1][key2] = []
                with open(self.files[key1][key2], 'r', encoding ='utf8') as f:
                    line = f.readline()
                    while line:
                        self.data[key1][key2].append(self._preData(line, key2))
                        line = f.readline()
        # 将data封装进字典，self.dict['T']['train']['PMID'] -> Sentence
        # Text 信息
        for key in keys:
            self.dict['T'][key] = { abstract.PMID:Sentence(abstract.title) for abstract in self.data[key]['abstracts'] }
            self.dict['A'][key] = { abstract.PMID:Sentence(abstract.abstract) for abstract in self.data[key]['abstracts'] }
        # Annotations 信息
        for key in keys:
            for annotation in self.data[key]['annotations']:
                self.dict[annotation.type_text][key][annotation.PMID].entities.append(
                    Entity(mention = annotation.mention, start = annotation.start,end = annotation.end,type_cem = annotation.type_cem)
                )
        # POS
        if system:
            self.posData(keys, system)
                    
    def posData(self, keys, system):
        for key1 in self.dict.keys():
            for key2 in keys:
                for _, sentence in self.dict[key1][key2].items():
                    sentence.pos(system)
    
    def getDataDict(self, key = 'train', type_text = 'T'):
        return self.dict[type_text][key]
    
    def getSentences(self, key = 'train'):
        return list(self.getDataDict(key, 'T').values()) + list(self.getDataDict(key, 'A').values())