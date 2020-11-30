# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:33:19 2020

@author: hanyl
"""
import copy

class Sentence:
    def __init__(self, text):
        self.text = text
        self.POS_char = [] # ['NNP', 'NNP', 'NNP', ...]
        self._POS_word = [] # list of tuple [('Mercury', 'NNP'), ('induces', 'VBZ'), ]
        # self.POS_word = [] # list of tuple [('Mercury', 'NNP', 0, 7), ]
        self.entities = [] # list of Entity
        self.relations = [] # list of Relation
        
    def pos(self, system):
        '''
        调用System中的Tokenizer 和 POStagger 对句子进行POS化，若Entities中有实体，则扩充其pos标签。
        '''
        self._POS_word = system.POStagger(system.Tokernizer(self.text))
        self._posChar()
        self._posEntity()
        
    def _posChar(self):
        assert len(self._POS_word) > 0
        pos = 0
        i_token = 0
        while pos < len(self.text):
            if self.text[pos] == self._POS_word[i_token][0][0]:
                len_token = len(self._POS_word[i_token][0])
                for _ in range(len_token):
                    self.POS_char.append(self._POS_word[i_token][1])
                i_token += 1
                pos += len_token
            else:
                self.POS_char.append(' ')
                pos += 1
    
    def _posEntity(self):
        assert len(self.POS_char) == len(self.text)
        for entity in self.entities:
            _, pos  = posGather(self.POS_char[entity.start:entity.end], bl_poly = False)
            entity.pos = pos 
            
    def __repr__(self):
        return "Sentence(text = '{}')".format(self.text)
    
# 定义到Sentence定义里面?
def EmptyEntities(sentences):
    ss = copy.deepcopy(sentences)
    for s in ss:
        s.entities = []
    return ss

class Entity:
    def __init__(self, mention, pos = None, start = -1, end = -1, type_cem = None):
        self.mention = mention
        self.pos = pos if pos else []
        self.start = int(start)
        self.end = int(end)
        self.type_cem = type_cem
    def __repr__(self):
        return "Entity(mention = '{}',start = {},end = {},type_cem = '{}')".format(
            self.mention, self.start, self.end, self.type_cem)
        
class Relation:
    def __init__(self, relation, e1, e2):
        self.relation = relation
        self.e1 = e1
        self.e2 = e2

def posGather(POS_char, bl_poly = True):
    '''
    Parameters
    ----------
    POS_char : TYPE
        逐字符的POS标记.
    bl_poly : TYPE, optional
        是否将pospath基于规则再次聚合?

    Returns
    -------
    posgather : list of (pos, len_of_pos)
    pospath : tuple of pos
    
    '''
    posgather = []
    pos = None
    n_pos = 0
    for i in range(len(POS_char)):
        if POS_char[i] == pos:
            n_pos += 1
        else:
            if pos:
                posgather.append((pos, n_pos))
            pos = POS_char[i]
            n_pos = 1
    posgather.append((pos, n_pos))
    if bl_poly:
        posgather = _posGatherStar(posgather)
    pospath = tuple([i[0] for i in posgather])
    return posgather, pospath

def _posGatherStar(posgather):
    '''
    长单词聚合。

    '''
    # 分词
    word_boundaries = ' '
    sentence_boundaries = ',.'
    splits = []
    bl_in_word = False
    for i, pg in enumerate(posgather):
        pos = pg[0]
        len_pos = pg[1]
        if pos in word_boundaries or \
                (pos in sentence_boundaries and \
                 ( i+1 == len(posgather) or posgather[i+1][0] in word_boundaries )):
            splits.append(i)
            bl_in_word = False
        elif not bl_in_word:
            splits.append(i)
            bl_in_word = True
    splits.append(len(posgather))
    # 确定POS标签
    # left_brackets = '([{'
    # right_brackets = ')]}'
    # brackets = left_brackets + right_brackets
    # deleted_symbols = brackets
    deleted_symbols = [] # 不删除任何symbols
    posgatherStar = []
    for start, end in zip(splits[:-1], splits[1:]):
        # 处理括号等删除符号
        if end - start == 1: 
            pos, len_pos = posgather[start]
        else:
            ds_indexes = []
            for pos_i in range(start, end):
                if posgather[pos_i][0] in deleted_symbols:
                    ds_indexes.append(pos_i)
            if len(ds_indexes) == 0: # <=2的旁路
                if end - start <= 2:
                    posgatherStar.extend(posgather[start:end])
                    continue
            if end - start - len(ds_indexes) <= 1:
                assert len([ p for p, _ in posgather[start:end] if p not in deleted_symbols ]) <= 1, \
                    '{'+'start:{}, end:{}, ds_indexes:{}, posgather:{}'.format(start, end, ds_indexes, posgather)+'}'
                if end - start - len(ds_indexes) == 0:
                    pos = 'DEL'
                else:
                    pos = [ p for p, _ in posgather[start:end] if p not in deleted_symbols ][0]
            else:
                pos = str(end - start - len(ds_indexes))
            len_pos = sum([ len_pos for _, len_pos in posgather[start:end] ])
        posgatherStar.append((pos, len_pos))
    return posgatherStar

def posStarLength(pos):
    length = 0
    for p in pos:
        if all([c in '0123456789' for c in p]):
            length += int(p)
        else:
            length += 1
    return length

