# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:06:19 2020

@author: hanyl
"""
import nltk
from flashtext import KeywordProcessor
from FilterDev import EntityInD, readFile
from Sentence import Entity, posGather, posStarLength


class System:
    def __init__(self):
        self.Tokernizer = nltk.word_tokenize
        self.POStagger = nltk.pos_tag
        # self.FilterDev = FilterDev()
        # self.DictLookuper = DictLookuper()
        # self.Filter = Filter()
        # self.NERer = None
        # self.REer = None
    
    def __call__(self, text):
        pass
    
class Filter:
    '''
    封装 DictLookuper 和 Sifter
    '''
    def __init__(self, dicts, \
                        list_pos, 
                        threshold, 
                        lists_acca, # 补丁
                        list_pos_star, # 补丁
                        ):
        self.dictLookuper = DictLookuper(case_sensitive = True)
        for d in dicts.values():
            self.dictLookuper.add_keywords_dict(d)
        self.sifter = Sifter(list_pos = list_pos, list_pos_star = list_pos_star, threshold = threshold)
        self.refiner = Refiner(lists_acca)
        
        self.bl_sifter = True
        self.bl_refiner = True
    
    def filt(self, sentences):
        self.dictLookuper.lookup(sentences)
        if self.bl_sifter:
            self.sifter.filt(sentences)
        if self.bl_refiner:
            self.refiner.refine(sentences)

    @classmethod
    def readFromFile(cls, path):
        return cls(*readFile(path))
    

class DictLookuper:
    def __init__(self, dict_record = None, case_sensitive = True):
        self.dict = {} # (mention, pos):type_cem
        self._kp = KeywordProcessor(case_sensitive = case_sensitive)
        self.add_keywords_dict(dict_record)
    
    def add_keywords_dict(self, dict_record):
        if dict_record:
            self.dict.update(dict_record)
            for entity, type_cem in dict_record.items():
                self._kp.add_keyword(entity.mention, (type_cem, entity.mention))
    
    def word(self, word):
        records = []
        for mention, pos in self.dict.keys():
            if mention == word:
                records.append((mention, pos, self.dict[(mention,pos)]))
        return records
    
    def lookup(self, sentences):
        for sentence in sentences:
            entities = self._lookup(sentence)
            sentence.entities.extend(entities)
    
    def _lookup(self, sentence):
        entities = self._extract_keywords(sentence.text, sentence.POS_char)
        return entities
    
    def _extract_keywords(self, text, POS_char):
        keywords = self._kp.extract_keywords(text, span_info = True)
        # [(('Monument', 'Taj Mahal'), 0, 9), (('Location', 'Delhi'), 16, 21)]
        entities = []
        for keyword in keywords:
            start = keyword[1]
            end = keyword[2]
            mention = keyword[0][1]
            _, pos = posGather(POS_char[start:end], bl_poly = False)
            type_cem = keyword[0][0]
            entity = EntityInD(keyword[0][1], pos)
            if entity in self.dict.keys() and self.dict[entity] == type_cem:
                entities.append(Entity(mention, pos, start, end, type_cem))
        return entities

class Sifter:
    def __init__(self, list_pos, list_pos_star, threshold):
        
        self.list_pos = list_pos
        self.max_distinct_word = max([len(pos) for pos in list_pos])
        self._list_pos_star = list_pos_star # 保存副本，在filt时再删除其中内容。
        self.min_length, self.max_length = threshold
        self.min_ratio_common = -0.1 # 调节common比例，小于该比例的进白名单。
        self.max_ratio_common = 1.1
        # self.min_common_tolerated = self.min_length
        self.min_common_tolerated = 3
        
        self.bl_be_space = True
        self.bl_long_word = True
        self.bl_threshold = True
        self.bl_len_common = True
        self.bl_list_pos = True
        self.bl_list_pos_star = len(self._list_pos_star) > 0
        
        
        
    def _filt(self, sentence):
        entities = []
        se_annotations = set()
        POS_char = sentence.POS_char
        text = sentence.text
        # Common 单词 和 空格 [True]
        common_char = [False] * len(POS_char)
        for entity in sentence.entities:
            if entity.type_cem == 'COMMON':
                for i in range(entity.start, entity.end):
                    common_char[i] = True
            else:
                entities.append(entity) # 把非平凡的加入到entities中，最终保留。
                se_annotations.add((entity.start, entity.end))
        for i in range(len(text)):
            if text[i] == ' ':
                common_char[i] = True
        '==bl_poly==False'
        pG, posPath = posGather(POS_char, bl_poly = False)
        indexes_space = set([ i for i in range(len(pG)) if pG[i][0] == ' '])
        for max_distinct_word in range(0, self.max_distinct_word):
            for i in range(len(pG) - max_distinct_word):
                # 空格始末判断
                if self.bl_be_space and i in indexes_space or i + max_distinct_word in indexes_space:
                    continue
                start = sum([ pG[w][1] for w in range(i)]) if i > 0 else 0
                end = start + sum([ pG[w][1] for w in range(i, i + max_distinct_word + 1)])
                mention_length = end - start
                # 长度判断
                if self.bl_threshold and \
                    (mention_length > self.max_length or mention_length < self.min_length):
                    continue
                
                mention_slice = slice(start, end)
                pos = posPath[i:i + max_distinct_word + 1]
                len_common_char_length = sum(common_char[mention_slice])
                # Common判断
                if len_common_char_length / mention_length > self.min_ratio_common: # 如果不是Uncommon的。
                    # Common长度判断
                    if self.bl_len_common and mention_length - len_common_char_length < self.min_common_tolerated:
                        continue
                    if len_common_char_length / mention_length > self.max_ratio_common: # 如果common成分过多
                        continue
                    # POS判断
                    if self.bl_list_pos and not pos in self.list_pos:
                        continue
                else:
                    pass
                    # POS判断
                # 重复判断
                if any([ start >= s and end <= e for s,e in se_annotations]):
                        continue
                mention = text[mention_slice]
                entities.append(Entity(mention, pos, start, end, type_cem = ''))
        '==bl_poly==补丁'
        if self.bl_list_pos_star:
            self.list_pos_star = set( pos for pos in self._list_pos_star if posStarLength(pos) > self.max_distinct_word)
            for p in [ ('6'), ('5'), ('4'), ]: # 加一个数字的
                self.list_pos_star.add(p) 
            pG, posPath = posGather(POS_char, bl_poly = True)
            indexes_space = set([ i for i in range(len(pG)) if pG[i][0] == ' '])
            for max_distinct_word in range(0, self.max_distinct_word):
                for i in range(len(pG) - max_distinct_word):
                    # 空格始末判断
                    if self.bl_be_space and i in indexes_space or i + max_distinct_word in indexes_space:
                        continue
                    pos = posPath[i:i + max_distinct_word + 1]
                    # POS长度判断
                    if posStarLength(pos) <= self.max_distinct_word:
                        continue
                    # POS判断
                    if not pos in self.list_pos_star:
                        continue
                    start = sum([ pG[w][1] for w in range(i)]) if i > 0 else 0
                    end = start + sum([ pG[w][1] for w in range(i, i + max_distinct_word + 1)])
                    # 重复判断
                    if any([ start >= s and end <= e for s,e in se_annotations]):
                            continue
                    mention = text[mention_slice]
                    entities.append(Entity(mention, pos, start, end, type_cem = ''))
        return entities
        
    def filt(self, sentences):
        for sentence in sentences:
            sentence.entities = self._filt(sentence)
        
class Refiner:
    def __init__(self, lists_acca):
        self.lists = lists_acca
        self.brackets = {')':'(',
                         ']':'[',
                         '}':'{'
                         }
        
    def _refine(self, sentence):
        se_new = set()
        se_cem = set()
        entities_new = []
        for entity in sentence.entities:
            if entity.type_cem not in ['','COMMON']: #检查是否有type_cem, 跳过
                entities_new.append(entity)
                se_cem.add((entity.start, entity.end))
                continue
            bl_cA = False
            bl_Ac = False
            starts = [entity.start]
            ends = [entity.end]
            for substr in self.lists['startswith']:
                if entity.mention.startswith(substr):
                    starts.append(entity.start + len(substr))
            for substr in self.lists['endswith']:
                if entity.mention.endswith(substr):
                    ends.append(entity.end - len(substr))
            for substr in self.lists['cA']: # 如果匹配，那么将清除原来的entity
                if entity.mention.startswith(substr):
                    starts.append(entity.start + len(substr))
                    bl_cA = True        
            for substr in self.lists['Ac']: # 如果匹配，那么将清除原来的entity
                if entity.mention.endswith(substr):
                    ends.append(entity.end - len(substr))
                    bl_Ac = True
            for start in starts:
                for end in ends:
                    if start < end:
                        se_new.add((start, end))
            if bl_cA and bl_Ac:
                se_new.discard((entity.start, entity.end))
                
                 
        for start, end in se_new:
                if (start, end) in se_cem:
                    continue
                new_slice = slice(start, end)
                mention = sentence.text[new_slice]
                if not self.well(mention):
                    continue
                _, pos = posGather(sentence.POS_char[new_slice], bl_poly = False)
                entities_new.append(Entity(mention, pos, start, end, type_cem = ''))
        return entities_new
    
    def well(self, mention):
        return all([self._brackets_balanced(mention),])
    
    def _brackets_balanced(self, mention):
        bracket_list = []
        for c in mention:
            if c in self.brackets.keys():
                if len(bracket_list) == 0:
                    return False
                else:
                    if bracket_list[-1] != self.brackets[c]:
                        return False
                    else:
                        bracket_list.pop()
            elif c in self.brackets.values():
                bracket_list.append(c)
        if len(bracket_list) == 0:
            return True
        else:
            return False
    
    def refine(self, sentences):
        for sentence in sentences:
            sentence.entities = self._refine(sentence)
    
    

