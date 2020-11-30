# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:37:26 2020

@author: hanyl
"""
import os
import json
import math
from collections import namedtuple, deque, Counter
import heapq
from Sentence import posGather
EntityInD = namedtuple('EntityInD', ['mention', 'pos'])
ItemInH = namedtuple('ItemInH',['mention', 'pos', 'positive_count', 'negative_count'])
ItemInPOSS = namedtuple('ItemInPOSS',['pos', 'pos_length', 'positive_count', 'negative_count'])
def _path(filename):
    return os.path.join('D:\Desktop\Python\CHEMDNER\dicts', '{}.json'.format(filename))


class FilterDev:
    def __init__(self, name = 'temp'):
        self.name = name
        
        self.entitySpliter = EntitySpliter(bl_onlyAnnotation=False)
        self.dictRecorder = DictRecorder()
        self.sizingSystem = SizingSystem()
        self.posStater = POSStater()
        # AccA stater
        self.accaStater = AccAStater()
        # EntitySpliter_star & posStater_star
        self.entitySpliter_star = EntitySpliter(bl_onlyAnnotation=True)
        self.posStater_star = POSStater()
        
        self.paras = {
            # general
                'COVERAGE': 0.95,
                'VERBOSE': 2,
            # Entity Spliter
                'MAX_DISTINCT_WORD': 3,
                'MIN_LENGTH': 0,
                'MAX_LENGTH': 10000,
            # Dict Recorder
                'POSITIVE_COUNT': 5,
                'RATIO_POS2NEG_DR': 16,
            # Sizing System
                'MIN_THRESHOLD': 3,
                'MARGINAL_BENEFIT': 0.001,
            # POS Stater
                # 'PRIORITY_STRATEGY': 2,
                'POS_LENGTH': 4, 
                'RATIO_POS2NEG_PS': 0.1,
                'MAX_POS_LENGTH': 4,
                'RATIO_POS2NEG_PS': 0.1,
            # AccA Stater
                'MIN_COUNT': 2,
                'RATIO_POS2NEG_AS': 5,
                'COVERAGE_AS': 0.8,

        }
        
        self.counters = None
        self.counter_annotation = Counter()
        self.counter_common = Counter()
        self.type_cems = []
    
    def __getattr__(self, name):
        cls = type(self)
        if name in self.paras.keys():
             return self.paras[name]
        msg = '{.__name__!r} object has no attribute {!r}'
        raise AttributeError(msg.format(cls, name))
    
    def setpara(self, name, value):
        if not name in self.__dict__.keys():    
            self.paras[name] = value
        else:
            print('Naming conflict.')
    
    def tune(self, sentences):
        self._devsplit(sentences)
        self._dictrecord()
        self._size()
        self._posstat()
        self._accastat(sentences)
        self._posstat_star(sentences)

    # DevSplit
    def _devsplit(self, sentences):
        self.entitySpliter.DevSplit(sentences, 
                MAX_DISTINCT_WORD = self.MAX_DISTINCT_WORD, 
                MIN_LENGTH = self.MIN_LENGTH, 
                MAX_LENGTH = self.MAX_LENGTH,
                BL_POLY = False
                )
        self.counters = self.entitySpliter.counters
        self.type_cems = [ key for key in self.counters.keys() if key != 'COMMON']
        for key in self.type_cems:
            self.counter_annotation += self.counters[key]
        self.counter_common = self.counters['COMMON']
    # Dict record, Sizing, POS Stat
    def _dictrecord(self):
        self.dictRecorder.record(self.counters,
                POSITIVE_COUNT = self.POSITIVE_COUNT, 
                RATIO_POS2NEG = self.RATIO_POS2NEG_DR, 
                POSITIVE_COVER_RATIO = self.COVERAGE, 
                VERBOSE = self.VERBOSE
                )
    def _size(self):
        self.sizingSystem.size(self.counter_annotation,
                RATIO = self.COVERAGE, 
                MIN_THRESHOLD = self.MIN_THRESHOLD, 
                MARGINAL_BENEFIT = self.MARGINAL_BENEFIT, 
                VERBOSE = self.VERBOSE
                )
    def _posstat(self):
        self.posStater.stat(self.counter_annotation, self.counter_common, 
                POS_LENGTH = self.POS_LENGTH, 
                RATIO_POS2NEG = self.RATIO_POS2NEG_PS,
                COVER_RATIO = self.COVERAGE, 
                VERBOSE = self.VERBOSE
                )
    # AccA stat
    def _accastat(self, sentences):
        self.accaStater.stat(sentences,
                RATIO_POS2NEG = self.RATIO_POS2NEG_AS,
                MIN_COUNT = self.MIN_COUNT,
                POSITIVE_COVER_RATIO = self.COVERAGE_AS,
                VERBOSE = self.VERBOSE
                )
    
    # DevSplit and POS Stat
    def _posstat_star(self, sentences):
        self.entitySpliter_star.DevSplit(sentences, 
                MAX_DISTINCT_WORD = self.MAX_DISTINCT_WORD, 
                MIN_LENGTH = self.MIN_LENGTH, 
                MAX_LENGTH = self.MAX_LENGTH,
                BL_POLY = True
                )
        es_star_counters = self.entitySpliter_star.counters
        type_cems = [ key for key in es_star_counters.keys() if key != 'COMMON']
        counter_annotation_star = Counter()
        for key in type_cems:
            counter_annotation_star += es_star_counters[key]
        self.posStater_star.stat(counter_annotation_star, Counter(), 
                POS_LENGTH = self.POS_LENGTH, 
                RATIO_POS2NEG = self.RATIO_POS2NEG_PS,
                COVER_RATIO = self.COVERAGE, 
                VERBOSE = self.VERBOSE
                )
    
    def getFilterSettings(self):
        return self.dictRecorder.dicts, \
                 self.posStater.list_pos, \
                   self.sizingSystem.threshold, \
                       self.accaStater.lists, \
                           self.posStater_star.list_pos

    def writeToFile(self):
        return writeFile(*self.getFilterSettings(), self.name)

class AccAStater:
    def __init__(self):        
        self.boundary_char = ' '
        
        self.counters = {}
        self.counters['Ac'] = Counter()
        self.counters['cA'] = Counter()
        self.counters['endswith'] = Counter()
        self.counters['startswith'] = Counter()
        
        self.lists = {}
        for key in self.counters.keys():
            self.lists[key] = [] 
        # self.list['Ac'] = [] # del
        # self.list['endswith'] = [] # keep
        # self.list['cA'] = [] # del
        # self.list['startswith'] = [] # keep
        
        self.verbose = [0, 2, 10, 300]
        
    def stat(self, sentences,
             RATIO_POS2NEG = 5, MIN_COUNT = 2, POSITIVE_COVER_RATIO = 0.8, 
             VERBOSE = 2
             ):
        for sentence in sentences:
            self._stat(sentence)
        self.endswith_stat(sentences)
        self.genlist(RATIO_POS2NEG, MIN_COUNT, POSITIVE_COVER_RATIO, VERBOSE)
        
    def _stat(self, sentence):
        for entity in sentence.entities:
            if not entity.type_cem in ['', 'COMMON']:
                text = sentence.text
                str_Ac = []
                str_cA = []
                for i in range(entity.end, len(text)):
                    if text[i] not in self.boundary_char:
                        str_Ac.append(text[i])
                    else:
                        break
                for i in range(entity.start - 1, 0, -1):
                    if text[i] not in self.boundary_char:
                        str_cA.append(text[i])
                    else:
                        break
                if len(str_Ac):
                    self.counters['Ac'].update([''.join(str_Ac)])
                if len(str_cA):
                    self.counters['cA'].update([''.join(str_cA[::-1])])
    
    def endswith_stat(self, sentences):
        counter_list = []
        for sentence in sentences:
            for entity in sentence.entities:
                for c in self.counters['Ac'].keys():
                    if entity.mention.endswith(c):
                        counter_list.append(c)
        self.counters['endswith'].update(counter_list)
        counter_list = []
        for sentence in sentences:
            for entity in sentence.entities:
                for c in self.counters['cA'].keys():
                    if entity.mention.startswith(c):
                        counter_list.append(c)
        self.counters['startswith'].update(counter_list)
    
    def genlist(self, RATIO_POS2NEG, MIN_COUNT, POSITIVE_COVER_RATIO, VERBOSE):
        self._genlist('Ac', 'endswith', RATIO_POS2NEG, MIN_COUNT, POSITIVE_COVER_RATIO, VERBOSE)
        self._genlist('cA', 'startswith', RATIO_POS2NEG, MIN_COUNT, POSITIVE_COVER_RATIO, VERBOSE)
        
    def _genlist(self, key_out, key_in, RATIO_POS2NEG, MIN_COUNT, POSITIVE_COVER_RATIO, VERBOSE):
        records = deque(maxlen = self.verbose[min(VERBOSE,len(self.verbose))])
        count_before = MIN_COUNT
        positive_cover_count = 0
        positive_cover_ratio = 0
        total_positive_count = sum(self.counters[key_out].values())
        for item in self.counters[key_out].most_common(): # 优先级是出现的次数。
            count_key_out = item[1]
            substr = item[0]
            key = ''
            if positive_cover_ratio <= POSITIVE_COVER_RATIO or count_key_out == count_before or count_key_out >= MIN_COUNT:
            # 相当贪心的做法...
                if self.counters[key_in][substr] / count_key_out <= 1 / RATIO_POS2NEG: # Positive
                    key = key_out
                elif self.counters[key_in][substr] / count_key_out > RATIO_POS2NEG: # Negative
                    pass
                else:
                    if count_key_out >= MIN_COUNT and self.counters[key_in][substr] >= MIN_COUNT: # Balanced
                        key = key_in

                if len(key):
                    self.lists[key].append(substr)
                    positive_cover_count += count_key_out
                    positive_cover_ratio = positive_cover_count / total_positive_count
                    records.append((substr, key, positive_cover_ratio, count_key_out, positive_cover_count, total_positive_count))
                else:
                    records.append((substr, key, positive_cover_ratio, 0, positive_cover_count, total_positive_count))                    
            else:
                break
            count_before = count_key_out
        for item in records:
            print("{:20} -> lists['{:8}'], {:.1%} = +{:2} -> {:4}/{:4}".format(*item))    
        if VERBOSE > 0:
            for key in [key_in, key_out]:
                print("Total in lists['{}']: length = {}".format(key, len(self.lists[key])))

    
    
class EntitySpliter:
    def __init__(self, bl_onlyAnnotation = True):
    # Control
        self.bl_onlyAnnotation = bl_onlyAnnotation
        #{(Word, pos, TypeCem): Frequency}
    #Results
        self.counters = {}
        self.counters['COMMON'] = Counter()
    
    def getCounter(self):
        return self.counters
    
    def _split(self, sentence, MAX_DISTINCT_WORD, MIN_LENGTH, MAX_LENGTH, BL_POLY):
        for entity in sentence.entities:
            if not entity.type_cem in self.counters.keys():
                self.counters[entity.type_cem] = Counter()
            _, pos = posGather(entity.pos, BL_POLY)
            self.counters[entity.type_cem][EntityInD(entity.mention, pos)] += 1
        if not self.bl_onlyAnnotation:
            self.counters['COMMON'] += Counter(self._splitCommon(sentence, MAX_DISTINCT_WORD, MIN_LENGTH, MAX_LENGTH))
            
    def _splitCommon(self, sentence, MAX_DISTINCT_WORD, MIN_LENGTH, MAX_LENGTH):
        '''
        对于句子进行分词，得到COMMON的Entities,然后送去计数统计。
        '''
        entities = []
        POS_char = sentence.POS_char
        text = sentence.text
        se = [(entity.start, entity.end) for entity in sentence.entities]
        pG, posPath = posGather(POS_char, bl_poly = False)
        indexes_space = set([ i for i in range(len(pG)) if pG[i][0] == ' '])
        for max_distinct_word in range(0, MAX_DISTINCT_WORD):
            for i in range(len(pG) - max_distinct_word):
                # 空格
                if i in indexes_space or i + max_distinct_word in indexes_space:
                    continue
                start = sum([ pG[w][1] for w in range(i)]) if i > 0 else 0
                end = start + sum([ pG[w][1] for w in range(i, i + max_distinct_word + 1)])
                # 包含
                if not all([ end <= s or start >= e for s, e in se]):
                    continue
                # 长度
                if end - start > MAX_LENGTH or end - start < MIN_LENGTH:
                    continue
                pos = posPath[i:i + max_distinct_word + 1]
                mention_slice = slice(start, end)
                mention = text[mention_slice]
                # print(start, end, mention, pos)
                # type_cem = 'COMMON'
                entities.append(EntityInD(mention, pos))
        return entities
    
    def DevSplit(self, sentences, MAX_DISTINCT_WORD = 3, MIN_LENGTH = 0, MAX_LENGTH = 10000, BL_POLY = False):
        '''
        开发Filter时，对Sentences进行分词，得到未标注集和标注集的(Word, pos, TypeCem, Frequency)组。
        '''
        for sentence in sentences:
            self._split(sentence, MAX_DISTINCT_WORD, MIN_LENGTH, MAX_LENGTH, BL_POLY)


def priorityRatio(ratio):
    return 1 - ratio
def priorityItemInH(item):
    return priorityRatio(item.negative_count / item.positive_count)

class PQ:
    def __init__(self, priorityfc):
        self._queue = []
        self.priorityfc = priorityfc
    
    def push(self, item):
        # 传入两个参数，一个是存放元素的数组，另一个是要存储的元素，这里是一个元组。
        # 由于heap内部默认有小到大排，所以对priority取负数
        priority = self.priorityfc(item)
        heapq.heappush(self._queue, (-priority, item))
  
    def pop(self):
        return heapq.heappop(self._queue)[-1]

class DictRecorder:
    def __init__(self):
        self.dicts = {}   
        
        self.verbose = [0, 2, 10, 1000]
    
    def record(self, counters,
               POSITIVE_COUNT = 5, RATIO_POS2NEG = 16, POSITIVE_COVER_RATIO = 0.9, 
               VERBOSE = 2):
        # 阳性
        type_cems = counters.keys()
        for cem in type_cems:
            if cem != 'COMMON':
                self.dicts[cem] = {}
                self._record(counters, cem, self.ALL,
                  POSITIVE_COUNT, RATIO_POS2NEG, POSITIVE_COVER_RATIO, VERBOSE)    
        # COMMON
        cem = 'COMMON'
        self.dicts[cem] = {}
        self._record(counters, cem, self.SINGLEWORD,
                  POSITIVE_COUNT, RATIO_POS2NEG, POSITIVE_COVER_RATIO, VERBOSE) 

    def _fn(self, key):
        '''
        组合筛选函数，以(mention, pos)作为输入，得到符合判断的
        '''
        pass
    def ALL(self, key):
        return True
    def NOSPACE(self, key):
        return not any([p == ' ' for p in key.pos]) 
    def SINGLEWORD(self, key):
        return len(key.pos) == 1

    def _record(self, counters, cem, fn,
                   POSITIVE_COUNT, RATIO_POS2NEG, COVER_RATIO, VERBOSE):
        '''
        counters 正例计数
        cem 标签
        fn 筛选函数
        **贪婪的字典(尽量多地根据用户的预期来收集词汇)**
        POSITIVE_COUNT 正例个数下限 #
        RATIO_POS2NEG 正/负比例 # 
        COVER_RATIO 期望的字典覆盖率 # 至少这么多，为此可以突破RATIO_POS2NEG
        VERBOSE 输出控制 0,1,2,3
        '''
        negative_keys = [key for key in counters.keys() if key != cem]
        positive_counter = {key:value for key, value in counters[cem].items() if fn(key)}
        total_counts_positive = sum(positive_counter.values())
        total_counts_positive_in_dict = 0
        pq_positive = PQ(priorityItemInH) # (mention, pos, positive_count, negative_count) 
        for entity, frequency in positive_counter.items():
            negative_count = 0
            for key in negative_keys:
                if entity in counters[key]:
                    negative_count += counters[key][entity]
            if frequency >= POSITIVE_COUNT and ( (negative_count == 0) or (frequency/negative_count > RATIO_POS2NEG) ):
                # 满足条件进入字典
                for key in negative_keys:
                    if key in self.dicts.keys(): # 无冲突
                        assert not entity in self.dicts[key], \
                            'Dataset Conflict:({},{}) {}-{}'.format(*entity, key, cem)
                self.dicts[cem].update({entity:cem})
                total_counts_positive_in_dict += frequency
            else: # 否则入优先级队列
                pq_positive.push(ItemInH(*entity, frequency, negative_count))
                
        ratio_positive_in_dict = total_counts_positive_in_dict / total_counts_positive
        if VERBOSE > 0:
            print('For type of cem is {}'.format(cem))
            print('Total in dict_positive: {:.1%} = {}/{}'.format(
                ratio_positive_in_dict, total_counts_positive_in_dict, total_counts_positive))
            
        records = deque(maxlen = self.verbose[min(VERBOSE,len(self.verbose))])

        item = pq_positive.pop()
        priorityPast = 1
        priorityThreshold = priorityRatio(1/RATIO_POS2NEG)
        while ratio_positive_in_dict < COVER_RATIO or priorityItemInH(item) == priorityPast or priorityItemInH(item) >= priorityThreshold:
            for key in negative_keys:
                    if key in self.dicts.keys(): # 无冲突
                        assert not entity in self.dicts[key], \
                            'Dataset Conflict:({},{}) {}-{}'.format(*entity, key, cem)
            self.dicts[cem].update({EntityInD(item.mention, item.pos):cem})
            total_counts_positive_in_dict += item.positive_count
            ratio_positive_in_dict = total_counts_positive_in_dict / total_counts_positive
            records.append(item)
            priorityPast = priorityItemInH(item)
            if ratio_positive_in_dict >= 1: # 设置退出条件，防止pop一个空的队列。
                break
            item = pq_positive.pop()
        
        for item in records:
            print('{} {} {} {}'.format(*item))    
        if VERBOSE > 0:
            print('Total in dict_positive: {:.1%} = {}/{}'.format(
                ratio_positive_in_dict, total_counts_positive_in_dict, total_counts_positive))

class SizingSystem:
    def __init__(self):
        self.threshold = None
        self.counter = Counter()
    
    def size(self, counter, 
                     RATIO = 0.9, MIN_THRESHOLD = 3, MARGINAL_BENEFIT = 0.0001, 
                     VERBOSE = 2):
        '''
        Return a tuple (MIN_THRESHOLD, max_threshold).
        RATIO - didn't involve the min_loss_ratio. There are a lot hints under MIN_THRESHOLD. But we can rely on the dict.
        As for the longer ones, maybe we need a system-name recognizer.
        VERBOSE - could be valued as 0,1,2,3.
        
        Recommand Settings:
        (0.8, 4, 0.003, 2)
        (0.9, 3, 0.003, 2)
        '''
        for key, value in counter.items():
            self.counter.update({len(key.mention):value})
        assert RATIO < 1
        assert MARGINAL_BENEFIT > 0
        MAX_THRESHOLD = max([key for key in self.counter.keys()])
        total_accumulative_amounts = sum([ value for value in self.counter.values()])
        min_accumulative_amounts = sum([ self.counter[i] for i in range(MIN_THRESHOLD) if i in self.counter.keys() ])
        min_loss_ratio = min_accumulative_amounts / total_accumulative_amounts
        if VERBOSE > 0:
            print('MIN THRESHOLD, MAX THRESHOLD: {}-{}'.format(MIN_THRESHOLD, MAX_THRESHOLD))
            print('Min loss ratio: {:.1%} = {}/{}'.format(min_loss_ratio, min_accumulative_amounts, total_accumulative_amounts))
        max_threshold = MIN_THRESHOLD
        accumulative_amounts = 0
        accumulative_ratio = accumulative_amounts / total_accumulative_amounts
        verbose = [0, 1, 5, 100]
        records = deque(maxlen = verbose[min(VERBOSE,3)])
        while accumulative_ratio < RATIO:
            density_amount = self.counter[max_threshold] if max_threshold in self.counter.keys() else 0
            # marginal_benefit = density_amount / total_accumulative_amounts
            marginal_benefit = max(density_amount / total_accumulative_amounts, 
                                       (1 - min_loss_ratio - accumulative_ratio) / (MAX_THRESHOLD + 1 - max_threshold) )
            accumulative_amounts += density_amount
            accumulative_ratio = accumulative_amounts / total_accumulative_amounts
            records.append((max_threshold, accumulative_ratio, accumulative_amounts, total_accumulative_amounts, marginal_benefit))
            if marginal_benefit < MARGINAL_BENEFIT:
                break
            max_threshold += 1
        for record in records:
            print('Set max threshold to {:3}, captured ratio:{:.1%} = {:>5}/{:>5}, marginal_benefit: {:.3%}'.format(*record))
        self.threshold = (MIN_THRESHOLD, max_threshold)

# def priorityItemInPOSS1(item):
#     return 1 + 2 * math.log10(item.positive_count) - 0.5 * item.negative_count / item.positive_count - item.pos_length
# # annotation中最常出现
# def priorityItemInPOSS2(item):
#     return item.positive_count
# # 捕捉长尾与annotation中最常出现数据
# def priorityItemInPOSS3(item):
#     return 1 + 5 * math.log10(item.positive_count) - 0.5 * item.negative_count / item.positive_count - item.pos_length
# # 零错误 10%
# def priorityItemInPOSS4(item):
#     return - item.negative_count / item.positive_count
# # 非线性/ ratio / length （10，4）
# def priorityItemInPOSS5(item):
#     ratio =  max(0, item.negative_count / item.positive_count - 10)
#     length = max(0, item.pos_length - 4)
#     return - math.log10(1+ratio) - length


class POSStater:
    def __init__(self):
        # self.priorityItemInPOSS = priorityItemInPOSS5
        # self.pq = PQ(self.priorityItemInPOSS)
        
        self.counter_positive = Counter()
        self.counter_negative = Counter()
        
        self.list_pos = []
        
        self.verbose = [0, 2, 10, 1000]
    
    def getPOSs(self):
        return set(self.list_pos)
    
    def __repr__(self):
        return "{.__name__}()".format(type(self))
    
    def stat(self, counter_annotation, counter_common,
            POS_LENGTH = 4, RATIO_POS2NEG = 0.1, COVER_RATIO = 0.9,
            VERBOSE = 2):
        def priorityItem(item):
            ratio =  max(0, item.negative_count / item.positive_count - 1 / RATIO_POS2NEG)
            length = max(0, item.pos_length - POS_LENGTH)
            return - math.log10(1+ratio) - length
        self.priorityItem = priorityItem
        self.pq = PQ(self.priorityItem)
        
        
        for entity, frequency in counter_annotation.items():
            self.counter_positive[entity.pos] += frequency
        for entity, frequency in counter_common.items():
            self.counter_negative[entity.pos] += frequency
        for pos in self.counter_positive.keys():
            self.pq.push(ItemInPOSS(pos, len(pos), self.counter_positive[pos], self.counter_negative[pos]))
        self._POSSelect(COVER_RATIO, VERBOSE)
    

    def _POSSelect(self, COVER_RATIO, VERBOSE):
        '''
        length  
        count
        ratio_pos2neg
        
        '''
        total_counts_positive = sum(self.counter_positive.values())
        total_counts_pos = len(self.counter_positive.keys())
        total_counts_positive_in_list = 0
        total_counts_pos_in_list = 0
        ratio_positive_in_list = total_counts_positive_in_list / total_counts_positive
        
        if VERBOSE > 0:
            print('Total counts of pos : {}'.format(total_counts_pos))
            print('Total counts of positive mention : {}'.format(total_counts_positive))
            
        records = deque(maxlen = self.verbose[min(VERBOSE,len(self.verbose))])

        item = self.pq.pop()
        priorityPast = self.pq.priorityfc(item)
        while ratio_positive_in_list < COVER_RATIO or self.pq.priorityfc(item) == priorityPast:
            self.list_pos.append(item.pos)
            total_counts_positive_in_list += item.positive_count
            total_counts_pos_in_list += 1
            ratio_positive_in_list = total_counts_positive_in_list / total_counts_positive
            records.append(item)
            priorityPast = self.pq.priorityfc(item)
            if ratio_positive_in_list >= 1: # 设置退出条件，防止pop一个空的队列。
                break
            item = self.pq.pop()
        
        for item in records:
            print('{} {} {} {}'.format(*item))    
        if VERBOSE > 0:
            print('Total in dict_positive: {:.1%} = {}/{}, {}/{}'.format(
                ratio_positive_in_list, total_counts_positive_in_list, total_counts_positive,
                                        total_counts_pos_in_list, total_counts_pos))
        
def writeFile(dicts, list_pos, threshold, lists_acca, list_pos_star, filename):
    '''
    保存FilterDev得到的dicts，list_pos,threshold到json文件。
    '''
    path = _path(filename)
    jsonstr = json.dumps({'dicts': Dicts2Dj(dicts),
                          'list_pos': list_pos,
                          'threshold': threshold,
                          'lists_acca': lists_acca,
                          'list_pos_star': list_pos_star,})
    with open(path, 'w') as f:
        f.write(jsonstr)
    return path

def readFile(path, bl_pathAbsolute = False):
    '''
    提供文件名路径，得到dicts, list_pos, threshold。
    
    '''
    path = _path(path) if not bl_pathAbsolute else path
    with open(path, 'r') as f:
        jsonstr = f.read()
    d = json.loads(jsonstr)
    dicts = Dj2Dicts(d['dicts'])
    list_pos = [ tuple(item) for item in d['list_pos'] ]
    threshold = tuple(d['threshold'])
    lists_acca = d['lists_acca']
    list_pos_star = [ tuple(item) for item in d['list_pos_star'] ]
    return dicts, list_pos, threshold, lists_acca, list_pos_star
    
def Dicts2Dj(dicts):
    '''
    将一个dicts对象转化为jsonstr,然后保存。
    dicts : A list of dict{EntityInD('mention', 'pos'): 'type_cem'}
    filename : 用以构建文件存储路径
    返回json文件路径。
    '''
    dicts_for_json = {}
    for key in dicts.keys():
        dicts_for_json[key] = {}
        for i, c in enumerate(dicts[key].keys()):
            dicts_for_json[key][i] = c._asdict()
    # jsonstr = json.dumps(dicts_for_json)
    
    return dicts_for_json

def Dj2Dicts(dicts_for_json):
    '''
    将一个jsonstr对象读取，并转化为dicts对象。
    path: json文件路径。
    '''
    # dicts_for_json = json.loads(jsonstr)
    # return dicts_for_json
    dicts = {}
    for key in dicts_for_json.keys():
        dicts[key] = {}
        for tuple_dict in dicts_for_json[key].values():
            entity_in_d = EntityInD(mention = tuple_dict['mention'], 
                                    pos = tuple(tuple_dict['pos']))
            dicts[key][entity_in_d] = key
    return dicts
 
