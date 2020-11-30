# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:53:56 2020

@author: hanyl
"""
from __future__ import unicode_literals, print_function, division
import os
import torch
import string
from io import open
import glob # 匹配文件名的包
# import unicodedata
from collections import defaultdict
import random

class TrainingData():
    def __init__(self):
        self.DirPath = 'D:\Desktop\Python\CHEMDNER\data'
        self.fn_wildcard = '\\*.txt'
        
        self.categories = None
        self.all_categories = []
        self.n_categories = 0
        
        self.normalizer = NormalizerLetter()
        self.vectorizer = VectorizerOH(self.normalizer.all_letters)
        # self.dataloaders = None {}
        
        self.train2valid_ratio = 0.8
        
        
    def getTrainingData(self, sentences, all_classes = False, threshold = None):
        '''
        目前只保留了文本信息.
        sentences entities 带有 chem 和 ‘’ 标记的数据
        all_classes Bool True：载入全部的分类; False:二分类
        threshold (4,27) 是否对长度进行一定限制；
        '''
        self.categories = defaultdict(list)
        for sentence in sentences:
            for entity in sentence.entities:
                if entity.type_cem == '':
                    self.categories['COMMON'].append(entity.mention)
                else:
                    if not threshold or len(entity.mention) in range(*threshold):
                        if all_classes:
                            self.categories[entity.type_cem].append(entity.mention)
                        else:
                            self.categories['chem'].append(entity.mention)
        self.all_categories = list(self.categories.keys())
        self.n_categories = len(self.all_categories)      
               
    def saveTrainingData(self):
        '''
        保存TrainingData到文件。
        '''
        for type_cem in self.categories.keys():
            with open(os.path.join(self.DirPath, '{}.txt'.format(type_cem)), 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.categories[type_cem]))

    def loadTrainingData(self):
        self.categories = defaultdict(list)
        self.all_categories = []
        print(self._findFiles(self.DirPath + self.fn_wildcard))
        for filename in self._findFiles(self.DirPath + self.fn_wildcard):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self._readLines(filename)
            self.categories[category] = lines
        self.n_categories = len(self.all_categories)
        
    # Find the files
    def _findFiles(self, path): return glob.glob((path))
    
    # Read the files
    def _readLines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return lines
    
    def normalize(self):
        for key, mentions in self.categories.items():
            self.categories[key] = [self.normalizer.normalize(mention) for mention in mentions]
    
    def trainValidSplit(self):
        self.training_categories = {}
        self.training_categories['train'] = {}
        self.training_categories['valid'] = {}
        for key, mentions in self.categories.items():
            indexes = list(range(len(mentions)))
            random.shuffle(indexes)
            index_split = int(len(indexes) * self.train2valid_ratio)
            self.training_categories['train'][key] = mentions[:index_split]
            self.training_categories['valid'][key] = mentions[index_split:]
        # del self.categories
    
    def loadDataLoaders(self):
        self.trainValidSplit()
        self.dataloaders = {}
        self.dataloaders['train'] = ChooserRandom(all_categories = self.all_categories, 
                                               categories = self.training_categories['train'],
                                               bl_random = True)
        self.dataloaders['valid'] = ChooserRandom(all_categories = self.all_categories, 
                                               categories = self.training_categories['valid'],
                                               bl_random = True)
    
    def sample(self, key, n):
        return self.dataloaders[key].choose(self.all_categories, self.training_categories[key], n = n)

    def sampleTensors(self, key = 'train', n = 16):
        samples = self.sample(key, n)
        x_tensor = self.vectorizer.linesToTensor(x_str for x_str, y in samples)
        y = torch.tensor([y for _, y in samples])
        return x_tensor, y

        
    
    
class NormalizerLetter:
    def __init__(self):
        self.all_letters = string.ascii_letters + " .,'–+/()" + string.digits + "*" 
             
    # Normalize the string.
    def normalize(self, s):  return "".join( i if i in self.all_letters else '*' for i in s )
    # def unicodeToAscii(self, s):
    #     return ''.join(
    #         c for c in unicodedata.normalize('NFD', s)
    #         if unicodedata.category(c) != 'Mn'
    #         and c in self.all_letters
    #     )

class ChooserRandom:
    def __init__(self, all_categories, categories, seed = 42, bl_random = True):
        self.bl_random = bl_random
        if bl_random:
            self.seed = seed
            random.seed(self.seed)
        # To index item in categories
        self.r2c = []
        self.r2i = []
        # To Choose
        self.shuffle(all_categories, categories)

    def shuffle(self, all_categories, categories):
        for i, key in enumerate(all_categories):
            count = len(categories[key])
            self.r2c.extend([i]*count)
            self.r2i.extend(range(count))
        self.total_count = len(self.r2c)
        self.r = list(range(self.total_count))
        if self.bl_random:
            random.shuffle(self.r)
    
    def _shuffle(self):
        if self.bl_random:
            random.shuffle(self.r)
        self.total_count = len(self.r2c)
    
    def choose(self, all_categories, categories, n = 16):
        if self.total_count == 0:
            self._shuffle()
        start = max(0, self.total_count - n)
        samples = [ (categories[all_categories[self.r2c[i]]][self.r2i[i]], self.r2c[i])
                       for i in self.r[start:self.total_count] ]  
        self.total_count = start    
        return samples

# One HOT Vectorizer
class VectorizerOH:
    def __init__(self, all_letters):
        self.all_letters = all_letters
        self.n_letters = len(self.all_letters)
        self.line_length = 30
        
    def letterToIndex(self, letter):
        return self.all_letters.find(letter)

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(self, line):
        tensor = torch.zeros(self.line_length, 1, self.n_letters) # pad
        line_length = min(self.line_length, len(line)) # trunc
        for li, letter in enumerate(line[:line_length]):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

    def linesToTensor(self, lines):
        tensor = torch.cat([self.lineToTensor(line) for line in lines], dim = 1)
        return tensor

    def indexToLetter(self, index):
        return self.all_letters[index]
    
    def _tensorToLine(self, t, i = 0):
        indexes_tensor = t[torch.nonzero(t[:,1] == i).squeeze()]
        if len(indexes_tensor.size()) == 1:
            indexes_tensor.unsqueeze_(0)
        indexes = indexes_tensor[:,2]
        return ''.join([self.indexToLetter(index) for index in indexes])
    
    def tensorToLines(self, tensor):
        t = torch.nonzero(tensor)
        lines = [ self._tensorToLine(t, i)
                 for i in range(torch.max(t[:,1]).item()+1) ]
        return lines
    
    
# def findFiles(path): return glob.glob(path)

# # print(findFiles(DirPath + '\\name\*.txt'))

# def normalize(s):
#     return "".join( i if i in all_letters else '*' for i in s )

# all_letters = string.ascii_letters + " .,'–+/()" + string.digits + "*" # 72
# # all_letters = string.ascii_letters + " .,;'"
# n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn'
#         and c in all_letters
#     )

# print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
# category_lines = {}
# all_categories = []

# Read a file and split into lines
# def readLines(filename):
#     lines = open(filename, encoding='utf-8').read().strip().split('\n')
#     return [unicodeToAscii(line) for line in lines]

# Find letter index from all_letters, e.g. "a" = 0
# def letterToIndex(letter):
#     return all_letters.find(letter)

# # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
# def letterToTensor(letter):
#     tensor = torch.zeros(1, n_letters) # n_letters 在上面定义了 all_letters = string.ascii_letters + " .,;'"  n_letters = len(all_letters)
#     tensor[0][letterToIndex(letter)] = 1
#     return tensor

# # Turn a line into a <line_length x 1 x n_letters>,
# # or an array of one-hot letter vectors
# def lineToTensor(line):
#     tensor = torch.zeros(len(line), 1, n_letters)
#     for li, letter in enumerate(line):
#         tensor[li][0][letterToIndex(letter)] = 1
#     return tensor

# category_lines = {}
# all_categories = []
# print(findFiles(DirPath + '\\name\*.txt'))
# for filename in findFiles(DirPath + '\\name\*.txt'):
#     category = os.path.splitext(os.path.basename(filename))[0]
#     all_categories.append(category)
#     lines = readLines(filename)
#     category_lines[category] = lines

# n_categories = len(all_categories)
# n_letters = len(all_letters)

