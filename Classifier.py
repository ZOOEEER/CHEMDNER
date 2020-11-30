# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:52:37 2020

@author: hanyl
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os

from Model import loadModel, saveModel
from Model import RNN2
from TrainingData import TrainingData

IMGPATH = 'D:\Desktop\Python\CHEMDNER\img'


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class ClassifierDev:
    def __init__(self, name = 'defaultName', model = None, model_name = '', data_dirPath = ''):
        self.name = name
        # data
        self.trainingdata = TrainingData(data_dirPath)
        self.trainingdata.loadTrainingData()
        self.trainingdata.loadDataLoaders()

        # device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model = self.model.to(self.device)
        
        # train settings
        self.batch_size = 128
        self.num_epochs = 10
        self.lr = 0.001
        # self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # model
        n_letters = self.trainingdata.vectorizer.n_letters # 72
        n_hidden = 128
        n_categories = self.trainingdata.n_categories # 2
        if not model and not model_name:
            self.setModel(RNN2(n_letters, n_hidden, n_categories))
        elif model_name != '':
            self.setModelbyName(model_name)
        elif model:
            self.setModel(model)
            
        # training process evaluation
        # self.all_losses = {}
        # self.all_losses['train'] = []
        # self.all_losses['valid'] = []
        # self.all_corrects = {}
        # self.all_corrects['train'] = []
        # self.all_corrects['valid'] = []       
                
    def setModel(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.model = self.model.to(self.device)
        
        self.all_losses = {}
        self.all_losses['train'] = []
        self.all_losses['valid'] = []
        self.all_corrects = {}
        self.all_corrects['train'] = []
        self.all_corrects['valid'] = [] 
    
    def setModelbyName(self, modelname):
        self.setModel(loadModel(modelname))
    
    def setData(self, DirPath):
        self.trainingdata = TrainingData(DirPath)
        self.trainingdata.loadTrainingData()
        self.trainingdata.loadDataLoaders()
    
    def _train(self, x_tensor, y, hidden):
        self.optimizer.zero_grad()
        output = self.model(x_tensor, hidden)

        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        predictions = torch.argmax(output, 1)
        n_correct = torch.sum(predictions == y)
        return n_correct.item(), loss.item()
    
    def train(self):
        
        n_iters = self.num_epochs * math.ceil(len(self.trainingdata.dataloaders['train'].r) / self.batch_size) 
        print_every = 10
        plot_every = 50
        
        current_loss = 0
        current_correct = 0
        current_sample_num = 0
        
        start = time.time()
        print('Train Begin! n_iters = {}'.format(n_iters))
        for n_iter in range(n_iters):
            x_tensor, y = self.trainingdata.sampleTensors('train', self.batch_size)
            sample_num = y.size()[0]
            hidden = self.model.initHidden(sample_num)
            x_tensor = x_tensor.to(self.device)
            y = y.to(self.device)
            hidden = hidden.to(self.device)
            n_correct, loss = self._train(x_tensor, y, hidden)
            current_loss += loss * sample_num
            current_correct += n_correct
            current_sample_num += sample_num
            
            # Add current loss avg to list of losses
            if n_iter % plot_every == 0:
                self.all_losses['train'].append(current_loss / current_sample_num)
                self.all_corrects['train'].append(current_correct / current_sample_num)
                self.valid(print_every)
                print('Validation_iters={} [{}]: Loss={:<5.4}, Accuracy={:<4.1%}'.format(
                    print_every, timeSince(start),
                    self.all_losses['valid'][-1], self.all_corrects['valid'][-1]))
                
            # Print iter number, loss, name and guess
            if n_iter % print_every == 0:
                print('N_iter={}({:.1%})[{}]: Loss={:<5.4}, Accuracy={:<4.1%}({}/{})'.format(
                    n_iter, n_iter/n_iters, timeSince(start),
                    current_loss / current_sample_num, current_correct / current_sample_num,
                    current_correct, current_sample_num))
                current_loss = 0 
                current_correct = 0
                current_sample_num = 0
        print('Train Over! total time = {}'.format(timeSince(start)))
            
            
    def _valid(self, x_tensor, y, hidden):
        output = self.model(x_tensor, hidden)
        loss = self.criterion(output, y)
        # print(output)
        # print(y)
        predictions = torch.argmax(output, 1)
        n_correct = torch.sum(predictions == y)
        return n_correct.item(), loss.item()
        
        
    def valid(self, n_iters):

        current_loss = 0
        current_correct = 0
        current_sample_num = 0
        

        for n_iter in range(n_iters):
            x_tensor, y = self.trainingdata.sampleTensors('valid', self.batch_size)
            sample_num = y.size()[0]
            hidden = self.model.initHidden(sample_num)
            x_tensor = x_tensor.to(self.device)
            y = y.to(self.device)
            hidden = hidden.to(self.device)
            n_correct, loss = self._valid(x_tensor, y, hidden)
            current_loss += loss * sample_num
            current_correct += n_correct
            current_sample_num += sample_num

        self.all_losses['valid'].append(current_loss / current_sample_num)
        self.all_corrects['valid'].append(current_correct / current_sample_num)

    def evaluation_training_process(self):
        import matplotlib.pyplot as plt
        # import matplotlib.ticker as ticker
        
        plt.figure()
        plt.plot(self.all_losses['train'])
        plt.plot(self.all_losses['valid'])
        if not self.name == 'defaultName':
            plt.savefig(os.path.join(IMGPATH, self.name + '-losses.png'))

        # plt.plot(self.all_losses['valid'])
        plt.figure()
        plt.plot(self.all_corrects['train'])
        plt.plot(self.all_corrects['valid'])
        if not self.name == 'defaultName':
            plt.savefig(os.path.join(IMGPATH, self.name + '-accuracy.png'))
    

from TrainingData import NormalizerLetter, VectorizerOH
class Classifier:
    def __init__(self, category_names, model = None, normalizer = None, vectorizer = None, model_name = ''):
        
        self.category_names = category_names
        self.normalizer = NormalizerLetter()
        self.vectorizer = VectorizerOH(self.normalizer.all_letters)
        if model:
            self.model = model
        elif model_name:
            self.model = loadModel(model_name)
        else:
            print('Warning! No Model.')
        self.device = "cuda:0"
        self.mapper = MapperNone()
    
    def tensorToType(self, tensor):
        output = self.model(tensor.to(self.device), self.model.initHidden(1).to(self.device))
        type_index = torch.argmax(output).item()
        return self.category_names[type_index]
    
    def mentionToType(self, mention):
        return self.tensorToType(self.vectorizer.lineToTensor(self.normalizer.normalize(mention)))
    
    # def _predict(self, sentence):
    #     chemChar = self.mapper.generateChemChar(sentence)
    #     for entity in sentence.entities[::-1]:
    #         if entity.type_cem == '':
    #             type_cem = self.mentionToType(entity.mention)
    #             entity.type_cem = type_cem
    #     self.mapper.postChemChar(sentence, chemChar)
    
    def tensorToTypes(self, tensor):
        output = self.model(tensor.to(self.device), 
                            self.model.initHidden(tensor.size()[1]).to(self.device))
        type_indexes = torch.argmax(output, 1)
        return [self.category_names[type_index] for type_index in type_indexes]
    def mentionsToTypes(self, mentions):
        return self.tensorToTypes(torch.cat([self.vectorizer.lineToTensor(self.normalizer.normalize(mention))
                                              for mention in mentions], 1))
    def _predict(self, sentence):
        chemChar = self.mapper.generateChemChar(sentence)
        mentions = [entity.mention for entity in sentence.entities if entity.type_cem == '']
        if len(mentions):
            type_cems = self.mentionsToTypes(mentions)
            i = 0
            for entity in sentence.entities:
                if entity.type_cem == '':
                    entity.type_cem = type_cems[i]
                    i += 1
            self.mapper.postChemChar(sentence, chemChar)
    
    def predict(self, sentences, a = 0):
        # self.device = self.model.device
        start = time.time()
        end = a if a else len(sentences)
        for i, sentence in enumerate(sentences[:end]):
            self._predict(sentence)
            if (i+1) % 1000 == 0:
                print(i+1, timeSince(start))

class MapperNone:
    def __init__(self):
        self.type_common = ['COMMON', '']
    def generateChemChar(self, sentence):
        return None
    
    def postChemChar(self, sentence, chemChar):
        sentence.entities = [ entity for entity in sentence.entities 
                             if not entity.type_cem in self.type_common]



class Mapper:
    def __init__(self, dict_point = 3, pred_point = 1, threshold = 0.01):
        self.dict_point = dict_point
        self.pred_point = pred_point
        self.threshold = threshold
        self.type_common = ['COMMON', '']
    
    def score(self, sentence, chemChar, point):
        for entity in sentence.entities:
            if not entity.type_cem in self.type_common:
                for i in range(entity.start, entity.end):
                    chemChar[i] += point
                
    def generateChemChar(self, sentence):
        chemChar = [0] * len(sentence.text)
        self.score(sentence, chemChar, self.dict_point)
        return chemChar
    
    def postChemChar(self, sentence, chemChar):
        self.score(sentence, chemChar, self.pred_point)
        local_max = self.lm(chemChar)
        new_entity = []
        for entity in sentence.entities:
            # print(entity)
            # print(chemChar[entity.start:entity.end])
            # print(local_max[entity.start:entity.end])
            if (not min(chemChar[entity.start:entity.end]) == 0) and \
                min([ chemChar[i]/local_max[i] for i in range(entity.start, entity.end)]) >= self.threshold:
                # print('OhYes! in!')
                new_entity.append(entity)
        sentence.entities = new_entity
                
            
    def lm(self, chemChar):
        local_max = [0] * len(chemChar)
        local_max[0] = chemChar[0]
        for i in range(1, len(chemChar)):
            local_max[i] = 0 if chemChar[i] == 0 else max(local_max[i-1], chemChar[i])
        for i in reversed(range(len(chemChar)-1)):
            local_max[i] = 0 if local_max[i] == 0 else max(local_max[i], local_max[i+1])
        return local_max





'==Eval 1== the training process =='
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# plt.figure()
# plt.plot(all_losses)

# '==Eval 2==confusion matrix=='
# # Keep track of correct guesses in a confusion matrix
# confusion = torch.zeros(n_categories, n_categories)
# n_confusion = 10000

# # Just return an output given a line
# def evaluate(line_tensor):
#     hidden = rnn.initHidden()
#     # line_tensor = line_tensor.to(device)
    
#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)
#     return output

# # Go through a bunch of examples and record which are correctly guessed
# for i in range(n_confusion):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     if line_tensor.size()[0]:
#         output = evaluate(line_tensor)
#     guess, guess_i = categoryFromOutput(output)
#     category_i = all_categories.index(category)
#     confusion[category_i][guess_i] += 1

# # Normalize by dividing every row by its sum
# for i in range(n_categories):
#     confusion[i] = confusion[i] / confusion[i].sum()

# # Set up plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(confusion.numpy())
# fig.colorbar(cax)

# # Set up axes
# ax.set_xticklabels([''] + all_categories, rotation=90)
# ax.set_yticklabels([''] + all_categories)

# # Force label at every tick
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# # sphinx_gallery_thumbnail_number = 2
# plt.show()

# '==Use=='
# def predict(input_line, n_predictions=1):
#     # print('\n> %s' % input_line)
#     with torch.no_grad():
#         output = evaluate(lineToTensor(input_line))
        
#         # Get top N categories
#         topv, topi = output.topk(n_predictions, 1, True)
#         predictions = []

#         for i in range(n_predictions):
#             value = topv[0][i].item()
#             category_index = topi[0][i].item()
#             # print('(%.2f) %s' % (value, all_categories[category_index]))
#             predictions.append([value, all_categories[category_index]])
#     return all_categories[category_index]

# predict('Molecular Simulation')
# predict('Ionic Liquid')
# predict('Ethanol')

# all_categories[0]
# category_lines

# for type_cem in [  'COMMON']:
#     preds = [predict(item, 1) == 'COMMON' for item in list(category_lines[type_cem])[:10]]
#     ratio = sum(preds)/len(preds)
#     print('{} {}'.format(type_cem, ratio))



