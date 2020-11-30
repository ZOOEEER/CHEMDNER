# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:37:06 2020

@author: hanyl
"""
import os
import torch
import torch.nn as nn

__all__ = [
    "LSTM", "GRU",
    "saveModel", "loadModel", 
]

MODELPATH = 'D:\Desktop\Python\CHEMDNER\model'
def _path(filename): return os.path.join(MODELPATH, '{}'.format(filename))

# Naive RNN
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()

#         self.hidden_size = hidden_size

#         self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
#         self.i2o = nn.Linear(input_size + hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def _forward(self, x_input, hidden):
#         combined = torch.cat((x_input, hidden), 1)
#         hidden = self.i2h(combined)
#         output = self.i2o(combined)
#         output = self.softmax(output)
#         return output, hidden

#     def forward(self, input, hidden):
#         for i in range(input.size()[0]):
#             output, hidden = self._forward(input[i], hidden)
#         return output
        
#     def initHidden(self, n_batch = 1):
#         return torch.zeros(n_batch, self.hidden_size)

class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN2, self).__init__()

        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2u = nn.Linear(input_size + hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.u2h = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.u2o = nn.Linear(hidden_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def _forward(self, x_input, hidden):
        combined = torch.cat((x_input, hidden[0]), 1)
        hidden0 = self.i2h(combined)
        u = self.i2u(combined)
        u = self.relu(u)
        combined2 = torch.cat((u, hidden[1]), 1)
        hidden1 = self.u2h(combined2)        
        output = self.u2o(combined2)
        return output, torch.cat((hidden0.unsqueeze(0), hidden1.unsqueeze(0)), 0)

    def forward(self, input, hidden):
        for i in range(input.size()[0]):
            output, hidden = self._forward(input[i], hidden)
        output = self.softmax(output)
        return output
    
    def initHidden(self, n_batch = 1):
        return torch.zeros(2, n_batch, self.hidden_size)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = self.num_layers)
        self.c2o = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, input, hidden_cell):
        output, (hidden, cell_state) = self.lstm(input, (hidden_cell[0], hidden_cell[1]))
        o = self.c2o(output[-1])
        # o = self.softmax(o)
        return o
        
    def initHidden(self, n_batch = 1):
        return torch.zeros(2, self.num_layers, n_batch, self.hidden_size)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout = 0, model_type = 'gru'):
        super(GRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        if model_type == 'gru':
            self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size,
                                num_layers = self.num_layers, dropout = dropout)
        else:
            self.gru = nn.RNN(input_size = input_size, hidden_size = hidden_size,
                                num_layers = self.num_layers, nonlinearity = 'relu')
        self.c2o = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, input, hidden):
        output, hn = self.gru(input, hidden)
        o = self.c2o(output[-1])
        # o = self.softmax(o)
        return o 
        
    def initHidden(self, n_batch = 1):
        return torch.randn(self.num_layers, n_batch, self.hidden_size)



def saveModel(model, filename):
    torch.save(model, _path(filename))

def loadModel(filename):
    return torch.load(_path(filename))
