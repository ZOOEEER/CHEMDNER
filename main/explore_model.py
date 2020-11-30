# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:52:21 2020

@author: hanyl
"""

from Model import saveModel, loadModel
from classifier import classiferDev
from Model import LSTM, GRU
# cd = classiferDev()
# cd.batch_size = 128
# cd.num_epochs = 4
# cd.lr = 0.0001
# cd.train()
# saveModel(cd.model, 'rnn2-128-train-train-5')
# cd.model = loadModel('rnn2-128-train-train-5')
# saveModel(cd.model, 'rnn2-128-train-train-10')


cd2 = classiferDev()
cd2.name = 'GRU_128_3-30'
cd2.setModel(GRU(input_size = 72, 
                 hidden_size = 128, 
                 output_size = 2, 
                 num_layers = 3,
                 dropout = 0, 
                 model_type = 'gru'))
# cd2.batch_size = 128
cd2.num_epochs = 10
# cd2.lr = 0.001
cd2.train()
saveModel(cd2.model, cd2.name)
cd2.evaluation_training_process()

'GRU_128_3-30'