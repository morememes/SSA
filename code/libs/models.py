import torch
from torch import nn, optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from copy import copy, deepcopy

import numpy as np
import pandas as pd

from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, Checkpoint
from skorch.helper import predefined_split
from skorch.dataset import Dataset

class FCModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(FCModel, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid1 = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)    
        out = x

        return out
    
class LSTM(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 36, output_size=1):
        super(LSTM, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        #self.seq_length = seq_length #sequence length
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True) #lstm
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_size, output_size) #fully connected 1 #fully connected last layer
    
    def forward(self, x):
        device = self.dummy_param.device
        
        h_0  = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(device) #internal state
        # Propagate input through LSTM
        x = x.view(len(x), -1, 1)
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        out = hn
        #out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        
        return out.reshape(-1, 1)
    
class GRU(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 36, output_size=1):
        super(GRU, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        #self.seq_length = seq_length #sequence length

        self.gru = nn.GRU(input_size, hidden_size, batch_first = True) #lstm
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_size, output_size) #fully connected 1 #fully connected last layer
    
    def forward(self, x):
        device = self.dummy_param.device
        
        h_0  = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(device) #hidden state #internal state
        # Propagate input through LSTM
        x = x.view(len(x), -1, 1)
        output, hn = self.gru(x, h_0) #lstm with input, hidden, and internal state
        out = hn
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        
        return out.reshape(-1, 1)
    
class RNN(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 36, output_size=1):
        super(RNN, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        #self.seq_length = seq_length #sequence length

        self.rnn = nn.RNN(input_size, hidden_size, batch_first = True, nonlinearity  = 'relu') #lstm
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_size, output_size)#nn.Linear(hidden_size, output_size) #fully connected 1 #fully connected last layer
    
    def forward(self, x):
        device = self.dummy_param.device
        
        h_0  = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(device) #hidden state #internal state
        # Propagate input through LSTM
        x = x.view(len(x), -1, 1)
        output, hn = self.rnn(x, h_0) #lstm with input, hidden, and internal state
        out = hn
        #out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        
        return out.reshape(-1, 1)
    
class MyEarlyStopping(EarlyStopping):
    def __init__(self,
            monitor='valid_loss',
            patience=5,
            threshold=1e-4,
            threshold_mode='rel',
            lower_is_better=True,
            sink=print,
    ):
        super().__init__(monitor, patience, threshold, threshold_mode, lower_is_better, sink)
        self.best_model_params_ = None

    def on_epoch_end(self, net, **kwargs):
        current_score = net.history[-1, self.monitor]
        if not self._is_score_improved(current_score):
            self.misses_ += 1
        else:
            self.misses_ = 0
            self.dynamic_threshold_ = self._calc_new_threshold(current_score)
            self.best_model_params_ = deepcopy(net.module_.state_dict())

        if self.misses_ == self.patience:
            net.module_.load_state_dict(self.best_model_params_)
            if net.verbose:
                self._sink("Stopping since {} has not improved in the last "
                           "{} epochs.".format(self.monitor, self.patience),
                           verbose=net.verbose)
            raise KeyboardInterrupt