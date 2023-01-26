# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:44:06 2023

@author: seongjoon
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
class Q_Net(nn.Module):
    # dueling Network
    def __init__(self, n_in, n_mid=128, n_out=20):
        super(Q_Net, self).__init__()
        self.hidden_size = 32
        self.num_layers = 2

        self.mlp = nn.LSTM(n_in,self.hidden_size, self.num_layers,
                           batch_first = True, bidirectional = False)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_out),

        )
        #self.fc_advt = nn.Linear(self.hidden_size, n_out) #
        #self.fc_v = nn.Linear(self.hidden_size, 1) #
    
    def forward(self, state):
        
        if state.dim() ==1:
            state = state[None][None]
        elif state.dim() == 2:
            state = state[:,None]
        #if state.dim() == 1:
        #    state = state[None]
        #out1 = self.fc(state)
        h0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size)) #
        c0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size))

        out1, _ = self.mlp(state, (h0, c0))
        #advantage = self.fc_advt(out1[:,-1,:]) #

        #value = self.fc_v(out1[:,-1,:]).expand(-1, advantage.shape[1]) #
        #output is the sum of value function and 
        #output = value + advantage - advantage.mean(1, keepdim = True).expand(-1, advantage.shape[1])
        output = self.fc(out1[:,-1,:])
        return output