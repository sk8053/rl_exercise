# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:44:06 2023

@author: seongjoon
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.autograd as autograd

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Q_Net(nn.Module):
    # dueling Network
    def __init__(self, n_in, n_mid=128, n_out=20):
        super(Q_Net, self).__init__()
        self.hidden_size = 32
        self.num_layers = 2

        self.mlp = nn.LSTM(n_in,self.hidden_size, self.num_layers,
                           batch_first = True, bidirectional = False)
       # self.fc_bs = nn.Sequential(
       #     nn.Linear(self.hidden_size, 128),
       #     nn.ReLU(),
            #nn.Linear(128, 64),
       # )
        self.fc = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_mid),
        )
        self.fc_advt = nn.Sequential(
            nn.Linear(n_mid, 128 ),
            nn.ReLU(),
            nn.Linear(128, n_out)
        )
        self.fc_v = nn.Sequential(
            nn.Linear(n_mid, 128),
            nn.ReLU(),
            nn.Linear(128,n_out)
        )
    
    def forward(self, state):
        
        #if state.dim() ==1:
        #    state = state[None][None]
        #elif state.dim() == 2:
        #    state = state[:,None]
        if state.dim() == 1:
            state = state[None]
        #out1 = self.fc(state)
        h0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size)) #
        c0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size))

        #out, _ = self.mlp(state, (h0, c0))
        out = self.fc(state)
        advantage = self.fc_advt(out) # [:,-1,:]

        value = self.fc_v(out).expand(-1, advantage.shape[1]) #[:,-1,:]
        #output is the sum of value function and 
        output = value + advantage - advantage.mean(1, keepdim = True).expand(-1, advantage.shape[1])
        #output1 = self.fc_bs(out[:,-1,:])
        #output2 = self.fc_uav(out[:, -1, :])

        return output[:,:8],output[:,8:]
    def reset_noise(self):
        return

# https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class Noisy_Q_Net(nn.Module):
    # dueling Network
    def __init__(self, n_in, n_mid=128, n_out=20, std_init = 0.3):
        super(Noisy_Q_Net, self).__init__()
        self.hidden_size = 32
        self.num_layers = 2

        self.mlp = nn.LSTM(n_in, self.hidden_size, self.num_layers,
                           batch_first=True, bidirectional=False)
        # self.fc_bs = nn.Sequential(
        #     nn.Linear(self.hidden_size, 128),
        #     nn.ReLU(),
        # nn.Linear(128, 64),
        # )
        self.fc = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.ReLU(),
            NoisyLinear(128, 128, std_init=std_init),
            nn.ReLU(),
            NoisyLinear(128, n_mid, std_init=std_init),
        )
        self.fc_advt = nn.Sequential(
            NoisyLinear(n_mid, 128, std_init=std_init),
            nn.ReLU(),
            nn.Linear(128, n_out)
        )
        self.fc_v = nn.Sequential(
            NoisyLinear(n_mid, 128, std_init=std_init),
            nn.ReLU(),
            nn.Linear(128, n_out)
        )

    def forward(self, state):
        # if state.dim() ==1:
        #    state = state[None][None]
        # elif state.dim() == 2:
        #    state = state[:,None]
        if state.dim() == 1:
            state = state[None]
        # out1 = self.fc(state)
        h0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size))  #
        c0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size))

        # out, _ = self.mlp(state, (h0, c0))
        out = self.fc(state)
        advantage = self.fc_advt(out)  # [:,-1,:]

        value = self.fc_v(out).expand(-1, advantage.shape[1])  # [:,-1,:]
        # output is the sum of value function and
        output = value + advantage - advantage.mean(1, keepdim=True).expand(-1, advantage.shape[1])
        # output1 = self.fc_bs(out[:,-1,:])
        # output2 = self.fc_uav(out[:, -1, :])

        return output[:, :8], output[:, 8:]

    def reset_noise(self):
        return