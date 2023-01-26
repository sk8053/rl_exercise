# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:08:12 2023

@author:seongjoon kang
"""

import numpy as np
import random
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import deque
from memory import memory
import torch

from q_net import Q_Net

class DQNAgent():
    
    def __init__(self, n_states = 8, n_actions = 20, memory_size = 10000, gamma = 0.9,
                 eps_min = 0.01, eps_max = 0.1, batch_size = 80, n_train = 200, 
                 device = None):
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = memory (capacity = memory_size)
        self.device = device
        
        self.eps_min = eps_min
        self.eps_max = eps_max
        
        self.model = Q_Net(n_states, 50, n_actions)
        self.model.to(device)
        print("Num params: ", sum(p.numel() for p in self.model.parameters()))
        
        self.target_model = Q_Net(n_states, 50, n_actions)
        self.target_model.to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-4)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr = 1e-3)
        
     
        self.n_train_sessions = n_train
        self.loss = -1
    def train(self):
        
        if len(self.memory) < self.batch_size:
            return -1.0
        loss_avg = 0.0
        
        for s in range(self.n_train_sessions):
            past_data = self.memory.sample(self.batch_size)
            
            state = torch.zeros((self.batch_size, len(past_data[0]['state']))).to(self.device)
            state_next = torch.zeros((self.batch_size, len(past_data[0]['state_next']))).to(self.device)
            action = torch.zeros((self.batch_size, len(past_data[0]['action'])), dtype = torch.int64).to(self.device)
            reward = torch.zeros((self.batch_size, len(past_data[0]['reward']))).to(self.device)
            #print (past_data[0], action.shape)
            for i in range(self.batch_size):
                
                state[i] = past_data[i]['state']
                state_next[i] = past_data[i]['state_next']
                #print(past_data[i]['action'].shape, action[i].shape)
                action[i] = past_data[i]['action']
                reward[i] =  past_data[i]['reward']
                #done= torch.cat([done, past_data[i]['done']],0)
                
            self.model.eval()
            self.target_model.eval()
            
            # get Q_t (s_t, a_t) from s_t
            
            Q_t = self.model(state) # batch size * number of actions 
            #  = Q_t (s_t, a_t) 
            
            Q_t_bs = Q_t[:,:12].reshape(self.batch_size, 2, -1)
            Q_t_uav = Q_t[:,12:].reshape(self.batch_size, 2, -1)
            
            state_action_value_bs = Q_t_bs.gather(2, action[:,:6][:,None])
            state_action_value_uav = Q_t_uav.gather(2, action[:,6:][:,None])
            
            #state_action_value = state_action_value_bs.sum(-1) + state_action_value_uav.sum(-1)
            state_action_value = torch.cat([state_action_value_bs.squeeze() , state_action_value_uav.squeeze()],1)
            
            # get Q_t(s_t+1, a_t) from s_t+1
            Q_t_1 = self.target_model(state_next) # [batch_size, 80]
            Q_t_1_bs = Q_t_1[:,:12].reshape(self.batch_size,2,-1)
            Q_t_1_uav = Q_t_1[:,12:].reshape(self.batch_size,2,-1)
            
            #Q_t_1 = Q_t_1.reshape(self.batch_size, 2, -1)
            next_act_bs, next_act_uav = torch.argmax(Q_t_1_bs, 1), torch.argmax(Q_t_1_uav, 1)
            
            
            #next_actions = torch.cat([next_act1,next_act2], 1)
            next_state_action_value_bs = Q_t_1_bs.gather(2, next_act_bs[:,None])
            next_state_action_value_uav = Q_t_1_uav.gather(2, next_act_uav[:,None])
            
            #next_state_action_value = Q_t_1[:,:64].gather(1,next_act1[None]) + Q_t_1[:,64:].gather(1,next_act2[None]) # shape = [batch size, 6]
            
            #next_state_action_value = next_state_action_value_bs.sum(-1) + next_state_action_value_uav.sum(-1)
            next_state_action_value = torch.cat([next_state_action_value_bs.squeeze(), next_state_action_value_uav.squeeze()], 1)
            #ind = torch.where(done == True)[0]
            #next_state_action_value[ind] = reward[ind]
           
            expected_state_action_value = reward + self.gamma*next_state_action_value        
            
            self.model.train()
            loss = F.smooth_l1_loss(state_action_value.squeeze(), expected_state_action_value.squeeze(), reduction = 'sum')/self.batch_size
            loss_avg += loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self. optimizer.step()
            
        loss_avg = loss_avg/(self.n_train_sessions)
        self.loss = loss_avg
        
    def decide_action(self, state, n_episode=1, rand = True):
        self.epsilon = max(self.eps_min, self.eps_max - (n_episode /3000))
        #self.register_buffer('epsilon', torch.ones(1) * self.epsilon)
        
        if random.uniform(0,1) <= self.epsilon:
            bs_beam_ind = torch.randperm(64)[0]
            uav_beam_ind = torch.randperm(16)[0]
            bs_bin_list = torch.tensor(np.array(list(bin(bs_beam_ind)[2:]), dtype =int ))
            uav_bin_list = torch.tensor(np.array(list(bin(uav_beam_ind)[2:]), dtype =int ))
            if len(bs_bin_list) < 6:
                bs_bin_list = torch.cat([torch.zeros(6-len(bs_bin_list)), bs_bin_list])
           
            if len(uav_bin_list)<4:
                uav_bin_list = torch.cat([torch.zeros(4-len(uav_bin_list)), uav_bin_list])
                
            action = torch.cat([bs_bin_list, uav_bin_list]).to(self.device)
            #print('action', len(action), len(bs_bin_list), len(uav_bin_list))
        else:
            with torch.no_grad():
                self.model.eval()
                Q_t = self.model(state) # (18,)
                Q_t = Q_t.reshape(2,-1)
                next_act1, next_act2 = torch.argmax(Q_t[:,:6], 0), torch.argmax(Q_t[:,6:],0)
                action =  torch.cat([next_act1, next_act2])
            #print('action  2', len(action))
        return action
    
    #def get_state_next(self, state, action):
        
        
    def update_model(self):
        
       self.target_model.load_state_dict(self.model.state_dict())
       
    def soft_update_model(self, tau= 0.5):
        
        for param_target, param in zip(self.target_model.parameters(), self.model.parameters()):
            param_target.data.copy_(param_target.data * (1-tau) + param.data *tau)
            
    def memorize(self, state, action, reward, state_next, done=False):


        # compute bellman error
        self.model.eval()
        self.target_model.eval()
        
        # get Q_t (s_t, a_t) from s_t
       
        Q_t = self.model(state) # batch size * number of actions 
        #  = Q_t (s_t, a_t) 
        Q_t = torch.squeeze(Q_t)
        #print(Q_t.shape)
        Q_t_bs = Q_t[:12].reshape(2, -1)
        Q_t_uav = Q_t[12:].reshape(2, -1)
        
        #print(Q_t_bs.shape, action.shape)
        action = torch.tensor(action, dtype = torch.int64)
        state_action_value_bs = Q_t_bs.gather(1, action[:6][None])
        state_action_value_uav = Q_t_uav.gather(1, action[6:][None])
        
        state_action_value = torch.cat([state_action_value_bs.squeeze() , state_action_value_uav.squeeze()])
        
        # get Q_t(s_t+1, a_t) from s_t+1
        Q_t_1 = self.target_model(state_next) # [batch_size, 80]
        Q_t_1 = torch.squeeze(Q_t_1)
        Q_t_1_bs = Q_t_1[:12].reshape(2,-1)
        Q_t_1_uav = Q_t_1[12:].reshape(2,-1)
       
        #Q_t_1 = Q_t_1.reshape(self.batch_size, 2, -1)
        next_act_bs, next_act_uav = torch.argmax(Q_t_1_bs, 0), torch.argmax(Q_t_1_uav, 0)
        
      
        #next_actions = torch.cat([next_act1,next_act2], 1)
        next_state_action_value_bs = Q_t_1_bs.gather(1, next_act_bs[None])
        next_state_action_value_uav = Q_t_1_uav.gather(1, next_act_uav[None])
      
        #next_state_action_value = Q_t_1[:,:64].gather(1,next_act1[None]) + Q_t_1[:,64:].gather(1,next_act2[None]) # shape = [batch size, 6]
        next_state_action_value = torch.cat([next_state_action_value_bs.squeeze(), next_state_action_value_uav.squeeze()])        #ind = torch.where(done == True)[0]
        #next_state_action_value[ind] = reward[ind]
        expected_state_action_value = torch.tensor(reward).to(self.device) + torch.tensor(self.gamma).to(self.device)*next_state_action_value

        bellman_error = F.l1_loss(state_action_value, expected_state_action_value,reduction = 'sum')
        
        self.memory.push(state, action, reward, state_next, bellman_error, done)
        
