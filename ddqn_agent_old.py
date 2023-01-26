# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:08:12 2023

@author:seongjoon kang
"""

import numpy as np
import random
from torch import nn
from torch import optim
import torch.nn.functional as F
#from collections import deque
from memory import memory
#from prioritized_memory import memory
import torch
import copy

from q_net import Q_Net

class DDQN_Agent():
    
    def __init__(self, n_states = 8, n_actions = 80, memory_size = 10000, gamma = 0.9,
                 eps_min = 0.01, eps_max = 0.1, batch_size = 80, n_train = 200, 
                 device = None, lr = 1e-4, tau = 5e-3):
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = memory (capacity = memory_size)
        self.device = device
        
        self.eps_min = eps_min
        self.eps_max = eps_max
        
        self.model_A = Q_Net(n_states, 128, n_actions)
        self.model_A_target = Q_Net(n_states, 128, n_actions)
        self.model_B = Q_Net(n_states, 128, n_actions)
        self.model_B_target = Q_Net(n_states, 128, n_actions)

        self._synchronize_models(self.model_A, self.model_A_target)
        self._synchronize_models(self.model_B, self.model_B_target)

        self.model_A.to(device)
        self.model_A_target.to(device)
        self.model_B.to(device)
        self.model_B_target.to(device)

        print("Num params: ", sum(p.numel() for p in self.model_A.parameters()))
        
        self.optimizer_A = optim.Adam(self.model_A.parameters(), lr = lr)
        self.optimizer_B = optim.Adam(self.model_B.parameters(), lr = lr)
        
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr = 1e-3)
        

        self.n_train_sessions = n_train
        self.alpha = tau
        self.epsilon = 0.7
        self.loss = torch.tensor(-1.0)
        
    @staticmethod
    def _synchronize_models(model_A: nn.Module, model_B: nn.Module):
        _= model_A.load_state_dict(model_B.state_dict())
        

           
    def train(self, model_name:str = 'A', update_enable = False):
        
        if len(self.memory) < self.batch_size:
            return torch.tensor(-1.0)
        loss_avg = 0.0
        
        if model_name == 'A':    
            model = copy.deepcopy(self.model_A)
            target_model = copy.deepcopy(self.model_B_target)
            optimizer = copy.deepcopy(self.optimizer_A)
        else:
            model = copy.deepcopy(self.model_B)
            target_model = copy.deepcopy(self.model_A_target)
            optimizer = copy.deepcopy(self.optimizer_B)
            
        for s in range(self.n_train_sessions):

            past_data = self.memory.sample(self.batch_size)
            state = torch.zeros((self.batch_size, len(past_data[0]['state']))).to(self.device)
            state_next = torch.zeros((self.batch_size, len(past_data[0]['state_next']))).to(self.device)
            action = torch.zeros((self.batch_size, len(past_data[0]['action'])),dtype = torch.int64).to(self.device)
            reward = torch.zeros((self.batch_size, len(past_data[0]['reward']))).to(self.device)

            for i in range(self.batch_size):
                state[i] = past_data[i]['state']
                state_next[i] = past_data[i]['state_next']

                action[i] = past_data[i]['action']
                reward[i] = past_data[i]['reward']
            '''
            data, idxs, is_weights = self.memory.sample(self.batch_size)

            state = torch.zeros((self.batch_size, len(data[0][0]))).to(self.device)
            action = torch.zeros((self.batch_size, len(data[0][1])), dtype = torch.int64).to(self.device)
            reward = torch.zeros((self.batch_size, len(data[0][2]))).to(self.device)
            state_next = torch.zeros((self.batch_size, len(data[0][3]))).to(self.device)

            for i in range(self.batch_size):
                
                state[i] = torch.tensor(data[i][0])
                action[i] = torch.tensor(data[i][1])
                reward[i] =  torch.tensor(data[i][2])
                state_next[i] = torch.tensor(data[i][3])
                #done= torch.cat([done, past_data[i]['done']],0)
            '''
            model.eval()
            target_model.eval()
            
            # get Q_t (s_t, a_t) from s_t
            
            Q_t = model(state) # batch size * number of actions 

            
            Q_t_bs = Q_t[:,:64]
            Q_t_uav = Q_t[:,64:]
            
    
            state_action_value_bs = Q_t_bs.gather(1, action[:,:1])
            state_action_value_uav = Q_t_uav.gather(1, action[:,1:])
            state_action_value = torch.cat([state_action_value_bs , state_action_value_uav],1)
            

            Q_t_1 = model(state_next) # [batch_size, 80]
            Q_t_1_bs = Q_t_1[:,:64]
            Q_t_1_uav = Q_t_1[:,64:]
            next_act_bs, next_act_uav = torch.argmax(Q_t_1_bs, -1), torch.argmax(Q_t_1_uav, -1)


            Q_t_1_target = target_model(state_next) # [batch_size, 80]
            Q_t_1_target_bs = Q_t_1_target[:,:64]
            Q_t_1_target_uav = Q_t_1_target[:,64:]
            next_state_action_value_target_bs = Q_t_1_target_bs.gather(1, next_act_bs[:,None])
            next_state_action_value_target_uav = Q_t_1_target_uav.gather(1, next_act_uav[:,None])

            next_state_action_value_target = torch.cat([next_state_action_value_target_bs, next_state_action_value_target_uav],1)


            expected_state_action_value = reward + self.gamma*next_state_action_value_target        
            
            model.train()

            loss = torch.nn.MSELoss()(state_action_value.squeeze(), expected_state_action_value.squeeze()).sum()
            
            loss_avg += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if update_enable == True:
            if model_name == 'A':
                self._soft_update_models(self.model_A_target, self.model_A, self.alpha)
            else:
                self._soft_update_models(self.model_B_target, self.model_B, self.alpha)

        loss_avg = loss_avg/(self.n_train_sessions)
        self.loss = loss_avg
        
        if model_name == 'A':    
            self.model_A = copy.deepcopy(model)
            self.model_B = copy.deepcopy(target_model)
            self.optimizer_A = copy.deepcopy(optimizer)
        else:
            self.model_B = copy.deepcopy(model)
            self.model_A = copy.deepcopy(target_model)
            self.optimizer_B = copy.deepcopy(optimizer)
            
    def decide_action(self, state:torch.tensor, n_episode:int =1) -> torch.tensor:
        
        self.epsilon = max(self.eps_min, self.eps_max - (n_episode /2100))
        
        if random.uniform(0,1) <= self.epsilon:
            bs_beam_ind = torch.randperm(64)[0]
            uav_beam_ind = torch.randperm(16)[0]
            action = torch.tensor([bs_beam_ind, uav_beam_ind]).to(self.device)
           
        else:
            with torch.no_grad():
                self.model_A.eval()
                self.model_B.eval()
                Q_t = self.model_A(state) + self.model_B(state)
                next_act1, next_act2 = torch.argmax(Q_t[:,:64], -1), torch.argmax(Q_t[:,64:],-1)
                action =  torch.cat([next_act1, next_act2])
         
        return action
    
 
            
    def memorize(self, state:torch.tensor, action:torch.tensor, 
                 reward:float, state_next:torch.tensor, done:bool = False):
        
        
        # compute bellman error
        #self.model_A.eval()
        #self.model_B.eval()
        
        # get Q_t (s_t, a_t) from s_t
       
        #Q_t = self.model_A(state) + self.model_B(state)  # batch size * number of actions
        #  = Q_t (s_t, a_t) 
        #Q_t = torch.squeeze(Q_t)
        #Q_t_bs = Q_t[:64]
        #Q_t_uav = Q_t[64:]
        
        #print(Q_t_bs.shape, action.shape)
        #action = torch.tensor(action, dtype = torch.int64)

        #state_action_value_bs = Q_t_bs.gather(0, action[:1])
        #state_action_value_uav = Q_t_uav.gather(0, action[1:])

        #state_action_value = torch.cat([state_action_value_bs, state_action_value_uav])
        
        # get Q_t(s_t+1, a_t) from s_t+1
        #Q_t_1 = self.model_A(state_next) + self.model_B(state_next)# [batch_size, 80]
        #Q_t_1 = torch.squeeze(Q_t_1)
        #Q_t_1_bs = Q_t_1[:64]#.reshape(2,-1)
        #Q_t_1_uav = Q_t_1[64:]#.reshape(2, -1)
        
        #next_act_bs, next_act_uav = torch.argmax(Q_t_1_bs, -1), torch.argmax(Q_t_1_uav, -1)
        
        #Q_t_1_bs = Q_t_1[:64].reshape(-1)
        #Q_t_1_uav = Q_t_1[64:].reshape(-1)


        #next_state_action_value_bs = Q_t_1_bs.gather(0, next_act_bs)
        #next_state_action_value_uav = Q_t_1_uav.gather(0, next_act_uav)
        
        #next_state_action_value = torch.tensor([next_state_action_value_bs, next_state_action_value_uav])        #ind = torch.where(done == True)[0]

        
        #expected_state_action_value = torch.tensor(reward).to(self.device) + torch.tensor(self.gamma).to(self.device)*next_state_action_value
        
        #bellman_error = F.mse_loss(state_action_value, expected_state_action_value).sum()
        bellman_error = 0
        self.memory.push(state, action, reward, state_next, bellman_error, done)

        #self.memory.add(bellman_error.detach().numpy(), (state.detach().numpy(), action.detach().numpy(), reward.detach().numpy(), state_next.detach().numpy(), False))
    def save(self, file_path:str):
        
        check_point={ 'model_A':self.model_A.state_dict(),
                     'optimizer_A':self.optimizer_A.state_dict(),
                     'model_B':self.model_A.state_dict(),
                     'optimizer_A':self.optimizer_A.state_dict(),
                     
                     'agent_hyper_params':{
                     'alpha':self.alpha,
                     'batch_size':self.batch_size,
                     'gamma':self.gamma,
                     'memory_size':len(self.memory)

                     }   
            }
        torch.save(check_point, file_path)
        
    @staticmethod
    def _soft_update_models(model_A: nn.Module, model_B: nn.Module, alpha:float):    
        for p1, p2 in zip(model_A.parameters(), model_B.parameters()):
           p1.data.copy_(alpha * p2.data + (1 - alpha) * p1.data)