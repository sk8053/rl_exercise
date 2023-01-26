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
#from memory import memory
from prioritized_memory import memory
import torch
import copy

from q_net import Q_Net, Noisy_Q_Net

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
        
        self.model = Noisy_Q_Net(n_states, 128, n_actions)
        self.target_model = Noisy_Q_Net(n_states, 128, n_actions)
        self._synchronize_models(self.model, self.target_model)

        self.model.to(device)
        self.target_model.to(device)


        print("Num params: ", sum(p.numel() for p in self.model.parameters()))
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)

        self.n_train_sessions = n_train
        self.alpha = tau
        self.epsilon = 0.7
        self.loss = torch.tensor(-1.0)
        self.criteria = F.mse_loss

    def eval(self):
        self.model.eval()
        self.target_model.eval()

    @staticmethod
    def _synchronize_models(model_A: nn.Module, model_B: nn.Module):
        _= model_A.load_state_dict(model_B.state_dict())
        

           
    def train(self, hard_update = False):
        
        if len(self.memory) < self.batch_size:
            return torch.tensor(-1.0)
        loss_avg = 0.0
            
        for s in range(self.n_train_sessions):
            '''
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

            self.target_model.eval()
            
            # get Q_t (s_t, a_t) from s_t
            
            Q_t_bs1, Q_t_bs2 = self.model(state) # batch size * number of actions

    
            state_action_value_bs1, state_action_value_bs2 = Q_t_bs1.gather(1, action[:,0][None]), Q_t_bs2.gather(1, action[:,1][None])
            #state_action_value_uav = Q_t_uav.gather(1, action[:,1:])
            state_action_value = torch.cat([state_action_value_bs1 , state_action_value_bs2],1)
            

            Q_t_1_bs1, Q_t_1_bs2  = self.model(state_next) # [batch_size, 80]
            #next_act_bs, next_act_uav = torch.argmax(Q_t_1_bs, -1), torch.argmax(Q_t_1_uav, -1)
            next_act_bs1 , next_act_bs2 = torch.argmax(Q_t_1_bs1, -1), torch.argmax(Q_t_1_bs2, -1)


            Q_t_1_target_bs1, Q_t_1_target_bs2 = self.target_model(state_next) # [batch_size, 80]
            next_state_action_value_target_bs1 = Q_t_1_target_bs1.gather(1, next_act_bs1[None])
            next_state_action_value_target_bs2 =  Q_t_1_target_bs2.gather(1, next_act_bs2[None])
            #next_state_action_value_target_uav = Q_t_1_target_uav.gather(1, next_act_uav[:,None])

            next_state_action_value_target = torch.cat([next_state_action_value_target_bs1, next_state_action_value_target_bs2],1)

            expected_state_action_value = reward + self.gamma*next_state_action_value_target        
            
            self.model.train()
            loss = self.criteria(state_action_value.squeeze(), expected_state_action_value.squeeze()).sum()
            loss_avg += loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.reset_noise()
            self.target_model.reset_noise()
        if hard_update is True:
            self._soft_update_models(self.target_model, self.model, 1)
        else:
            self._soft_update_models(self.target_model, self.model, self.alpha)

        loss_avg = loss_avg/(self.n_train_sessions)
        self.loss = loss_avg
            
    def get_action(self, state:torch.tensor, n_episode:int =1, random:bool = True) -> torch.tensor:
        self.epsilon = max(self.eps_min, self.eps_max*(1 - (n_episode /5000)))
        
        if random is True and random.uniform(0,1) <= self.epsilon:
            bs_beam_ind1 = torch.randperm(8)[0]
            bs_beam_ind2 = torch.randperm(8)[0]
            #uav_beam_ind = torch.randperm(16)[0]
            #action = torch.tensor([bs_beam_ind, uav_beam_ind]).to(self.device)
            action = torch.tensor([bs_beam_ind1, bs_beam_ind2])
        else:
            with torch.no_grad():
                self.model.eval()
                Q_t_bs1, Q_t_bs2 = self.model(state)  # , Q_t_uav
                #next_act1, next_act2 = torch.argmax(Q_t_bs, -1), torch.argmax(Q_t_uav,-1)
                next_act1 = torch.argmax(Q_t_bs1, -1)
                next_act2 = torch.argmax(Q_t_bs2, -1)
                action =  torch.tensor([next_act1, next_act2])
         
        return action
    
 
            
    def memorize(self, state:torch.tensor, action:torch.tensor, 
                 reward:float, state_next:torch.tensor, done:bool = False):

        Q_t_bs1, Q_t_bs2 = self.model(state)
        Q_t_bs1, Q_t_bs2 = Q_t_bs1.squeeze(), Q_t_bs2.squeeze()
        state_action_value_bs1, state_action_value_bs2 = Q_t_bs1[action[0]], Q_t_bs2[action[1]]
        # state_action_value_uav = Q_t_uav.gather(1, action[:,1:])
        state_action_value = torch.tensor([state_action_value_bs1, state_action_value_bs2])

        Q_t_1_bs1, Q_t_1_bs2 = self.model(state_next)  # [batch_size, 80]
        Q_t_1_bs1, Q_t_1_bs2 = Q_t_1_bs1.squeeze(), Q_t_1_bs2.squeeze()
        # next_act_bs, next_act_uav = torch.argmax(Q_t_1_bs, -1), torch.argmax(Q_t_1_uav, -1)
        next_act_bs1, next_act_bs2 = torch.argmax(Q_t_1_bs1), torch.argmax(Q_t_1_bs2)

        Q_t_1_target_bs1, Q_t_1_target_bs2 = self.target_model(state_next)  # [batch_size, 80]
        Q_t_1_target_bs1, Q_t_1_target_bs2 = Q_t_1_target_bs1.squeeze(), Q_t_1_target_bs2.squeeze()
        next_state_action_value_target_bs1 = Q_t_1_target_bs1[next_act_bs1]
        next_state_action_value_target_bs2 = Q_t_1_target_bs2[next_act_bs2]
        # next_state_action_value_target_uav = Q_t_1_target_uav.gather(1, next_act_uav[:,None])

        next_state_action_value_target = torch.tensor([next_state_action_value_target_bs1, next_state_action_value_target_bs2])

        expected_state_action_value = reward + self.gamma * next_state_action_value_target


        bellman_error = self.criteria(state_action_value, expected_state_action_value).sum()
        
        #bellman_error = 0
        #self.memory.push(state, action, reward, state_next, bellman_error, done)

        self.memory.add(bellman_error.detach().numpy(), (state.detach().numpy(), action.detach().numpy(), reward.detach().numpy(), state_next.detach().numpy(), False))
    def save(self, file_path:str):
        
        check_point={ 'model':self.model.state_dict(),
                     'optimizer':self.optimizer.state_dict(),

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