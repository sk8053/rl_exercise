# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:40:39 2023

@author: seongjoon kang
"""
import random
import torch
import numpy as np
class memory():
    
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.memory = []
        self.bellman_error_memory = []
        self.index = 0
    
    def push(self, state:torch.tensor, action:torch.tensor, reward:float,
             state_next:torch.tensor, bellman_error:float = 0, done=False):
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.bellman_error_memory.append(None)
        
        self.memory[self.index] = {'state':state, 'action':action,
                                   'state_next':state_next, 'reward':reward}
        self.bellman_error_memory[self.index] = bellman_error
       
        self.index +=1
        self.index = self.index % self.capacity
        
    def sample(self, batch_size:int):
        return random.sample(self.memory, batch_size)
        #I = torch.multinomial(torch.tensor(self.bellman_error_memory), batch_size, replacement=True)
        #return np.array(self.memory)[I]
    def reset(self):
        self.memory = []
        self.bellman_error_memory = []
        self.index = 0

    def __len__(self):
        return len(self.memory)
    
    