# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:09:32 2023

@author: seongjoon kang
"""
import torch.nn as nn
import torch


class Actor(nn.Module):

    def __init__(self, n_state=20, n_action=2):
        super(Actor, self).__init__()

        self.hidden_size = 128
        self.num_layers = 2
        self.input_size = n_state

        #self.fc = nn.Sequential(
        #    nn.Linear(n_state, 400),
        #    nn.ReLU(),
        #    nn.Linear(400, 200),
        #    nn.ReLU(),
        #    nn.Linear(200, 12),
        #    nn.Tanh()
        #)

        self.mlp0 = nn.Sequential(
            nn.Linear(self.hidden_size, n_action),
            #nn.ReLU(),
            #nn.Linear(2*self.hidden_size, n_action),
            nn.Sigmoid()
        )

        self.mlp = nn.LSTM(self.input_size,self.hidden_size, self.num_layers,
                           batch_first = True, bidirectional = False)

    def forward(self, state):
        if state.dim() ==1:
            state = state[None][None]
        elif state.dim() == 2:
            state = state[:,None]
        #h0 = torch.zeros((self.num_layers*2 , state.size(0), self.hidden_size)) #
        h0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size)) #
        c0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size))

        output, _ = self.mlp(state, (h0, c0))
        output = self.mlp0(output[:,-1,:]).squeeze()

        #output = self.sigmoid(output).squeeze()

        #if state.dim() != 2:
        #    state = state[None]
        #output = self.fc(state)

        #print(output.shape)
        # output1 = torch.argmax(output[:, :64], 1, keepdim=True)
        # output2 = torch.argmax(output[:, 64:], 1, keepdim=True)
        # output = torch.cat([output1, output2], 1).type(torch.float32)


        return output.squeeze()


class Critic(nn.Module):

    def __init__(self, n_state=20, n_action=2):
        super(Critic, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
        )
        self.hidden_size = 128
        self.num_layers = 1
        self.lstm = nn.LSTM(n_state, self.hidden_size, self.num_layers,
                           batch_first = True, bidirectional = False)

        self.action_encoder = nn.Sequential(
            nn.Linear(n_action, 128),
            nn.ReLU(),
        )

        self.q_estimator_1 = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
            )
        self.q_estimator_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, state, action):
        #print(action.shape)
        if action.dim() ==1:
            action = action[None]
        if action.dim() ==0:
            action = torch.tensor(action)[None][None]

        if state.dim() ==1:
            state = state[None][None]
        #    state = state[None]
        elif state.dim() == 2:
            state = state[:,None]
        #print(action.shape, state.shape)
        h0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size))  #
        c0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size))
        state_emb, _ = self.lstm(state, (h0, c0))
        state_emb = state_emb[:,-1,:]

        #print(state_emb, self.action_encoder(action).shape)
        emb = torch.cat([state_emb, self.action_encoder(action)], dim=-1)

        #emb = torch.cat([self.state_encoder(state), self.action_encoder(action)], dim=-1)
        output1 = self.q_estimator_1(emb)
        output2 = self.q_estimator_2(emb)

        return output1, output2

    def actor_Q(self, state, action):
        if action.dim() == 1:
            action = action[:,None]

        if state.dim() ==1:
            state = state[None][None]
            #state = state[None]
        elif state.dim() == 2:
            state = state[:,None]
        h0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size))  #
        c0 = torch.zeros((self.num_layers, state.size(0), self.hidden_size))
        state_emb, _ = self.lstm(state, (h0, c0))
        state_emb = state_emb[:,-1,:]
        emb = torch.cat([state_emb, self.action_encoder(action)], dim=-1)
        #emb = torch.cat([self.state_encoder(state), self.action_encoder(action)], dim=-1)
        output = self.q_estimator_1(emb)
        #output, _ = self.mlp1(emb[:,None])
        #output = self.fc1(output)
        return output
