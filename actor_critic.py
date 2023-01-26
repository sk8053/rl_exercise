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

        self.fc = nn.Sequential(
            nn.Linear(n_state, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, n_action),
            nn.Sigmoid()
        )



    def forward(self, state):

        if state.dim() != 2:
            state = state[None]
        output = self.fc(state)

        return output.squeeze()


class Critic(nn.Module):

    def __init__(self, n_state=20, n_action=2):
        super(Critic, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
        )

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


        if state.dim() ==1:
            state = state[None]
        if action.dim() ==1:
            action = action[None]

        emb = torch.cat([self.state_encoder(state), self.action_encoder(action)], dim=-1)
        output1 = self.q_estimator_1(emb)
        output2 = self.q_estimator_2(emb)

        return output1, output2

    def actor_Q(self, state, action):
        if action.dim() == 1:
            action = action[None]
        if state.dim() ==1:
            state = state[None]

        emb = torch.cat([self.state_encoder(state), self.action_encoder(action)], dim=-1)
        output = self.q_estimator_1(emb)
        return output
