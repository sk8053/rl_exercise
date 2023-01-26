# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:09:32 2023

@author: gangs
"""
import torch.nn as nn
import torch


class Actor(nn.Module):

    def __init__(self, n_state=20):
        super(Actor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()


        )

    def forward(self, state):
        # Action space of Pendulum v0 is [-2.0, 2.0]
        output = self.mlp(state)
        if output.dim() != 2:
            output = output[None]

        # output1 = torch.argmax(output[:, :64], 1, keepdim=True)
        # output2 = torch.argmax(output[:, 64:], 1, keepdim=True)
        # output = torch.cat([output1, output2], 1).type(torch.float32)

        # output[:,0] = torch.tensor(64)*output[:,0]
        # output[:,1] = torch.tensor(16)*output[:,1]

        return output


class Critic(nn.Module):

    def __init__(self, n_state=20, n_action=2):
        super(Critic, self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(n_state, 48),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(n_action, 8),
            nn.ReLU()
        )
        self.q_estimator_1 = nn.Sequential(
            nn.Linear(56, 112),
            nn.ReLU(),
            nn.Linear(112,32),
            nn.ReLU(),
            nn.Linear(32, 1),
            )
        self.q_estimator_2 = nn.Sequential(
            nn.Linear(56, 112),
            nn.ReLU(),
            nn.Linear(112, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state, action):
        emb = torch.cat([self.state_encoder(state), self.action_encoder(action)], dim=-1)
        output1 = self.q_estimator_1(emb)
        output2 = self.q_estimator_2(emb)
        return output1, output2

    def actor_Q(self, state, action):
        emb = torch.cat([self.state_encoder(state), self.action_encoder(action)], dim=-1)
        output = self.q_estimator_1(emb)
        return output