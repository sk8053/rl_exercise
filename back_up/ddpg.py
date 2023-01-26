# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:08:58 2023

@author: seongjoon kang
the below code is reference by https://github.com/sfujim/TD3/blob/master/TD3.py
"""
import torch.nn as nn
import torch
from memory import memory
import numpy as np
from actor_critic import Actor, Critic
import copy

class DDPG(nn.Module):

    def __init__(self,
                 n_state=20,
                 n_action=2,
                 lr_critic: float = 0.005,
                 lr_actor: float = 0.001,
                 gamma: float = 0.9,
                 memory_size=3000,
                 batch_size=128,
                 tau=0.005,
                 n_train=50,
                 device='cpu',
                 actor_update_period = 3,
                 noise_std = 1.0,
                 policy_noise = 0.1,
                 noise_clip = 0.2):

        super(DDPG, self).__init__()

        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.tau = tau
        self.n_train = n_train
        self.memory = memory(capacity=memory_size)
        # setup optimizers

        self.critic = Critic(n_state, n_action)
        self.critic_target = Critic(n_state, n_action)
        #self.critic_B = Critic(n_state, n_action)
        #self.critic_B_target = Critic(n_state, n_action)

        print("Num params of critic: ", sum(p.numel() for p in self.critic.parameters()))
        self._synchronize_models(self.critic, self.critic_target)
        #self._synchronize_models(self.critic_B, self.critic_B_target)

        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(),
                                             lr=lr_critic)


        self.actor = Actor(n_state)
        print("Num params of actor: ", sum(p.numel() for p in self.actor.parameters()))

        self.actor_target = Actor(n_state)
        self._synchronize_models(self.actor, self.actor_target)
        self.actor_opt = torch.optim.Adam(params=self.actor.parameters(),
                                          lr=lr_actor)

        # setup target networks

        self.criteria = nn.SmoothL1Loss()
        self.epsilon_records = []
        self.loss = torch.tensor(-1)
        self.actor_update_period = actor_update_period
        self.noise_std = noise_std
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def get_action(self, state):
        with torch.no_grad():
            a = self.actor(state)
        return a

    def train(self, iter_=0):

        if len(self.memory) < self.batch_size:
            return -1.0
        loss_avg = 0.0
        if torch.rand(1) > 0.5:
            model_name = 'A'
        else:
            model_name = 'B'
        for s in range(self.n_train):
            past_data = self.memory.sample(self.batch_size)

            state = torch.zeros((self.batch_size, len(past_data[0]['state']))).to(self.device)
            state_next = torch.zeros((self.batch_size, len(past_data[0]['state_next']))).to(self.device)
            action = torch.zeros((self.batch_size, len(past_data[0]['action'])), dtype=torch.float32).to(self.device)
            reward = torch.zeros((self.batch_size, len(past_data[0]['reward']))).to(self.device)

            for i in range(self.batch_size):
                state[i] = past_data[i]['state']
                state_next[i] = past_data[i]['state_next']

                action[i] = past_data[i]['action']
                reward[i] = past_data[i]['reward']

            loss = self.update(state, action, reward, state_next, model_name, s)
            loss_avg += loss

            if s % self.actor_update_period == 0:
                self._soft_update_models(self.actor, self.actor_target, self.tau)
                self._soft_update_models(self.critic, self.critic_target, self.tau)



        loss_avg = loss_avg / (self.n_train)
        self.loss = loss_avg

    def update(self, state, action, reward, state_next, model_name='A', iter_=0):
        s, a, r, ns = state, action, reward, state_next
        noise = torch.normal(mean=0.0, std=torch.tensor(self.noise_std), size=a.shape) * self.policy_noise
        noise = noise.clamp(self.noise_clip, self.noise_clip).to(self.device)

        self.critic_target.eval()

        # compute critic loss and update the critic parameters
        with torch.no_grad():
            target_A, target_B = self.critic_target(ns, self.actor_target(ns) + noise)
            target = torch.minimum(target_A, target_B)
            critic_target = r + self.gamma * target
        cur_Q_A, cur_Q_B = self.critic(s, a)
        critic_loss = self.criteria(cur_Q_A, critic_target) + self.criteria(cur_Q_B, critic_target)

        self.critic.train()
        self.actor.train()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if iter_ % self.actor_update_period == 0:
            # compute actor loss and update the actor parameters
            actor_loss = -self.critic.actor_Q(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        else:
            actor_loss = 0

        return critic_loss + actor_loss

    def memorize(self, state: torch.tensor, action: torch.tensor,
                 reward: float, state_next: torch.tensor, done: bool = False):

        bellman_error = 0  # F.l1_loss(state_action_value.squeeze(), expected_state_action_value.squeeze()).sum()

        self.memory.push(state, action, reward, state_next, bellman_error, done)

    def save(self, file_path: str):

        check_point = {'actor': self.actor.state_dict(),
                       'actor_opt': self.actor_opt.state_dict(),
                       'critic': self.critic.state_dict(),
                       'critic_opt': self.critic_opt.state_dict(),

                       'agent_hyper_params': {
                           'batch_size': self.batch_size,
                           'lr_cr': self.lr_critic,
                            'lr_ac':self.lr_actor,
                            'memory_size': len(self.memory),
                            'tau':self.tau,
                            'n_train':self.n_train,
                            'noise_std':self.noise_std,
                            'policy_noise':self.policy_noise,
                           'noise_clip':self.noise_clip
                       }
                       }
        torch.save(check_point, file_path)

    def load(self, file_name):

        check_points = torch.load(file_name)
        self.critic.load_state_dict(check_points['critic'])
        self.critic_opt.load_state_dict(check_points['critic_opt'])
        self.critic_target  = copy.deepcopy(self.critic)

        self.actor.load_state_dict(check_points['actor'])
        self.actor_opt.load_state_dict(check_points['actor_opt'])
        self.actor_target = copy.deepcopy(self.actor)

        self.batch_size = check_points['agent_hyper_params']['batch_size']
        self.lr_critic = check_points['agent_hyper_params']['lr_cr']
        self.lr_actor = check_points['agent_hyper_params']['lr_ac']
        self.memory.capacity = check_points['agent_hyper_params']['memory_size']
        self.tau = check_points['agent_hyper_params']['tau']
        self.n_train = check_points['agent_hyper_params']['n_train']

        self.noise_std =check_points['agent_hyper_params']['noise_std']
        self.noise_clip =check_points['agent_hyper_params']['noise_clip']
        self.policy_noise = check_points['agent_hyper_params']['policy_noise']

    def reset(self):
        self.memory.reset()

    @staticmethod
    def _soft_update_models(model: nn.Module, model_target: nn.Module, alpha: float):
        for p_target, p in zip(model_target.parameters(), model.parameters()):
            p_target.data.copy_(alpha * p.data + (1 - alpha) * p_target)

    @staticmethod
    def _synchronize_models(model_A: nn.Module, model_B: nn.Module):
        _ = model_A.load_state_dict(model_B.state_dict())


class OrnsteinUhlenbeckProcess:
    """
    OU process; The original implementation is provided by minimalRL.
    https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    """

    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x