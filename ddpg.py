# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:08:58 2023

@author: seongjoon kang
the below code is reference by https://github.com/sfujim/TD3/blob/master/TD3.py
"""
import torch.nn as nn
import torch
#from memory import memory
from prioritized_memory import memory
import numpy as np
from actor_critic import Actor, Critic
import copy

class DDPG(nn.Module):

    def __init__(self,
                 n_state=20,
                 n_action=2,
                 lr_critic: float = 0.001,
                 lr_actor: float = 0.001,
                 gamma: float = 0.9,
                 memory_size=5000,
                 batch_size=128,
                 tau=0.005,
                 n_train=50,
                 device='cpu',
                 actor_update_period = 3,
                 noise_std = 1.0,
                 policy_noise = 0.1,
                 noise_clip = 0.2,
                 loading = False):

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
        self.actor = Actor(n_state, n_action)

        self._init_weights(self.actor)
        self._init_weights(self.critic)

        self.critic_target = Critic(n_state, n_action)
        self.actor_target = Actor(n_state, n_action)
        #self.critic_B = Critic(n_state, n_action)
        #self.critic_B_target = Critic(n_state, n_action)

        print("Num params of critic: ", sum(p.numel() for p in self.critic.parameters()))
        #self._synchronize_models(self.critic_B, self.critic_B_target)

        self.critic_opt = torch.optim.Adam(params=self.critic.parameters(),
                                             lr=lr_critic)



        print("Num params of actor: ", sum(p.numel() for p in self.actor.parameters()))

        self.actor_target = Actor(n_state, n_action)
        self.actor_opt = torch.optim.Adam(params=self.actor.parameters(),
                                          lr=lr_actor)
        if loading is True:
            self.load('checkpoints/TD3_1.pt')
        self._synchronize_models(self.actor, self.actor_target)
        self._synchronize_models(self.critic, self.critic_target)
        # setup target networks

        self.criteria =  nn.SmoothL1Loss()
        self.epsilon_records = []
        self.cr_loss = torch.tensor(-1)
        self.ac_loss = torch.tensor(-1)
        self.actor_update_period = actor_update_period
        self.noise_std = noise_std
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def eval(self):
        self.critic.eval()
        self.actor.eval()
        self.critic_target.eval()
        self.actor_target.eval()

    def get_action(self, state, random = False):
        with torch.no_grad():
            action = self.actor(state)
        return action

    def train(self, iter_=0):
        if len(self.memory) < self.batch_size:
            return -1.0
        cr_loss_avg = 0.0
        ac_loss_avg = 0.0
        for s in range(self.n_train):

            #past_data = self.memory.sample(self.batch_size)

            #state = torch.zeros((self.batch_size, len(past_data[0]['state']))).to(self.device)
            #state_next = torch.zeros((self.batch_size, len(past_data[0]['state_next']))).to(self.device)
            #action = torch.zeros((self.batch_size, len(past_data[0]['action'])), dtype=torch.float32).to(self.device)
            #reward = torch.zeros((self.batch_size, len(past_data[0]['reward']))).to(self.device)

            #for i in range(self.batch_size):
            #    state[i] = past_data[i]['state']
            #    state_next[i] = past_data[i]['state_next']

                #action[i] = past_data[i]['action']
                #reward[i] = past_data[i]['reward']

            # past_data = self.memory.sample(self.batch_size)
            data, idxs, is_weights = self.memory.sample(self.batch_size)

            state = torch.zeros((self.batch_size, len(data[0][0])) ).to(self.device)
            action = torch.zeros((self.batch_size, len(data[0][1]))).to(self.device)
            reward = torch.zeros((self.batch_size, len(data[0][2]))).to(self.device)
            state_next = torch.zeros((self.batch_size, len(data[0][3]))).to(self.device)

            for i in range(self.batch_size):
                state[i] = torch.tensor(data[i][0], dtype=torch.float64)
                action[i] = torch.tensor(data[i][1], dtype=torch.float64)
                reward[i] = torch.tensor(data[i][2], dtype=torch.float64)
                state_next[i] = torch.tensor(data[i][3], dtype=torch.float64)

            cr_loss, ac_loss = self.update(state, action, reward, state_next,  s)
            cr_loss_avg += cr_loss
            ac_loss_avg += ac_loss

            if s % self.actor_update_period == 0:
                self._soft_update_models(self.actor, self.actor_target, self.tau)
                self._soft_update_models(self.critic, self.critic_target, self.tau)



        cr_loss_avg = cr_loss_avg / (self.n_train)
        ac_loss_avg = ac_loss_avg / (self.n_train)
        self.cr_loss = cr_loss_avg
        self.ac_loss = ac_loss_avg

    def update(self, state, action, reward, state_next,  iter_=0):
        s, a, r, ns = state, action, reward, state_next
        noise = torch.normal(mean=0.0, std=torch.tensor(self.noise_std), size=a.shape) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip).to(self.device)

        self.critic_target.eval()
        #x = self.actor_target(ns) + noise.squeeze()

        # compute critic loss and update the critic parameters
        with torch.no_grad():
            next_action = self.actor_target(ns) + noise.squeeze()

            bs_beam_idx1, bs_beam_idx2 = torch.argmax(next_action[:,:8],1), torch.argmax(next_action[:,8:],1)
            v1,bs_beam_idx1 = torch.max(next_action[:, 8:], 1)
            v2, bs_beam_idx2 = torch.max(next_action[:, 8:], 1)
            #print(v1)
            next_action_cp = torch.zeros_like(next_action)
            next_action_cp[:,:8][range(self.batch_size),bs_beam_idx1] = v1
            next_action_cp[:, 8:][range(self.batch_size), bs_beam_idx2] = v2

            next_action = next_action_cp
            #print(next_action[0])
            #next_action[:bs_beam_idx1], next_action[bs_beam_idx1 + 1:8] = 0, 0
            #next_action[8:bs_beam_idx2 + 8], next_action[bs_beam_idx2 + 1 + 8:] = 0, 0

            target_A, target_B = self.critic_target(ns, next_action)
            target = torch.min(target_A, target_B)
            critic_target = r + self.gamma * target
        cur_Q_A, cur_Q_B = self.critic(s, a)
        critic_loss = self.criteria(cur_Q_A, critic_target) + self.criteria(cur_Q_B, critic_target)

        self.critic.train()
        self.actor.train()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.8)
        self.critic_opt.step()

        if iter_ % self.actor_update_period == 0:
            # compute actor loss and update the actor parameters
            actor_loss = -self.critic.actor_Q(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.8)
            self.actor_opt.step()
        else:
            actor_loss = 0 #torch.tensor([0])
        #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # https://stackoverflow.com/questions/67236480/why-is-the-clip-grad-norm-function-used-here
        return critic_loss,  actor_loss

    def memorize(self, state: torch.tensor, action: torch.tensor,
                 reward: float, state_next: torch.tensor, done: bool = False):
        if action.dim() ==0:
            action = action[None]
        self.critic_target.eval()
        s, a = torch.tensor(state,dtype= torch.float32), torch.tensor(action, dtype = torch.float32)
        r, ns = torch.tensor(reward, dtype = torch.float32), torch.tensor(state_next, dtype = torch.float32)
        with torch.no_grad():
            target_A, target_B = self.critic_target(ns, self.actor_target(ns))
            target = torch.minimum(target_A, target_B)
            critic_target = r + self.gamma * target
        cur_Q_A, cur_Q_B = self.critic(s, a)
        critic_loss = self.criteria(cur_Q_A, critic_target) + self.criteria(cur_Q_B, critic_target)
        bellman_error = critic_loss  # F.l1_loss(state_action_value.squeeze(), expected_state_action_value.squeeze()).sum()

        #self.memory.push(state, action, reward, state_next, bellman_error, done)
        self.memory.add(bellman_error.detach().numpy(), (state.detach().numpy(), action.detach().numpy(), reward.detach().numpy(), state_next.detach().numpy(), False))

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