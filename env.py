import gym
import gzip
import numpy
import torch
import numpy as np
import sys
sys.path.append("/home/nyu_wireless/Desktop/Boston/DQN/")


class Env(gym.Env):
    def __init__(self, env_config):
        self.file_list = env_config['file_list']
        self.n_prev_t = env_config['n_prev_t']
        self.file_counter = 0
        #self.observation_space = None
        self.action_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape=(16,))#gym.spaces.Discrete(64)
        self.observation_space = gym.spaces.Box(low = -1, high = 64, shape=[2*self.n_prev_t])
        #self.env_config = env_config
        self.reset()
    def get_init_state(self):
        return self.init_state

    def reset(self):
        #self.file_counter +=1
        file_path = self.file_list[0]
        self.file_list = self.file_list[1:]
        with gzip.open(file_path, 'rb') as f:
            trj_data = np.load(f)
            # clip SNR values having more than 30dB
            trj_data[trj_data > 30] = 30.0
            trj_data = trj_data / 10
            trj_data = np.transpose(trj_data, (1, 0, 2, 3))

        self.trj_data = trj_data
        self.state_set = np.argmax(trj_data, axis=1)
        self.best_snr_set = np.max(trj_data, axis=1)
        self.iter_max = len(trj_data)
        # set initial state
        # set initial state
        state = []
        b_ind_window = []
        bs_beam_ind_window = []
        for k in range(self.n_prev_t):
            v = np.max(trj_data[k, :, :, :])
            I = np.where(v == trj_data[k, :, :, :])

            if len(I[0]) > 1:
                b_ind = I[0][0]
                beam_ind_bs = I[1][0]
                beam_ind_uav = I[2][0]
            else:
                b_ind = I[0]
                beam_ind_bs = I[1]
                beam_ind_uav = I[2]

            #snr_k = trj_data[k, b_ind, beam_ind_bs, beam_ind_uav]
            #state.append(snr_k.item())
            #state.append((b_ind + 1) / 43)
            #state.append((beam_ind_bs + 1) / 64)
            #state.append((beam_ind_uav + 1) / 16)
            b_ind_window.append(b_ind)
            bs_beam_ind_window.append(beam_ind_bs)

        self.b_ind_window, self.bs_beam_ind_window = np.array(b_ind_window), np.array(bs_beam_ind_window)
        state = np.append(self.b_ind_window , self.bs_beam_ind_window )
        state = np.array(state, dtype=float)

        self.iter_max = len(trj_data)
        self.iter_ = self.n_prev_t
        self.state = np.array(state, dtype=float).squeeze()

        return self.state
    def step(self, action):

        bs_beam_idx1, bs_beam_idx2  = np.argmax(action[:8]), np.argmax(action[8:])
        #bs_beam_idx1, bs_beam_idx2 = np.argmax(action[0]), np.argmax(action[1])
        bs_beam_idx = 8*bs_beam_idx1 + bs_beam_idx2
        bs_beam_idx  = np.array(bs_beam_idx, dtype = int)

        uav_beam_idx = np.argmax(self.best_snr_set[self.iter_, bs_beam_idx, :])
        # always choose best BS
        new_bs_ind = self.state_set[self.iter_, bs_beam_idx, uav_beam_idx]
        snr = np.max(self.trj_data[self.iter_,new_bs_ind, bs_beam_idx, uav_beam_idx])

        if snr < 0:
            reward = -1
        else:
            reward = snr #np.mean(snr_window[-5:])

        if self.iter_ == self.iter_max-1:
            done = True
        else:
            done = False
        self.bs_beam_ind_window = (self.bs_beam_ind_window + bs_beam_idx) % 64
        self.bs_beam_ind_window = np.append(self.bs_beam_ind_window, np.array([bs_beam_idx], dtype = int))
        self.bs_beam_ind_window = self.bs_beam_ind_window[1:]

        self.b_ind_window = (self.b_ind_window + new_bs_ind) % 43
        self.b_ind_window = np.append(self.b_ind_window, np.array([new_bs_ind], dtype = int))
        self.b_ind_window = self.b_ind_window[1:]
        #self.state_next = np.append(self.state[3:], np.array([(new_bs_ind + 1) / 43, (1 + bs_beam_idx) / 64,
        #                                            (1 + uav_beam_idx) / 16], dtype=float))
        self.state_next = np.append(self.b_ind_window, self.bs_beam_ind_window )

        self.iter_ += 1

        return self.state_next, reward, done, {}
