# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 08:05:11 2023

@author: seongjoon kang
"""
import numpy as np
import torch
from ddqn_agent import DDQN_Agent
import glob
import gzip
import matplotlib.pyplot as plt
import argparse
import copy
from utils import dqn_test_agent
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--n_prev_t", default= 25, type = int)
parser.add_argument("--batch_size", default= 128, type=int)
parser.add_argument("--memory_size", default= 10000, type=int)
parser.add_argument("--gamma", default=0.9, type = float)


parser.add_argument("--file_n", default = '1' )
args = parser.parse_args()

device = 'cpu'  # cuda' if torch.cuda.is_available else 'cpu'

n_prev_t = args.n_prev_t

tau = 5e-3
kwargs = {
    'n_states': args.n_prev_t *2,
    'n_actions':16,
    'memory_size':args.memory_size,
    'batch_size':args.batch_size,
    'n_train':200,
     'device':device,
    'gamma':args.gamma,
     'eps_min':0.0005,
    'eps_max':0.6,
    'lr': 1e-4,
    'tau':tau
}

print(kwargs)
agent = DDQN_Agent(**kwargs)
#kwargs['n_state'] = args.n_prev_t*8
#agent2 = DDPG(**kwargs)

sampling_until = int(2 * kwargs['memory_size'] / 3)
save_file_name = 'checkpoints/TD3_%s' % args.file_n
save_file_path = save_file_name + '.pt'

file_list0 = glob.glob('../trjs_data_third_trial/*.gzip')
file_list1 = glob.glob('../trjs_data_first_trial/hor/*.gzip')
file_list2 = glob.glob('../trjs_data_first_trial/ver/*.gzip')
file_list3 = glob.glob('../trjs_data_second_trial/hor/*.gzip')
file_list4 = glob.glob('../trjs_data_second_trial/ver/*.gzip')
file_list5 = glob.glob('../trjs_data_fourth_trial/*.gzip')
file_list6 = glob.glob('../trjs_data_fifth_trial/hor/*.gzip')
file_list7 = glob.glob('../trjs_data_fifth_trial/ver/*.gzip')
file_list8 = glob.glob('../trjs_data_sixth_trial/ver/*.gzip')
file_list9 = glob.glob('../trjs_data_sixth_trial/hor/*.gzip')
file_list10 = glob.glob('../trjs_data_seventh_trial/ver/*.gzip')
file_list11 = glob.glob('../trjs_data_seventh_trial/hor/*.gzip')
#print(len(file_list5), len(file_list6), len(file_list7))

file_list = np.append(file_list0, file_list1)
file_list = np.append(file_list, file_list2)
file_list = np.append(file_list, file_list3)
file_list = np.append(file_list, file_list4)
file_list = np.append(file_list, file_list5)
file_list = np.append(file_list, file_list6)
file_list = np.append(file_list, file_list7)
file_list = np.append(file_list, file_list8)
file_list = np.append(file_list, file_list9)
file_list = np.append(file_list, file_list10)
file_list = np.append(file_list, file_list11)

np.random.shuffle(file_list)


test_files = file_list[:150]
train_file_list = file_list[150:]
n_episode = len(file_list)

reward_records, snr_records = [], []
loss_records = []
epsilon_records = []

for episode in range(n_episode):

    with gzip.open(file_list[episode], 'rb') as f:
        trj_data = np.load(f)
        # clip SNR values having more than 30dB
        trj_data[trj_data > 30] = 30.0
        trj_data = trj_data/10
        trj_data = np.transpose(trj_data, (1, 0, 2, 3))
    #for f in range (2):
    #    if f ==1:
    #        trj_data = np.flip(trj_data, axis =0)

        state_set = np.argmax(trj_data, axis=1)
        best_snr_set = torch.tensor(np.max(trj_data, axis=1), dtype=torch.float32)

        # set initial state
        state = []
        snr_window = []
        b_ind_window = []
        bs_beam_ind_window = []

        for k in range(n_prev_t):
            v = np.max(trj_data[k, :, :, :])
            I = np.where(v == trj_data[k, :, :, :])

            if len(I[0]) > 1:
                b_ind = I[0][0]
                beam_ind_bs = I[1][0]
                beam_ind_uav = I[2][0]
            else:
                b_ind = I[0].item()
                beam_ind_bs = I[1].item()
                beam_ind_uav = I[2].item()

            snr_k = trj_data[k, b_ind, beam_ind_bs, beam_ind_uav]
            snr_window.append(snr_k)
            b_ind_window.append(b_ind)
            bs_beam_ind_window.append(beam_ind_bs)
            #state.append(snr_k.item())
            #state.append((b_ind+1)/43)
            #state.append((beam_ind_bs+1)/64)
            #state.append((beam_ind_uav+1)/16)
        b_ind_window, bs_beam_ind_window = torch.tensor(b_ind_window), torch.tensor(bs_beam_ind_window)
        state = torch.cat([b_ind_window/43, bs_beam_ind_window/64])
        state = torch.tensor(state,dtype = torch.float32)
        iter_max = (len(trj_data) - 1)
        reward_history = []
        snr_list = []

        for iter_ in np.arange(n_prev_t, iter_max):

            action = agent.get_action(state, episode, random = False)  # (9,)
            bs_act1 = action[0]
            bs_act2 = action[1]
            bs_beam_idx = bs_act1 * 8 + bs_act2
            # find index of UAV beamforming
            uav_beam_idx = torch.argmax(best_snr_set[iter_, bs_beam_idx, :])
            # always choose best BS
            new_bs_ind = state_set[iter_, int(bs_beam_idx), int(uav_beam_idx)]

            snr = trj_data[iter_,new_bs_ind, bs_beam_idx, uav_beam_idx]
            snr_list.append(snr)
            snr_window.append(snr)
            snr_window = snr_window[1:]

            if snr <0:
                reward = -1
            else:
                reward = snr

            reward_history.append(reward)
            bs_beam_ind_window = (bs_beam_ind_window  + bs_beam_idx) %64
            bs_beam_ind_window= torch.cat([bs_beam_ind_window, torch.tensor([bs_beam_idx])])
            bs_beam_ind_window = bs_beam_ind_window[1:]

            b_ind_window = (b_ind_window + new_bs_ind)%43
            b_ind_window = torch.cat([b_ind_window, torch.tensor([new_bs_ind])])
            b_ind_window = b_ind_window[1:]

            #state_next = torch.cat([state[4:], torch.tensor([snr, (new_bs_ind+1)/43, (1+bs_beam_idx)/64, (1+uav_beam_idx)/16], dtype = torch.float32)])
            state_next = torch.cat([b_ind_window / 43, bs_beam_ind_window / 64])
            agent.memorize(state, torch.tensor(action), torch.tensor([reward]), state_next)
            state = state_next.to(device).type(torch.float32)

        if len(agent.memory) > sampling_until:
            agent.train()


        snr_list = np.array(snr_list)
        if episode % 10 == 0:
            print("[ep {}/{}] iter_max={}, sum_rew={:.2f}, avg_reward = {:.3f},eps={:.3f}, loss={:.2f}, snr>0  {:.2f}".format
                  (episode + 1, n_episode, iter_max, sum(reward_history), sum(reward_history) / iter_max, agent.epsilon,
                   agent.loss, np.sum(snr_list > 0) / len(snr_list)), flush=True)


    reward_records.append(sum(reward_history))
    snr_records.append(sum(snr_list))
    loss_records.append(agent.loss.item())

    if episode % 200 == 0 and episode !=0:
        print ("================== test agent ================")
        record_data = {
            'reward_records': reward_records,
            'snr_records': snr_records,
            'loss_records': loss_records,
        }
        dqn_test_agent(episode, test_files, agent, record_data, save_dir='checkpoints/ddqn_imgs', n_prev_t = n_prev_t, ddqn = True)

agent.save(save_file_path)

# epsilon_records = agent.epsilon_records
plt.figure()
plt.plot(reward_records)
plt.title('Reward')
plt.grid()
plt.savefig(save_file_name + '_reward.png')
#np.save(save_file_name + '_reward.npy', reward_records)

plt.figure()
plt.plot(snr_records)
plt.title('SNR ')
plt.grid()
plt.savefig(save_file_name + '_SNR.png')
#np.save(save_file_name + '_SNR.npy', snr_records)


plt.figure()
plt.plot(loss_records)
plt.title('Loss ')
plt.grid()
plt.savefig(save_file_name + '_loss.png')
#np.save(save_file_name + '_loss.npy', loss_records)






























