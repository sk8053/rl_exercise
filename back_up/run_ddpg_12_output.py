# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 08:05:11 2023

@author: seongjoon kang
"""
import numpy as np
import torch
from ddpg import DDPG
from ddpg import OrnsteinUhlenbeckProcess as OUProcess
import glob
import gzip
import matplotlib.pyplot as plt
import argparse
import copy
from tqdm import tqdm
import os
from utils import test_agent

torch.manual_seed(100)
np.random.seed(100)

parser = argparse.ArgumentParser()
parser.add_argument("--n_prev_t", default = 10, type = int)
parser.add_argument("--batch_size", default = 100, type=int)
parser.add_argument("--memory_size", default = 10000, type=int)
parser.add_argument("--tau", default = 5e-5 , type = float)
parser.add_argument("--gamma", default=0.9, type = float)
parser.add_argument("--lr_cr", default=0.0001, type = float)
parser.add_argument("--lr_ac", default=0.0001, type = float)
parser.add_argument("--noise_std", default = 1.0, type = float)
parser.add_argument("--policy_noise", default=0.1, type = float)
parser.add_argument("--noise_clip", default=0.5, type = float)
parser.add_argument("--file_n", default = '1' )
args = parser.parse_args()

device = 'cpu'  # cuda' if torch.cuda.is_available else 'cpu'

n_prev_t = args.n_prev_t

n_action = 12
kwargs = {
    'n_state': args.n_prev_t *2,
    'n_action':n_action,
    'memory_size':args.memory_size,
    'batch_size':args.batch_size,
    'n_train':200,
     'device':device,
    'gamma':args.gamma,
    'tau':args.tau,
    'lr_critic':args.lr_cr,
    'lr_actor': args.lr_ac,
    'actor_update_period':5,
    'noise_std':args.noise_std,
     'policy_noise':args.policy_noise,
     'noise_clip':args.noise_clip,
    'loading':False
}

print(kwargs)
agent = DDPG(**kwargs)

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
n_episode = len(train_file_list)
# actor, actor_target = Actor(n_state = n_state), Actor(n_state = n_state)
# critic, critic_target = Critic(n_state = n_state, n_action = n_action), Critic(n_state = n_state, n_action = n_action)
noise_std, policy_noise, noise_clip = torch.tensor(args.noise_std), torch.tensor(args.policy_noise), torch.tensor(args.noise_clip)
reward_records, snr_records = [], []
cr_loss_records = []
ac_loss_records = []

epsilon_records = []
test_snr_list = []
error_list = []


for episode in range(n_episode):

    with gzip.open(train_file_list[episode], 'rb') as f:
        trj_data = np.load(f)
        trj_data[trj_data > 30.0] = 30.0
        trj_data[trj_data < -5.0] = -5
        trj_data = trj_data / 10
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
        state = torch.cat([b_ind_window, bs_beam_ind_window])
        state = torch.tensor(state,dtype = torch.float32)

        snr_prev = snr_k.item()
        bs_beam_idx_prev = beam_ind_bs
        uav_beam_idx_prev = beam_ind_uav
        bs_ind_prev = b_ind
        action_prev = bs_ind_prev/64

        iter_ = n_prev_t
        iter_max = (len(trj_data) - 1)
        reward_history = []
        snr_list = []

        for iter_ in np.arange(n_prev_t, iter_max):
            action = agent.get_action(state).squeeze()
            noise = torch.normal(mean=0.0, std=noise_std, size=(n_action,)) * policy_noise
            action = action + noise.clamp(-noise_clip, noise_clip)
            
            action1, action2 = action[:6], action[6:]
            action_value = torch.vstack((action1,action2))
            binary_value = torch.argmax(action_value, 0)
            bs_beam_idx = binary_value*torch.tensor([32,16,8,4,2,1])
            bs_beam_idx = bs_beam_idx.sum()
            
            min_idx = torch.argmin(action_value, 0)
            action_value[min_idx, range(6)] =0 
            action = action_value.reshape(-1)
            #print(action)
            #bs_beam_idx1, bs_beam_idx2  = torch.argmax(action[:8]), torch.argmax(action[8:])   
            #bs_beam_idx = 8*bs_beam_idx1 + bs_beam_idx2

            bs_beam_idx  = torch.tensor(bs_beam_idx, dtype = torch.int64)
            uav_beam_idx = torch.argmax(best_snr_set[iter_, bs_beam_idx, :])
            # always choose best BS
            new_bs_ind = state_set[iter_, bs_beam_idx, uav_beam_idx]
            snr = np.max(trj_data[iter_,new_bs_ind, bs_beam_idx, uav_beam_idx])

            snr_list.append(snr)
            snr_window.append(snr)
            snr_window = snr_window[1:]

            if snr < 0:
                reward = -1
            else:
                reward = np.mean(snr_window[-3:])/5

            reward_history.append(reward)
            
            bs_beam_ind_window = (bs_beam_ind_window  + bs_beam_idx) %64
            bs_beam_ind_window= torch.cat([bs_beam_ind_window, torch.tensor([bs_beam_idx])])
            bs_beam_ind_window = bs_beam_ind_window[1:]

            b_ind_window = (b_ind_window + new_bs_ind)%43
            b_ind_window = torch.cat([b_ind_window, torch.tensor([new_bs_ind])])
            b_ind_window = b_ind_window[1:]
            #snr = snr + torch.rand(1)/100
             #+ torch.rand(1)/100
            #state_next = torch.cat([state[3:], torch.tensor([(new_bs_ind+1)/43, (1+bs_beam_idx)/64,(1+uav_beam_idx)/16], dtype = torch.float32)])
            state_next = torch.cat([b_ind_window, bs_beam_ind_window])
            agent.memorize(state, torch.tensor(action), torch.tensor([reward]), state_next)
            state = state_next.to(device).type(torch.float32)
            snr_prev = snr
            uav_beam_idx_prev = uav_beam_idx
            bs_beam_idx_prev = bs_beam_idx_prev
            
        if len(agent.memory) > sampling_until:
            agent.train()


        if episode % 200 == 0  and episode !=0:
            print('===========================test agent====================')
            record_data = {
                'reward_records':reward_records,
                'snr_records': snr_records,
                'cr_loss_records': cr_loss_records,
                'ac_loss_records': ac_loss_records
            }
            #episode, test_files, agent, record_data, save_dir = 'checkpoints/imgs',n_prev_t = 25
            test_agent(episode, test_files, agent, record_data, n_action=n_action, save_dir='checkpoints/imgs', n_prev_t= n_prev_t)

    snr_list = np.array(snr_list)
    max_snr = np.max(trj_data, 1).max(1).max(1)
    max_snr = max_snr[n_prev_t+1:]
    test_snr_list.append(sum(snr_list))
    L = len(max_snr)
    error = np.linalg.norm(max_snr*10 - snr_list*10, 2) / np.sqrt(L)
    error_list.append(error)

    if episode %10 ==0:
        print("[ep {}/{}] iter_max={}, rew={:.2f}, error = {:.3f}, cr/ac loss= {:.2f} {:.2f}, snr>0  {:.2f}".format
              (episode + 1, n_episode, iter_max, sum(reward_history), error,
               agent.cr_loss.item(),agent.ac_loss.item(), np.sum(snr_list>0)/len(snr_list)), flush=True)

    reward_records.append(sum(reward_history))
    snr_records.append(sum(snr_list))
    cr_loss_records.append(agent.cr_loss.item())
    ac_loss_records.append(agent.ac_loss.item())

agent.save(save_file_path)

# epsilon_records = agent.epsilon_records
plt.figure()
plt.plot(reward_records)
plt.title('Reward')
plt.grid()
plt.savefig(save_file_name + '_reward.png')
np.save(save_file_name + '_reward.npy', reward_records)

plt.figure()
plt.plot(snr_records)
plt.title('SNR ')
plt.grid()
plt.savefig(save_file_name + '_SNR.png')
np.save(save_file_name + '_SNR.npy', snr_records)

plt.figure()
plt.plot(test_snr_list)
plt.title('TEST_SNR ')
plt.grid()
plt.savefig(save_file_name + 'TEST_SNR.png')

plt.figure()
plt.plot(cr_loss_records)
plt.title('CR_Loss ')
plt.grid()
plt.savefig(save_file_name + '_cr_loss.png')
np.save(save_file_name + '_cr_loss.npy', cr_loss_records)

plt.figure()
plt.plot(ac_loss_records)
plt.title('AC_Loss ')
plt.grid()
plt.savefig(save_file_name + '_ac_loss.png')
np.save(save_file_name + '_ac_loss.npy', ac_loss_records)

np.save(save_file_name + '_error.npy', error_list )


'''
    if episode % 10 == 0:

        if episode % 40 == 0:
            print(
                "TEST=> [ep {}/{}] iter_max={}, rew={:.2f}, error = {:.3f}, cr/ac loss={:.2f} {:.2f}, snr >0  {:.2f}".format
                (episode + 1, n_episode, iter_max, sum(reward_history), error,
                 agent.cr_loss, agent.ac_loss, np.sum(snr_list>0)/len(snr_list)), flush=True)


            plt.figure()
            plt.scatter(np.arange(len(snr_list)), np.array(snr_list)*10, c = 'k')
            plt.scatter(np.arange(len(max_snr)), max_snr*10, c = 'r')
            plt.grid()
            plt.title("file->{}".format(file_list[episode]))
            plt.savefig('checkpoints/imgs/%d.png'%episode)
'''



























