# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:59:17 2023

@author: seongjoon kang
"""

import numpy as np
#from dqn_agent import DQNAgent
from ddqn_agent import DDQN_Agent
import gzip
import torch
import glob
import matplotlib.pyplot as plt


def convert_dec_to_bin(bin_list):
 
    bin_list = torch.tensor(bin_list, dtype = torch.int64)
    bb = '0b'+''.join(np.array(bin_list.detach().cpu().numpy(), dtype = str))

    return int(bb, 2)

loss = -1 # track loss
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

np.random.shuffle(file_list)

n_episode = len(file_list)
device = 'cpu' #cuda' if torch.cuda.is_available else 'cpu'

capacity = 1000 # small capacity is better

sampling_until = int(2*capacity/3)
target_update_interval = 3 # for hard-update
n_prev_t = 20
gamma = 0.9
eps_min = 0.01
eps_max = 0.6
batch_size = 120 # small batch size is better
train_numbers = 100

save_file_name = 'checkpoints/n_prev_t=%d_train_n=%d_batchsize=%d'%( n_prev_t, train_numbers, batch_size)
save_file_path = save_file_name+'.pt'

agent = DDQN_Agent(n_states=n_prev_t, 
                 n_actions =8*2 + 4*2, 
                 memory_size = capacity, 
                 gamma = gamma,
                 eps_min = eps_min, 
                 eps_max = eps_max, 
                 batch_size = batch_size, 
                 n_train = train_numbers, 
                 device = device)



reward_records, snr_records = [], []
loss_records = []
epsilon_records = []
for episode in range(n_episode):

    with gzip.open(file_list[episode], 'rb') as f:
        trj_data = np.load(f)
        trj_data = np.transpose(trj_data, (1,0,2,3))/10

    state_set = np.argmax(trj_data, axis=1)
    best_snr_set = torch.tensor(np.max(trj_data, axis=1), dtype=torch.float32)
    
    #snr_set = torch.tensor(np.max(trj_data, axis = 1), dtype = torch.float32)
    #snr_set = torch.tensor(2*snr_set, dtype = torch.float32)
    #snr_set[snr_set==-10] += torch.rand((snr_set[snr_set==-10].shape))
    #snr_set = snr_set + torch.rand((snr_set.shape))
    
    #snr_set = torch.tensor(snr_set/10, dtype = torch.float32)
    # set initial state
    state_snr =[]
    state_bs_ind = []
    for k in range(n_prev_t):
        v = np.max(trj_data[k,:,:,:])
        I = np.where(v == trj_data[k,:,:,:])
        
        if len(I[0]) >1:
            b_ind = I[0][0]
            beam_ind_bs = I[1][0]
            beam_ind_uav = I[2][0]
        else:
            b_ind = I[0].item()
            beam_ind_bs = I[1].item()
            beam_ind_uav = I[2].item()
            
        snr_k = trj_data[k, b_ind,beam_ind_bs,beam_ind_uav]
        state_snr.append(snr_k.item())
        state_bs_ind.append(b_ind)
        
    
    #state = np.append(state_bs_ind, state_snr)  
    state = state_snr #np.append(state_bs_ind, state_snr)  
    state = torch.tensor(state, dtype = torch.float32).to(device)
    
    iter_ =n_prev_t
    iter_max = (len(trj_data)-1)
    reward_history = []
    snr_list = []
    while (iter_+1) <= iter_max :
        action = agent.decide_action(state, episode) # (9,)
        bs_act, uav_act = action[0]*8 + (action[1]-8), action[2]*4 + (action[3]-4)
        
        snr = snr_set[iter_, bs_act, uav_act]
        snr_list.append(snr)
        #reward = snr/10
        if snr <= 0:
            reward = -1.0
        #elif snr >0 and snr <10:
        #    reward = 0.5
        else:
            reward = snr
        reward_history.append(reward)
        iter_ +=1
        
        new_bs_ind = state_set[iter_, bs_act, uav_act] #+ torch.rand(1)/10
        new_snr = snr_set[iter_,bs_act, uav_act] #+ torch.rand(1)/10
        
        state_bs_ind = torch.cat([state[1:n_prev_t], torch.tensor([new_bs_ind]).to(device)])
        state_snr  = torch.cat([state[1:], torch.tensor([new_snr]).to(device)])
    
        state_next = state_snr # torch.cat([state_bs_ind, state_snr]).to(device)
     
        agent.memorize(state, action, torch.tensor([reward]), state_next)
        state = torch.tensor(state_next,  dtype = torch.float32).to(device)
     
        
    if agent.memory.__len__() >sampling_until:
        agent.train()
    
        
    print("[ep {}/{}] iter_max={}, rew={:.2f}, rew/iter_max = {:.3f} mem={}, eps={:.3f}, loss={:.2f}, snr sum {:.2f}".format
              (episode+1, n_episode, iter_max, sum(reward_history),sum(reward_history)/iter_max, 
               len(agent.memory), 
               agent.epsilon, agent.loss, sum(snr_list)), flush=True)
    
    reward_records.append(sum(reward_history))
    snr_records.append(sum(snr_list))
    loss_records.append(agent.loss.item())
    epsilon_records.append(agent.epsilon)
    #if episode % 3 == 0:
    if torch.rand(1)>0.5:
        agent.train(model_name = 'A')
    else:
        agent.train(model_name = 'B')

   #if episode% target_update_interval ==0:
   #     agent.update_model()
    #agent.soft_update_model(tau = 0.7)

agent.save(save_file_path)

#epsilon_records = agent.epsilon_records
plt.figure()
plt.plot(reward_records)
plt.title('Reward')
plt.grid()
plt.savefig(save_file_name+'_reward.png')
np.save(save_file_name+'_reward.npy', reward_records)

plt.figure()
plt.plot(snr_records)
plt.title('SNR ')
plt.grid()
plt.savefig(save_file_name+'_SNR.png')
np.save(save_file_name+'_SNR.npy', snr_records)

plt.figure()
plt.plot(loss_records)
plt.title('Loss ')
plt.grid()
plt.savefig(save_file_name+'_loss.png')
np.save(save_file_name+'_loss.npy', loss_records)

