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
torch.manual_seed(100)
np.random.seed(100)

parser = argparse.ArgumentParser()
parser.add_argument("--n_prev_t", default= 20, type = int)
parser.add_argument("--batch_size", default= 100, type=int)
parser.add_argument("--memory_size", default= 3000, type=int)
parser.add_argument("--tau", default=0.005, type = float)
parser.add_argument("--gamma", default=0.9, type = float)
parser.add_argument("--lr_cr", default=0.0001, type = float)
parser.add_argument("--lr_ac", default=0.0001, type = float)
parser.add_argument("--noise_std", default = 1.0, type = float)
parser.add_argument("--policy_noise", default=0.1, type = float)
parser.add_argument("--noise_clip", default=0.2, type = float)
parser.add_argument("--file_n", default = '1' )
args = parser.parse_args()

device = 'cpu'  # cuda' if torch.cuda.is_available else 'cpu'

n_prev_t = args.n_prev_t


kwargs = {
    'n_state': args.n_prev_t,
    'n_action':2,
    'memory_size':args.memory_size,
    'batch_size':args.batch_size,
    'n_train':100,
     'device':device,
    'gamma':args.gamma,
    'tau':args.tau,
    'lr_critic':args.lr_cr,
    'lr_actor': args.lr_ac,
    'actor_update_period':2,
    'noise_std':args.noise_std,
     'policy_noise':args.policy_noise,
     'noise_clip':args.noise_clip
}

print(kwargs)
agent = DDPG(**kwargs)
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
#print(len(file_list5), len(file_list6), len(file_list7))

file_list = np.append(file_list0, file_list1)
file_list = np.append(file_list, file_list2)
file_list = np.append(file_list, file_list3)
file_list = np.append(file_list, file_list4)
file_list = np.append(file_list, file_list5)
file_list = np.append(file_list, file_list6)
file_list = np.append(file_list, file_list7)

np.random.shuffle(file_list)

n_episode = len(file_list)

# actor, actor_target = Actor(n_state = n_state), Actor(n_state = n_state)
# critic, critic_target = Critic(n_state = n_state, n_action = n_action), Critic(n_state = n_state, n_action = n_action)
noise_std, policy_noise, noise_clip = torch.tensor(args.noise_std), torch.tensor(args.policy_noise), torch.tensor(args.noise_clip)
reward_records, snr_records = [], []
loss_records = []
epsilon_records = []
test_snr_list = []
for episode in range(n_episode):
    #ou_noise = OUProcess(mu=np.zeros(1))
    #agent.reset()
    with gzip.open(file_list[episode], 'rb') as f:
        trj_data = np.load(f)
        trj_data = np.transpose(trj_data, (1, 0, 2, 3))

    state_set = np.argmax(trj_data, axis=1) + 1
    # state_set = torch.tensor(state_set/10, dtype =torch.float32)

    snr_set = torch.tensor(np.max(trj_data, axis=1), dtype=torch.float32)/10
    # snr_set = torch.tensor(2*snr_set, dtype = torch.float32)
    #snr_set[snr_set == -10] += torch.rand((snr_set[snr_set == -10].shape))
    # snr_set = snr_set + torch.rand((snr_set.shape))

    snr_set = torch.tensor(snr_set, dtype=torch.float32)
    # set initial state
    state_snr = []
    state_bs_ind = []
    action_ind =[]
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
        state_snr.append(snr_k.item())
        state_bs_ind.append(b_ind)
        #action_ind.append(beam_ind_bs/10)
        #action_ind.append(beam_ind_uav/10)
    # state = np.append(state_bs_ind, state_snr)
    #state = np.append(state_bs_ind, state_snr)
    #action_ind = torch.tensor(action_ind)
    state_snr = torch.tensor(state_snr)
    state= state_snr #np.append(action_ind, state_snr)
    state = torch.tensor(state, dtype=torch.float32).to(device)

    iter_ = n_prev_t
    iter_max = (len(trj_data) - 1)
    reward_history = []
    snr_list = []

    while (iter_ + 1) <= iter_max:
        action = agent.get_action(state).squeeze()
        if episode % 20 != 0:
            noise = torch.normal(mean=0.0, std=torch.tensor(noise_std), size = (2,))* policy_noise
            action = action + noise.clamp(-noise_clip, noise_clip)
            bs_act, uav_act = int(action[0]*64) % 64, int(action[1]*16) % 16
            #noise = noise.clamp(-noise_clip, noise_clip)
            #noise[0] = noise[0]*64
            #noise[1] = noise[1]*64
            #bs_act, uav_act = action[0]%64, action[1]%16
        else:
            bs_act, uav_act = int(action[0]*64) % 64, int(action[1]*16) % 16
            #bs_act, uav_act = action

        #bs_act, uav_act = bs_act.type(torch.int64), uav_act.type(torch.int64)
        snr = snr_set[iter_, bs_act, uav_act]
        snr_list.append(snr)
        reward = snr#/10
        #if snr <= 0:
        #    reward = -1
        #elif snr >0 and snr <10:
        #    reward = 1
        #else:
        #    reward = 1
        reward_history.append(reward)
        iter_ += 1

        new_bs_ind = state_set[iter_, bs_act, uav_act]  # + torch.rand(1)/10
        new_snr = snr_set[iter_, bs_act, uav_act]  # + torch.rand(1)/10

        state_bs_ind = torch.cat([state[1:n_prev_t], torch.tensor([new_bs_ind]).to(device)])
        state_snr = torch.cat([state_snr[1:], torch.tensor([new_snr]).to(device)])

        #action_ind = torch.cat([action_ind[2:], torch.tensor([bs_act/10, uav_act/10])]).to(device)
        #print(action_ind.shape)
        #state_next = torch.cat([action_ind, state_snr])
        state_next = state_snr #torch.cat([state_bs_ind, state_snr]).to(device)

        agent.memorize(state, action, torch.tensor([reward]), state_next)
        state = torch.tensor(state_next, dtype=torch.float32).to(device)

    if len(agent.memory) > sampling_until:
        agent.train()

    if episode % 10 == 0:
        if episode % 20 == 0:
            print(
                "TEST-> [ep {}/{}] iter_max={}, rew={:.2f}, rew/iter_max = {:.3f}, loss={:.2f}, snr sum {:.2f}".format
                (episode + 1, n_episode, iter_max, sum(reward_history), sum(reward_history) / iter_max,
                 agent.loss, sum(snr_list)), flush=True)
            test_snr_list.append(sum(snr_list))
        else:
            print("[ep {}/{}] iter_max={}, rew={:.2f}, rew/iter_max = {:.3f}, loss={:.2f}, snr sum {:.2f}".format
                  (episode + 1, n_episode, iter_max, sum(reward_history), sum(reward_history) / iter_max,
                   agent.loss, sum(snr_list)), flush=True)

    reward_records.append(sum(reward_history))
    snr_records.append(sum(snr_list))
    loss_records.append(agent.loss.item())

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
plt.plot(test_snr_list)
plt.title('TEST_SNR ')
plt.grid()
plt.savefig(save_file_name + 'TEST_SNR.png')

plt.figure()
plt.plot(loss_records)
plt.title('Loss ')
plt.grid()
plt.savefig(save_file_name + '_loss.png')
#np.save(save_file_name + '_loss.npy', loss_records)






























