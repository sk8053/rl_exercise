
import numpy as np
import os
from tqdm import tqdm
import gzip
import matplotlib.pyplot as plt
import torch
import glob

def test_agent(episode, test_files, agent, record_data,n_action = 2,  save_dir = 'checkpoints/imgs',n_prev_t = 25, ddqn = False):
    agent.eval()
    dir_ = '%s/_test_%d'%(save_dir,episode)
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    folder_name = '_test_%d'%episode
    error_list_test = []
    for j, test_file in tqdm(enumerate(test_files)):
        with gzip.open(test_file, 'rb') as f:
            trj_data = np.load(f)
            trj_data[trj_data >30.0] = 30.0
            trj_data[trj_data < -5] = -5
            trj_data = trj_data/30

            trj_data = np.transpose(trj_data, (1, 0, 2, 3))
            state_set = np.argmax(trj_data, axis=1)
            best_snr_set = torch.tensor(np.max(trj_data, axis=1), dtype=torch.float32)
            # set initial state
            state = []
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
                #state.append(snr_k.item())
                #state.append((b_ind + 1) / 43)
                #state.append((beam_ind_bs + 1) / 64)
                #state.append((beam_ind_uav + 1) / 16)
                b_ind_window.append(b_ind)
                bs_beam_ind_window.append(beam_ind_bs)

            state = torch.tensor(state, dtype=torch.float32)
            b_ind_window, bs_beam_ind_window = torch.tensor(b_ind_window, dtype = torch.float32), torch.tensor(bs_beam_ind_window, dtype = torch.float32)
            state = torch.cat([b_ind_window, bs_beam_ind_window ])
            #snr_prev = snr_k.item()
            bs_act_prev = beam_ind_bs
            #uav_act_prev = beam_ind_uav
            #bs_ind_prev = b_ind
            #action_prev = bs_act_prev / 64
            snr_prev = snr_k.item()
            bs_beam_idx_prev = beam_ind_bs
            uav_beam_idx_prev = beam_ind_uav
            bs_ind_prev = b_ind
            action_prev = bs_ind_prev / 64
            iter_max = (len(trj_data) - 1)
            snr_list = []

            for iter_ in np.arange(n_prev_t, iter_max):
                #if ddqn is True:
                action = agent.get_action(state, random = False)
                #noise = torch.normal(mean=0.0, std=1.0, size=(n_action,)) * 0.1
                #action = action + noise.clamp(-0.1, 0.1)

                # action2 = agent2.get_action(state, random=False)

                bs_beam_idx1, bs_beam_idx2  = torch.argmax(action[:8]), torch.argmax(action[8:])
                bs_beam_idx = 8*bs_beam_idx1 + bs_beam_idx2
                    #bs_beam_idx = int(action * 64) % 64
                    #bs_beam_idx = int(bs_beam_idx/8)
                    #bs_beam_idx = bs_beam_idx * 8


                # find index of UAV beamforming
                snr = torch.max(best_snr_set[iter_, bs_beam_idx, :]).item()
                uav_beam_idx = torch.argmax(best_snr_set[iter_, bs_beam_idx, :])
                # always choose best BS
                new_bs_ind = state_set[iter_, int(bs_beam_idx), int(uav_beam_idx)]
                #if trj_data[iter_, bs_ind_prev, bs_beam_idx_prev, uav_beam_idx_prev] > trj_data[iter_, new_bs_ind, bs_beam_idx, uav_beam_idx]:
                #    new_bs_ind = bs_ind_prev
                #    bs_beam_idx = bs_beam_idx_prev
                #    uav_beam_idx = uav_beam_idx_prev
                #    snr = trj_data[iter_, bs_ind_prev, bs_beam_idx_prev, uav_beam_idx_prev]
                #else:
                #    snr = trj_data[iter_, new_bs_ind, bs_beam_idx, uav_beam_idx]
                #    bs_ind_prev = new_bs_ind
                #    bs_beam_idx_prev = bs_beam_idx
                #    uav_beam_idx_prev = uav_beam_idx

                snr = trj_data[iter_, new_bs_ind, bs_beam_idx, uav_beam_idx]

                bs_beam_ind_window = (bs_beam_ind_window + bs_beam_idx) % 64
                bs_beam_ind_window = torch.cat([bs_beam_ind_window, torch.tensor([bs_beam_idx])])
                bs_beam_ind_window = bs_beam_ind_window[1:]

                b_ind_window = (b_ind_window + new_bs_ind) % 43
                b_ind_window = torch.cat([b_ind_window, torch.tensor([new_bs_ind])])
                b_ind_window = b_ind_window[1:]

                #state = torch.cat([state[3:], torch.tensor([(new_bs_ind + 1) / 43, (beam_ind_bs + 1) / 64, (beam_ind_uav + 1) / 16])])
                state = torch.cat([b_ind_window , bs_beam_ind_window ])
                state = torch.tensor(state, dtype = torch.float32)
                snr_list.append(snr)

            snr_list = np.array(snr_list)
            max_snr = np.max(trj_data, 1).max(1).max(1)
            max_snr = max_snr[n_prev_t + 1:]
            L = len(max_snr)
            error = np.linalg.norm(max_snr*30 - snr_list*30, 2) / np.sqrt(L)
            error_list_test.append(error)

            plt.figure()
            plt.scatter(np.arange(len(snr_list)), np.array(snr_list) * 30, c='k')
            plt.scatter(np.arange(len(max_snr)), max_snr * 30, c='r')
            plt.grid()
            plt.title("file->{}".format(test_file))
            plt.savefig('%s/%s/test_%d_%d.png' % (save_dir,folder_name, j,episode))
    plt.figure()
    #plt.scatter(np.arange(len(error_list_test)), np.array(error_list_test) * 10, c='k')
    plt.plot(np.sort(error_list_test), np.linspace(0,1,len(error_list_test)))
    plt.grid()
    plt.title("file->{}".format(test_file))
    plt.savefig('%s/%s/error_%d.png' % (save_dir, folder_name,episode))
    np.save('%s/%s/_error_%d.npy' % (save_dir, folder_name,episode), error_list_test)

    for key in record_data.keys():
        plt.figure()
        d = record_data[key]
        plt.scatter(np.arange(len(d)), d)
        plt.title("{}".format(key))
        plt.savefig('%s/%s/%s_%d.png' % (save_dir, folder_name,key, episode))
        np.save('%s/%s/_%s_%d.npy' % (save_dir, folder_name, key, episode), d)

    return error_list_test
def dqn_test_agent(episode, test_files, agent, record_data,n_action = 2,  save_dir = 'checkpoints/imgs',n_prev_t = 25, ddqn = False):
    agent.eval()
    dir_ = '%s/_dqn_test_%d'%(save_dir,episode)
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    folder_name = '_dqn_test_%d'%episode
    error_list_test = []
    for j, test_file in tqdm(enumerate(test_files)):
        with gzip.open(test_file, 'rb') as f:
            trj_data = np.load(f)
            trj_data[trj_data >30.0] = 30.0
            trj_data[trj_data < -5] = -5
            trj_data = trj_data/10

            trj_data = np.transpose(trj_data, (1, 0, 2, 3))
            state_set = np.argmax(trj_data, axis=1)
            best_snr_set = torch.tensor(np.max(trj_data, axis=1), dtype=torch.float32)
            # set initial state
            state = []
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
                b_ind_window.append(b_ind)
                bs_beam_ind_window.append(beam_ind_bs)

            state = torch.tensor(state, dtype=torch.float32)
            b_ind_window, bs_beam_ind_window = torch.tensor(b_ind_window), torch.tensor(bs_beam_ind_window)
            state = torch.cat([b_ind_window / 43, bs_beam_ind_window / 64])
            #snr_prev = snr_k.item()
            bs_act_prev = beam_ind_bs
            #uav_act_prev = beam_ind_uav
            #bs_ind_prev = b_ind
            #action_prev = bs_act_prev / 64
            snr_prev = snr_k.item()
            bs_beam_idx_prev = beam_ind_bs
            uav_beam_idx_prev = beam_ind_uav
            bs_ind_prev = b_ind
            action_prev = bs_ind_prev / 64
            iter_max = (len(trj_data) - 1)
            snr_list = []

            for iter_ in np.arange(n_prev_t, iter_max):
                #if ddqn is True:
                action = agent.get_action(state, random = False)
                #noise = torch.normal(mean=0.0, std=1.0, size=(n_action,)) * 0.1
                #action = action + noise.clamp(-0.1, 0.1)

                # action2 = agent2.get_action(state, random=False)

                bs_beam_idx1, bs_beam_idx2  = action[0], action[1]
                bs_beam_idx = 8*bs_beam_idx1 + bs_beam_idx2
                    #bs_beam_idx = int(action * 64) % 64
                    #bs_beam_idx = int(bs_beam_idx/8)
                    #bs_beam_idx = bs_beam_idx * 8


                # find index of UAV beamforming
                snr = torch.max(best_snr_set[iter_, bs_beam_idx, :]).item()
                uav_beam_idx = torch.argmax(best_snr_set[iter_, bs_beam_idx, :])
                # always choose best BS
                new_bs_ind = state_set[iter_, int(bs_beam_idx), int(uav_beam_idx)]
                #if trj_data[iter_, bs_ind_prev, bs_beam_idx_prev, uav_beam_idx_prev] > trj_data[iter_, new_bs_ind, bs_beam_idx, uav_beam_idx]:
                #    new_bs_ind = bs_ind_prev
                #    bs_beam_idx = bs_beam_idx_prev
                #    uav_beam_idx = uav_beam_idx_prev
                #    snr = trj_data[iter_, bs_ind_prev, bs_beam_idx_prev, uav_beam_idx_prev]
                #else:
                #    snr = trj_data[iter_, new_bs_ind, bs_beam_idx, uav_beam_idx]
                #    bs_ind_prev = new_bs_ind
                #    bs_beam_idx_prev = bs_beam_idx
                #    uav_beam_idx_prev = uav_beam_idx

                snr = trj_data[iter_, new_bs_ind, bs_beam_idx, uav_beam_idx]

                bs_beam_ind_window = (bs_beam_ind_window + bs_beam_idx) % 64
                bs_beam_ind_window = torch.cat([bs_beam_ind_window, torch.tensor([bs_beam_idx])])
                bs_beam_ind_window = bs_beam_ind_window[1:]

                b_ind_window = (b_ind_window + new_bs_ind) % 43
                b_ind_window = torch.cat([b_ind_window, torch.tensor([new_bs_ind])])
                b_ind_window = b_ind_window[1:]

                #state = torch.cat([state[3:], torch.tensor([(new_bs_ind + 1) / 43, (beam_ind_bs + 1) / 64, (beam_ind_uav + 1) / 16])])
                state = torch.cat([b_ind_window / 43, bs_beam_ind_window / 64])
                state = torch.tensor(state, dtype = torch.float32)
                snr_list.append(snr)

            snr_list = np.array(snr_list)
            max_snr = np.max(trj_data, 1).max(1).max(1)
            max_snr = max_snr[n_prev_t + 1:]
            L = len(max_snr)
            error = np.linalg.norm(max_snr*10 - snr_list*10, 2) / np.sqrt(L)
            error_list_test.append(error)

            plt.figure()
            plt.scatter(np.arange(len(snr_list)), np.array(snr_list) * 10, c='k')
            plt.scatter(np.arange(len(max_snr)), max_snr * 10, c='r')
            plt.grid()
            plt.title("file->{}".format(test_file))
            plt.savefig('%s/%s/test_%d_%d.png' % (save_dir,folder_name, j,episode))
    plt.figure()
    #plt.scatter(np.arange(len(error_list_test)), np.array(error_list_test) * 10, c='k')
    plt.plot(np.sort(error_list_test), np.linspace(0,1,len(error_list_test)))
    plt.grid()
    plt.title("file->{}".format(test_file))
    plt.savefig('%s/%s/error_%d.png' % (save_dir, folder_name,episode))
    np.save('%s/%s/_error_%d.npy' % (save_dir, folder_name,episode), error_list_test)

    for key in record_data.keys():
        plt.figure()
        d = record_data[key]
        plt.scatter(np.arange(len(d)), d)
        plt.title("{}".format(key))
        plt.savefig('%s/%s/%s_%d.png' % (save_dir, folder_name,key, episode))
        np.save('%s/%s/_%s_%d.npy' % (save_dir, folder_name, key, episode), d)

    return error_list_test

def read_file_list():
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
    # print(len(file_list5), len(file_list6), len(file_list7))

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
    return file_list