# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:29:03 2023

@author: seongjoon kang
"""

import numpy as np
import gzip
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import os
#file_list0 = glob.glob('../trjs_data_third_trial/*.npy')
file_list1 = glob.glob('../trjs_data_seventh_trial/hor/*.npy')
#dir_ = '../trjs_data_first_trial/ver/snr_plots/'
#file_list2 = glob.glob('../trjs_data_first_trial/ver/*.npy')
#file_list = np.append(file_list0, file_list1)
file_list = file_list1 #np.append(file_list, file_list2)
#file_list = np.sort(file_list)


#if not os.path.exists(dir_):
    #os.makedirs(dir_)
for k, file_ in enumerate(tqdm(file_list)):
    with open(file_, 'rb') as f:
        snr_data = np.load(f)
    I = np.where(snr_data <-9)[0]
    if len(I) > 10 :
        print(file_)
        n = file_.split('.')[-2].split('_')[-1]
        os.remove('../trjs_data_seventh_trial/hor/trjs_%s.gzip'%n)
    #plt.figure()
    #plt.scatter(range(len(snr_data)), snr_data)
    #plt.grid()
    #plt.title (file_)

    #plt.savefig(dir_ + file_.split('_')[-1]+'.png' )
