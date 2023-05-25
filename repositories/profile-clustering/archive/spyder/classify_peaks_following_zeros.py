# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:19:22 2020

@author: arasy
"""

#%% imports
import sys
from pathlib import Path
sys.path.append(Path(__file__).absolute().parent)

import numpy as np
import matplotlib.pyplot as plt
import itertools
from operator import itemgetter

from energyclustering import data as evd
import data_cleaning.find_problems_in_data as fpid

from scipy.signal import find_peaks

from sklearn_extra.cluster import KMedoids

# %% Import data:
master_table = evd.get_master_table()
# data_reading_full = evd.get_data_reading_full()
df = evd.get_data_reading_preprocessed()

iIDs = fpid.get_iID_info()[0]

# time_indices_full, time_first, time_last = fpid.get_time_info()

# %% Find zeros:
# Zero consumptions and the peaks following them for houses with a single meter (excluding houses with two meters):
no_of_consecutive_zeros, no_of_consecutive_zeros_, zero_duration_hours, inds_zeros, inds_zeros_, times_after_zeros, times_after_zeros_ = fpid.get_zeros_with_ts(typ='Consumption', single_meter=True)
peaks_after_zeros, peaks_after_zeros_ = fpid.peaks_after_zeros(typ='Consumption', single_meter=True)

consumption_after_zeros_length = 15 # length of time series following the zero intervals
# the consumption time series following each of the zero intervals:
consumption_after_zeros, consumption_after_zeros_, inds_after_zeros, inds_after_zeros_, times_after_zeros, times_after_zeros_ = fpid.series_after_zeros(typ='Consumption', single_meter=True, series_length=consumption_after_zeros_length)

# %% Analyze a case with double peaks:
iID_double_peaks = '/OK5PqFX3nlnYQ'
iID_double_peaks_ind = np.where(iIDs == iID_double_peaks)

conss = consumption_after_zeros[iID_double_peaks_ind[0][0]]

# %% Find peaks:   (NOT USED)
def find_pks(x, params=dict()):
    pk_inds = find_peaks(x, **params)[0]
    pks = itemgetter(*pk_inds)(x) if len(pk_inds) else []
    return pk_inds, pks

def plot_pks(x, pk_inds):
    pks = itemgetter(*pk_inds)(x)
    plt.figure()
    plt.plot(x)
    plt.plot(pk_inds, pks, '*r')

# peak_inds = [[find_pks(x)[0] if x != None and len(x) else [] for x in c] if len(c) else [] for c in consumption_after_zeros]
# peaks = [[find_pks(x)[1] if x != None and len(x) else [] for x in c] if len(c) else [] for c in consumption_after_zeros]

peak_inds = []
peaks = []
for c in consumption_after_zeros:
    p = []
    i = []
    if len(c):
        for x in c:
            if x != None and len(x):
                pk_inds, pks = find_pks(x)
            else:
                pk_inds, pks = [], []
            i.append(pk_inds)
            p.append(pks)
    peak_inds.append(i)
    peaks.append(p)
    
peak_inds_ = list(itertools.chain(*peak_inds))
peaks_ = list(itertools.chain(*peaks))

# %% Cluster consumptions (as time series) after zeros:
# N = len(no_of_consecutive_zeros_)
mask = []
X = []
for i, c in enumerate(consumption_after_zeros_):
    if len(c) >= consumption_after_zeros_length and no_of_consecutive_zeros_[i] >= 4 and not None in c: # the period follows the zzero interval is long enough, min number of zeros in the zero interval, no None values
        X.append(c)
        mask.append(True)
    else:
        mask.append(False)
X = np.array(X)
N = X.shape[0]
M = X.shape[1]

# plt.figure()
# plt.plot(X.transpose(), LineWidth=0.5)
# plt.xlabel('time sample')

K = 2 # number of clusters

# kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
kmeans = KMedoids(n_clusters=K, random_state=0).old_fit(X)

L = kmeans.labels_

U = np.unique(L)
C = len(U)

plt.figure()
fig, axs = plt.subplots(C, 1, sharex=True, sharey=True)
for i, u in enumerate(U):
    axs[i].plot(np.arange(0, M/4, .25), X[L == i].transpose(), '.-', LineWidth=0.5)
    axs[i].set_ylabel('cluster {}'.format(i+1))
    if i == 0: axs[i].set_title("time series clustering")
plt.xlabel("time (hr)")

# %% Cluster consumptions after zeros based on features extracted from them:
mask = []
Y = []
for i, c in enumerate(consumption_after_zeros_):
    if len(c) >= consumption_after_zeros_length and no_of_consecutive_zeros_[i] >= 4 and not None in c: # the period follows the zzero interval is long enough, min number of zeros in the zero interval, no None values
        y = [c[1], # first consumption after the zero period (possibly a peak)
             c[2], # second consumption measurement after the zero period
             np.mean(c[2:]), # average consumption excluding the first measurement
             np.std(c[2:]), # standard deviation of consumtption excluding the first measurement
             max(c[2:]), # maximum value of consumption excluding the first measurement
             no_of_consecutive_zeros_[i] # duration (number of samples) of  the zero interval
            ]
        Y.append(y)
        mask.append(True)
    else:
        mask.append(False)
Y = np.array(Y)
N = Y.shape[0]
M = len(consumption_after_zeros_[0])
M_ = Y.shape[1]

# Normalize data because the last feature has a different unit than the rest:
std1 = np.std(Y[:, 0:5])
std2 = np.std(Y[:, 5])
Y[:, 0:5] = Y[:, 0:5]/std1
Y[:, 5] = Y[:, 5]/std2

K = 2 # number of clusters

# kmeans = KMeans(n_clusters=K, random_state=0).fit(Y)
kmeans = KMedoids(n_clusters=K, random_state=0).old_fit(Y)

L = kmeans.labels_

U = np.unique(L)
C = len(U)

plt.figure()
fig, axs = plt.subplots(C, 1, sharex=True, sharey=True)
for i, u in enumerate(U):
    # axs[i].plot(np.arange(0, M_, 1), Y[L == i].transpose(), '.-', LineWidth=0.5)
    axs[i].plot(np.arange(0, M/4, .25), X[L == i].transpose(), '.-', LineWidth=0.5)
    axs[i].set_ylabel('cluster {}'.format(i+1))
    if i == 0: axs[i].set_title("feature-based clustering")
# plt.ylabel("feature")
plt.xlabel("time (hr)")