# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 2018

@author: toshiki.ishikawa
"""

import os
import sys
import gc
import utils
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

PATH = os.path.join('..', 'feature')

KEY = 'card_id'

# =============================================================================
#
# =============================================================================
files = os.listdir(PATH)

test = pd.read_csv(os.path.join('..', 'data', 'test.csv'))

features = ['f105.pkl', 'f109.pkl', 'f111.pkl']
features += ['f107_N.pkl', 'f107_Y.pkl',
             'f108_N.pkl', 'f108_Y.pkl',
             'f110_N.pkl', 'f110_Y.pkl']
features += ['f205.pkl', 'f209.pkl']
features += ['f207_N.pkl', 'f207_Y.pkl',
             'f208_N.pkl', 'f208_Y.pkl',
             'f210_N.pkl', 'f210_Y.pkl']
features += ['f302.pkl', 'f303.pkl', 'f304.pkl', 'f305.pkl', 'f306.pkl']
features += ['f403.pkl', 'f404.pkl', 'f409.pkl', 'f411.pkl']
features += ['f406_N.pkl', 'f406_Y.pkl',
             'f407_N.pkl', 'f407_Y.pkl',
             'f408_N.pkl', 'f408_Y.pkl']
# features += ['f501.pkl']

for f in features:
    print(f'Merge: {f}', end=' ')
    test = pd.merge(test, pd.read_pickle(os.path.join('..', 'feature', f)), on=KEY, how='left')
    print('Done!!')

drop_col = [
    'N_authorized_flag_x',
    'Y_authorized_flag_x',
    'N_authorized_flag_y',
    'Y_authorized_flag_y',
    'union_transactions_count_x',
    'union_transactions_count_y',
]

test = test.drop(drop_col, axis=1)

for f in [
    'hist_purchase_date_max', 'hist_purchase_date_min',
    'N_hist_auth_purchase_date_max', 'N_hist_auth_purchase_date_min',
    'Y_hist_auth_purchase_date_max', 'Y_hist_auth_purchase_date_min',
    'new_purchase_date_max', 'new_purchase_date_min',
    'N_new_auth_purchase_date_max', 'N_new_auth_purchase_date_min',
    'Y_new_auth_purchase_date_max', 'Y_new_auth_purchase_date_min',
    'union_purchase_date_max', 'union_purchase_date_min',
    'N_union_auth_purchase_date_max', 'N_union_auth_purchase_date_min',
    'Y_union_auth_purchase_date_max', 'Y_union_auth_purchase_date_min']:
    test[f] = test[f].astype(np.int64) * 1e-9

drop = []
columns = test.columns[1:]
for i in tqdm(range(0, len(columns))):
    for j in range(i+1, len(columns)):
        if sum(test[columns[i]] != test[columns[j]]) == 0:
            drop.append(columns[j])

print(drop)

# =============================================================================
utils.end(__file__)
