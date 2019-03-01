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

features = []

features += [f'f10{i}.pkl' for i in (2, )]
features += [f'f11{i}_{j}.pkl' for i in (1, 2)
                               for j in ('Y', 'N')]
features += [f'f12{i}.pkl' for i in (1,)]
features += [f'f13{i}.pkl' for i in (1, 2)]

features += [f'f20{i}.pkl' for i in (2, 3)]
features += [f'f21{i}_{j}.pkl' for i in (1, 2)
             for j in ('Y', 'N')]
features += [f'f23{i}.pkl' for i in (1, 2)]

# features += [f'f40{i}.pkl' for i in (2, 3)]
# features += [f'f41{i}_{j}.pkl' for i in (1, 2)
#                                for j in ('Y', 'N')]
# features += [f'f42{i}.pkl' for i in (1, 2)]

# features += [f'f50{i}.pkl' for i in (2, )]

for f in tqdm(features):
    test = pd.merge(test, pd.read_pickle(os.path.join('..', 'feature', f)), on=KEY, how='left')


cols = test.columns.values
for f in [
    'new_purchase_date_max', 'new_purchase_date_min',
    'hist_purchase_date_max', 'hist_purchase_date_min',
    'Y_hist_auth_purchase_date_max', 'Y_hist_auth_purchase_date_min',
    'N_hist_auth_purchase_date_max', 'N_hist_auth_purchase_date_min',
    'Y_new_auth_purchase_date_max', 'Y_new_auth_purchase_date_min',
    'N_new_auth_purchase_date_max', 'N_new_auth_purchase_date_min',
]:
    if f in cols:
        test[f] = test[f].astype(np.int64) * 1e-9

drop = []
columns = test.columns[1:]
for i in tqdm(range(0, len(columns))):
    ti = test[columns[i]].values
    ti_max = max(test[columns[i]].values)
    ti_min = min(test[columns[i]].values)
    for j in range(i+1, len(columns)):
        tj = test[columns[j]]
        if ti_max != max(tj) or ti_min != min(tj):
            continue
        if sum(ti != tj) == 0:
            drop.append(columns[j])

print(drop)

# =============================================================================
utils.end(__file__)
