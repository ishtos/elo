#!/usr/bin/env python3
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

PREF = 'f101'

KEY = 'card_id'

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

historical_transactions = pd.read_csv('../input/historical_transactions.csv')
historical_transactions = historical_transactions.query('purchase_amount < 6000000')

# =============================================================================
#
# =============================================================================
# historical_transactions['purchase_amount'] = historical_transactions['purchase_amount'].apply(lambda x: min(x, 0.8))

historical_transactions['category_2'].fillna(1.0, inplace=True)
historical_transactions['category_3'].fillna('A', inplace=True)
historical_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)

historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
historical_transactions['category_1'] = historical_transactions['category_1'].map({'Y': 1, 'N': 0}).astype(int)
historical_transactions['category_2'] = historical_transactions['category_2'].astype(int)
historical_transactions['category_3'] = historical_transactions['category_3'].map({'A':0, 'B':1, 'C':2}).astype(int)

historical_transactions = utils.reduce_mem_usage(historical_transactions)
historical_transactions.to_csv(os.path.join('..', 'data', 'historical_transactions.csv'), index=False)

# =============================================================================
#
# =============================================================================
# historical_transactions.loc[historical_transactions.installments == 999, 'installments'] = -1
# historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].apply(lambda x: np.where(x == 'Y', 1, 0))
# historical_transactions['installments'] = historical_transactions['installments'].astype('category')

# map_dict = {'Y': 0, 'N': 1}
# historical_transactions['category_1'] = historical_transactions['category_1'].apply(lambda x: map_dict[x]).astype('category')
# map_dict = {'A': 0, 'B': 1, 'C': 2, 'nan': 3}
# historical_transactions['category_3'] = historical_transactions['category_3'].apply(lambda x: map_dict[str(x)]).astype('category')

# historical_transactions = utils.reduce_mem_usage(historical_transactions)
# historical_transactions.to_csv(os.path.join('..', 'data', 'historical_transactions.csv'), index=False)

# =============================================================================
utils.end(__file__)
