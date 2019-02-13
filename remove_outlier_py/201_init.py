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

PREF = 'f201'

KEY = 'card_id'

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')
# new_merchant_transactions = new_merchant_transactions.query('purchase_amount < 80')

# =============================================================================
#
# =============================================================================
# new_merchant_transactions['purchase_amount'] = new_merchant_transactions['purchase_amount'].apply(lambda x: min(x, 0.8))

new_merchant_transactions['category_2'].fillna(1.0, inplace=True)
new_merchant_transactions['category_3'].fillna('A', inplace=True)
new_merchant_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)

new_merchant_transactions['authorized_flag'] = new_merchant_transactions['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
new_merchant_transactions['category_1'] = new_merchant_transactions['category_1'].map({'Y': 1, 'N': 0}).astype(int)
new_merchant_transactions['category_2'] = new_merchant_transactions['category_2'].astype(int)
new_merchant_transactions['category_3'] = new_merchant_transactions['category_3'].map({'A':0, 'B':1, 'C':2}).astype(int)

new_merchant_transactions = utils.reduce_mem_usage(new_merchant_transactions )
new_merchant_transactions.to_csv(os.path.join('..', 'remove_outlier_data', 'new_merchant_transactions.csv'), index=False)

# =============================================================================
#
# =============================================================================
# new_merchant_transactions.loc[new_merchant_transactions.installments == 999, 'installments'] = -1

# new_merchant_transactions['authorized_flag'] = new_merchant_transactions['authorized_flag'].apply(lambda x: np.where(x == 'Y', 1, 0))
# new_merchant_transactions['installments'] = new_merchant_transactions['installments'].astype('category')

# map_dict = {'Y': 0, 'N': 1}
# new_merchant_transactions['category_1'] = new_merchant_transactions['category_1'].apply(lambda x: map_dict[x]).astype('category')
# map_dict = {'A': 0, 'B': 1, 'C': 2, 'nan': 3}
# new_merchant_transactions['category_3'] = new_merchant_transactions['category_3'].apply(lambda x: map_dict[str(x)]).astype('category')

# new_merchant_transactions = utils.reduce_mem_usage(new_merchant_transactions )
# new_merchant_transactions.to_csv(os.path.join('..', 'remove_outlier_data', 'new_merchant_transactions.csv'), index=False)

# =============================================================================
utils.end(__file__)
