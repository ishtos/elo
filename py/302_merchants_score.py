#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 2018

@author: toshiki.ishikawa
"""

import os
import sys
import gc
import utils
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


def subtraction(x):
    return x[0] - x[1]


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f302'

KEY = 'card_id'

stats = ['sum', 'std']

PATH = os.path.join('..', 'data')

# =============================================================================
#
# =============================================================================
use_cols = ['merchant_id', 'numerical_1', 'numerical_2',
            'most_recent_sales_range', 'most_recent_purchases_range',
            'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
            'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
            'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']
merchants = pd.read_csv(os.path.join(PATH, 'merchants.csv'), usecols=use_cols)
merchants = merchants.drop_duplicates(subset=['merchant_id'], keep='first').reset_index(drop=True)
merchants['numerical_1'] = np.round(merchants['numerical_1'] / 0.009914905 + 5.79639, 0)
merchants['numerical_2'] = np.round(merchants['numerical_2'] / 0.009914905 + 5.79639, 0)
historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'), usecols=['card_id', 'merchant_id'])
merchants = pd.merge(historical_transactions, merchants, on='merchant_id', how='left')

del historical_transactions
gc.collect()

# =============================================================================
#
# =============================================================================
merchants['sum_numerical'] = merchants['numerical_1'] + merchants['numerical_2']
for i in (3, 6, 12):
    merchants[f'avg_rate_lag{i}'] = merchants[f'avg_sales_lag{i}'] / (merchants[f'avg_purchases_lag{i}'] + 1e-9)

num_aggregations = {
    # 'most_recent_sales_range': 'mean', 
    # 'most_recent_purchases_range': 'mean', 
    'numerical_1': ['nunique', 'std', 'min', 'max'],
    'numerical_2': ['nunique', 'std', 'min', 'max'],
    'avg_rate_lag3': stats,
    'avg_rate_lag6': stats,
    'avg_rate_lag12': stats,
    'active_months_lag3': stats,
    'active_months_lag6': stats,
    'active_months_lag12': stats,
}

agg = merchants.groupby('card_id').agg(num_aggregations).reset_index()
agg.columns = [f'{c[0]}_{c[1]}'.strip('_') for c in agg.columns]
agg.to_pickle(f'../feature/{PREF}.pkl')

#==============================================================================
utils.end(__file__)


