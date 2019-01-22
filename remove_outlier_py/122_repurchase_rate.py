#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 2018

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


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f122'

KEY = 'card_id'

# stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', 'count']
stats = ['min', 'max', 'mean', 'std', 'count']


# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'), usecols=['purchase_date', 'merchant_id', 'card_id'])

historical_transactions.purchase_date = pd.to_datetime(historical_transactions.purchase_date)
historical_transactions['month'] = historical_transactions.purchase_date.dt.month
historical_transactions['year'] = historical_transactions.purchase_date.dt.year
historical_transactions = historical_transactions[['card_id', 'merchant_id', 'month', 'year']]
historical_transactions['cnt'] = 1

hg = historical_transactions.groupby(['card_id', 'merchant_id', 'month', 'year'])['cnt'].count().reset_index()
hgg = hg.groupby(['card_id', 'month', 'year']).agg({'cnt': stats})
del hg
gc.collect()

hgg = hgg.reset_index()
hgg.columns = [f'{c[0]}_{c[1]}'.strip('_') for c in hgg.columns]
pt = hgg.pivot_table(index='card_id',
                     columns=['month', 'year'],
                     values=['cnt_min', 'cnt_max',
                             'cnt_mean', 'cnt_std',
                             'cnt_count'])

del hgg
gc.collect()

pt.columns = [f'{c[0]}_{c[1]}_{c[2]}' for c in pt.columns]
pt = pt.reset_index()
pt = pt.fillna(0)

pt.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

#==============================================================================
utils.end(__file__)
