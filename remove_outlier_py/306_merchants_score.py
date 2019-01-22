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

PREF = 'f306'

KEY = 'card_id'

stats = ['nunique', 'min', 'max', 'mean', 'median', 'std', 'var', 'skew']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

merchants = pd.read_csv(os.path.join(PATH, 'merchants.csv'))
merchants = merchants[['merchant_id', 
                       'numerical_1', 'numerical_2',
                       'most_recent_sales_range', 'most_recent_purchases_range',
                       'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
                       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
                       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']]
union_base = pd.read_csv(os.path.join(PATH, 'union_base.csv'))
union_base = pd.merge(union_base, merchants, on='merchant_id', how='left')

del merchants
gc.collect()

# =============================================================================
#
# =============================================================================
columns = ['numerical_1', 'numerical_2']
columns3 = ['avg_sales_lag3', 'avg_purchases_lag3']
columns6 = ['avg_sales_lag6', 'avg_purchases_lag6']
columns12 = ['avg_sales_lag12', 'avg_purchases_lag12']
union_base['score_numerical'] = union_base[columns].apply(np.sum, axis=1)
union_base['score_log3'] = union_base[columns3].apply(subtraction, axis=1)
union_base['score_log6'] = union_base[columns6].apply(subtraction, axis=1)
union_base['score_log12'] = union_base[columns12].apply(subtraction, axis=1)

num_aggregations = {
    'score_numerical': stats,
    'score_log3': stats,
    'score_log6': stats,
    'score_log12': stats,
}

pt = union_base.groupby('card_id').agg(num_aggregations).reset_index()
pt.columns = [f'{c[0]}_{c[1]}' for c in pt.columns]
pt = pt.rename(columns={'card_id_': 'card_id'})

pt.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

#==============================================================================
utils.end(__file__)


