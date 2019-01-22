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


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f302'

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'std']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

merchants = pd.read_csv(os.path.join(PATH, 'merchants.csv'))
historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'), usecols=['card_id', 'merchant_id', 'purchase_date', 'month_lag'])

merchants = merchants.drop_duplicates(subset=['merchant_id'], keep='first').reset_index(drop=True) # TODO: change first
historical_transactions = pd.merge(historical_transactions, merchants, on='merchant_id', how='left')

del merchants
gc.collect()

# =============================================================================
#
# =============================================================================
def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = historical_transactions.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    df = historical_transactions.groupby('card_id').size().reset_index(name='{}transactions_count'.format(prefix))

    df = pd.merge(df, agg, on='card_id', how='left')
    df.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

    return

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {
            'prefix': 'historica_merchants_',
            'key': ['card_id'],
            'num_aggregations': {
                'merchant_group_id': ['nunique'],
                'numerical_1': stats,
                'numerical_2': stats,
                'category_1': ['sum', 'median'],
                'category_2': ['sum', 'median'],
                'category_4': ['sum', 'median'],
                'city_id': ['nunique'],
                'state_id': ['nunique']
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
