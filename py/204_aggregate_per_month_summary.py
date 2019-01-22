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
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f204'

SUMMARY = 30

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', 'count']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'data')

# train = pd.read_csv(os.path.join(PATH, 'train.csv.gz'))[[KEY]]
# test = pd.read_csv(os.path.join(PATH, 'test.csv.gz'))[[KEY]]

new_merchant_transactions = pd.read_csv(os.path.join(PATH, 'new_merchant_transactions.csv'))
new_merchant_transactions['installments'] = new_merchant_transactions['installments'].astype(int)

new_merchant_transactions['purchase_date'] = pd.to_datetime(new_merchant_transactions['purchase_date'])
new_merchant_transactions['month_diff'] = (datetime.today() - new_merchant_transactions['purchase_date']).dt.days // SUMMARY  # TODO: change today
new_merchant_transactions['month_diff'] += new_merchant_transactions['month_lag']

# =============================================================================
#
# =============================================================================
def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    grouped = new_merchant_transactions.groupby(key)
    agg = grouped.agg(num_aggregations)
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    agg = agg.groupby('card_id').agg(['mean', 'std', 'var', 'skew'])
    agg.columns = [prefix+'_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    agg = agg.rename(columns={prefix+KEY: KEY})

    agg.to_pickle(f'../feature/{PREF}.pkl')

    return

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {
            'prefix': 'new_',
            'key': ['card_id', 'month_diff'],
            'num_aggregations': {
                'purchase_amount': stats,
                'installments': stats
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
