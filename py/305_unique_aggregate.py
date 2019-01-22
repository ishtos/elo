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

PREF = 'f305'

KEY = 'card_id'

stats = ['nunique', 'min', 'max', 'mean', 'median', 'std', 'var', 'skew']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'data')

merchants = pd.read_csv(os.path.join(PATH, 'merchants.csv'))
merchants = merchants[['merchant_id', 'most_recent_sales_range', 'most_recent_purchases_range',
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
def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = union_base.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip()
                   for col in agg.columns.values]
    agg.reset_index(inplace=True)

    df = union_base.groupby('card_id').size().reset_index(
        name='{}transactions_count'.format(prefix))

    df = pd.merge(df, agg, on='card_id', how='left')
    df.to_pickle(f'../feature/{PREF}.pkl')

    return

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {
            'prefix': 'merchants_',
            'key': ['card_id'],
            'num_aggregations': {
                'most_recent_sales_range': ['nunique'], 
                'most_recent_purchases_range': ['nunique'], 
                'avg_sales_lag3': stats,
                'avg_purchases_lag3': stats,
                'active_months_lag3': stats,
                'avg_sales_lag6': stats,
                'avg_purchases_lag6': stats,
                'active_months_lag6': stats,
                'avg_sales_lag12': stats,
                'avg_purchases_lag12': stats,
                'active_months_lag12': stats
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
