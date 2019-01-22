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

PREF = 'f407'

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

union = pd.read_csv(os.path.join(PATH, 'union.csv'), usecols=['card_id', 'month_lag', 'purchase_amount', 'installments', 'authorized_flag'])
union['installments'] = union['installments'].astype(int)

grouped_Y = union[union.authorized_flag == 1]
grouped_Y = grouped_Y.drop('authorized_flag', axis=1)
grouped_Y = grouped_Y.groupby(['card_id', 'month_lag'])

grouped_N = union[union.authorized_flag == 0]
grouped_N = grouped_N.drop('authorized_flag', axis=1)
grouped_N = grouped_N.groupby(['card_id', 'month_lag'])

# =============================================================================
#
# =============================================================================
def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = grouped_Y.agg(num_aggregations)
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    agg = agg.groupby('card_id').agg(['mean', 'std', 'var', 'skew'])
    agg.columns = [prefix+'Y_'+'_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    agg = agg.rename(columns={prefix+KEY: KEY})

    agg.to_pickle(f'../feature/{PREF}_Y.pkl')

    agg = grouped_N.agg(num_aggregations)
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    agg = agg.groupby('card_id').agg(['mean', 'std', 'var', 'skew'])
    agg.columns = [prefix+'N_'+'_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    agg = agg.rename(columns={prefix+KEY: KEY})

    agg.to_pickle(f'../feature/{PREF}_N.pkl')

    return

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {
            'prefix': 'union_',
            'key': ['card_id'],
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
