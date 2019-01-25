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

PREF = 'f404'

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'data')

train = pd.read_csv(os.path.join(PATH, 'train.csv'))[[KEY]]
test = pd.read_csv(os.path.join(PATH, 'test.csv'))[[KEY]]

union = pd.read_csv(os.path.join(PATH, 'union.csv'))

union['purchase_date'] = pd.to_datetime(union['purchase_date'])
union['purchase_month'] = union['purchase_date'].dt.month
union['year'] = union['purchase_date'].dt.year
union['weekofyear'] = union['purchase_date'].dt.weekofyear
union['month'] = union['purchase_date'].dt.month
union['dayofweek'] = union['purchase_date'].dt.dayofweek
union['weekend'] = (union['purchase_date'].dt.weekday >= 5).astype(int)
union['hour'] = union['purchase_date'].dt.hour
union['month_diff'] = (datetime.today() - union['purchase_date']).dt.days // 30  # TODO: change today
union['month_diff'] += union['month_lag']
union['installments'] = union['installments'].astype(int)

union.loc[:, 'purchase_date'] = pd.DatetimeIndex(
    union['purchase_date']).astype(np.int64) * 1e-9
union = pd.get_dummies(union, columns=['category_2', 'category_3'])

# =============================================================================
#
# =============================================================================


def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = union.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip()
                   for col in agg.columns.values]
    agg.reset_index(inplace=True)

    df = union.groupby('card_id').size().reset_index(
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
            'prefix': 'union_',
            'key': ['card_id'],
            'num_aggregations': {
                'year': ['nunique'],
                'weekofyear': ['nunique'],
                'month': ['nunique'],
                'dayofweek': ['nunique'],
                'weekend': ['nunique'],
                'hour': ['nunique'],
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)