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

PREF = 'f108'

SUMMARY = 30

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

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'))

historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month
historical_transactions['year'] = historical_transactions['purchase_date'].dt.year
historical_transactions['weekofyear'] = historical_transactions['purchase_date'].dt.weekofyear
historical_transactions['month'] = historical_transactions['purchase_date'].dt.month
historical_transactions['dayofweek'] = historical_transactions['purchase_date'].dt.dayofweek
historical_transactions['weekend'] = (historical_transactions['purchase_date'].dt.weekday >= 5).astype(int)
historical_transactions['hour'] = historical_transactions['purchase_date'].dt.hour
historical_transactions['month_diff'] = (datetime.today() - historical_transactions['purchase_date']).dt.days // SUMMARY  # TODO: change today
historical_transactions['month_diff'] += historical_transactions['month_lag']
historical_transactions['installments'] = historical_transactions['installments'].astype(int)
# historical_transactions.loc[:, 'purchase_date'] = pd.DatetimeIndex(historical_transactions['purchase_date']).astype(np.int64) * 1e-9

# =============================================================================
#
# =============================================================================


def aggregate(args):
    PREF, prefix, key, num_aggregations = args['feature'], args['prefix'], args['key'], args['num_aggregations']

    agg = historical_transactions.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip()
                   for col in agg.columns.values]
    agg.reset_index(inplace=True)

    df = historical_transactions.groupby('card_id').size().reset_index(name='{}transactions_count'.format(prefix))
    df = pd.merge(df, agg, on='card_id', how='left')
    df['hist_auth_purchase_date_diff'] = (df['hist_auth_purchase_date_max'] - df['hist_auth_purchase_date_min']).dt.days
    df['hist_auth_purchase_date_average'] = df['hist_auth_purchase_date_diff']/df['hist_auth_transactions_count']
    df['hist_auth_purchase_date_uptonow'] = (datetime.today() - df['hist_auth_purchase_date_max']).dt.days
    df['hist_auth_purchase_date_ptp'] = df['hist_auth_purchase_date_ptp'].dt.days
    
    df_Y = df[df.authorized_flag == 1].add_prefix('Y_')
    df_Y = df_Y.rename(columns={'Y_card_id': 'card_id'})
    df_N = df[df.authorized_flag == 0].add_prefix('N_')
    df_N = df_N.rename(columns={'N_card_id': 'card_id'})

    df_Y.to_pickle(f'../feature/{PREF}_Y.pkl')
    df_N.to_pickle(f'../feature/{PREF}_N.pkl')

    return

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {   
            'feature': 'f108',
            'prefix': 'hist_auth_',
            'key': ['card_id', 'authorized_flag'],
            'num_aggregations': {
                # 'card_id': ['size'],
                'category_1': ['sum', 'mean', 'nunique'],
                'category_2': ['nunique'],
                'category_3': ['nunique'],
                'merchant_id': ['nunique'],
                'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
                'installments': ['sum', 'mean', 'max', 'min', 'std'],
                'purchase_month': ['mean', 'max', 'min', 'std'],
                'purchase_date': [np.ptp, 'max', 'min'],
                'month_lag': ['mean', 'max', 'min', 'std'],
                'merchant_category_id': ['nunique'],
                'state_id': ['nunique'],
                'subsector_id': ['nunique'],
                'city_id': ['nunique'],
                'month_diff': ['mean', 'max', 'min', 'std'],
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
