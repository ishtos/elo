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
# from datetime import datetime, dat
import datetime
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f111'

SUMMARY = 30

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'data')

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'))
historical_transactions['purchase_amount'] = np.log1p(historical_transactions['purchase_amount'] - historical_transactions['purchase_amount'].min())

historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
historical_transactions['year'] = historical_transactions['purchase_date'].dt.year
historical_transactions['month'] = historical_transactions['purchase_date'].dt.month
historical_transactions['day'] = historical_transactions['purchase_date'].dt.day
historical_transactions['hour'] = historical_transactions['purchase_date'].dt.hour
historical_transactions['weekofyear'] = historical_transactions['purchase_date'].dt.weekofyear
historical_transactions['weekday'] = historical_transactions['purchase_date'].dt.weekday
historical_transactions['weekend'] = (historical_transactions['purchase_date'].dt.weekday >= 5).astype(int)

historical_transactions['price'] = historical_transactions['purchase_amount'] / (historical_transactions['installments'] + 1)

historical_transactions['month_diff'] = ((datetime.date(2018, 4, 30) - historical_transactions['purchase_date'].dt.date).dt.days) // 30
historical_transactions['month_diff'] += historical_transactions['month_lag']

historical_transactions['duration'] = historical_transactions['purchase_amount'] * historical_transactions['month_diff']
historical_transactions['amount_month_ratio'] = historical_transactions['purchase_amount'] / historical_transactions['month_diff']

# historical_transactions = utils.reduce_mem_usage(historical_transactions)

# =============================================================================
#
# =============================================================================
def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = historical_transactions.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    for c in ['hist_auth_purchase_date_max', 'hist_auth_purchase_date_min']:
        agg[c] = pd.to_datetime(agg[c]) 
    agg['hist_auth_purchase_date_diff'] = (agg['hist_auth_purchase_date_max'].dt.date - agg['hist_auth_purchase_date_min'].dt.date).dt.days
    agg['hist_auth_purchase_date_average'] = agg['hist_auth_purchase_date_diff'] / agg['hist_auth_card_id_count']
    agg['hist_auth_purchase_date_uptonow'] = (datetime.date(2018, 4, 30) - agg['hist_auth_purchase_date_max'].dt.date).dt.days
    agg['hist_auth_purchase_date_uptomin'] = (datetime.date(2018, 4, 30) - agg['hist_auth_purchase_date_min'].dt.date).dt.days
   
    agg_Y = agg[agg.authorized_flag == 1].add_prefix('Y_')
    agg_Y = agg_Y.rename(columns={'Y_card_id': 'card_id'})

    agg_N = agg[agg.authorized_flag == 0].add_prefix('N_')
    agg_N = agg_N.rename(columns={'N_card_id': 'card_id'})

    agg_Y.to_pickle(f'../feature/{PREF}_Y.pkl')
    agg_N.to_pickle(f'../feature/{PREF}_N.pkl')

    return

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {   
            'prefix': 'hist_auth_',
            'key': ['card_id', 'authorized_flag'],
            'num_aggregations': {
                'subsector_id': ['nunique'],
                'merchant_id': ['nunique'],
                'merchant_category_id': ['nunique'],

                'year': ['nunique'],
                'month': ['nunique', 'mean', 'min', 'max'],
                'hour':  ['nunique', 'mean', 'min', 'max'],
                'weekofyear': ['nunique', 'mean', 'min', 'max'],
                'day':  ['nunique', 'mean', 'min', 'max'],
                'weekday': ['nunique', 'mean', 'min', 'max'], # 'std'
                'weekend': ['mean', 'sum'], # 'sum', 'std'

                'purchase_amount': ['sum', 'max', 'min', 'mean', 'std', 'skew'],
                'installments': ['sum', 'max', 'min', 'mean', 'std', 'skew'], # 'sum'
                'purchase_date': ['max', 'min'],
                'month_lag': ['max', 'min', 'mean', 'std', 'skew'],
                'month_diff': ['mean', 'std', 'skew'],
                'category_1': ['mean'],
                'category_2': ['nunique', 'mean', 'std'], # 'mean'
                'category_3': ['nunique', 'mean', 'std'], # 'mean'
                'card_id': ['count'],
                'price': ['sum', 'mean', 'max', 'min', 'std', 'skew'], # 'skew'
              
                'duration': ['sum', 'max', 'min', 'mean', 'std', 'skew'], 
                'amount_month_ratio': ['sum', 'max', 'min', 'mean', 'std', 'skew'], 
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)