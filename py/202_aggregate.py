#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 2018

@author: toshiki.ishikawa
"""

import os
import sys
import gc
import utils
import numpy as np
import pandas as pd

from tqdm import tqdm
# from datetime import datetime, date
import datetime
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f202'

SUMMARY = 30

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'std']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'data')

new_merchant_transactions = pd.read_csv(os.path.join(PATH, 'new_merchant_transactions.csv'))
new_merchant_transactions['installments'].replace(-1, np.nan, inplace=True)
new_merchant_transactions['installments'].replace(999, np.nan, inplace=True)
# new_merchant_transactions['purchase_amount'] = np.log1p(new_merchant_transactions['purchase_amount'] - new_merchant_transactions['purchase_amount'].min())
new_merchant_transactions['purchase_amount'] = np.round(new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)


new_merchant_transactions['purchase_date'] = pd.to_datetime(new_merchant_transactions['purchase_date'])
new_merchant_transactions['year'] = new_merchant_transactions['purchase_date'].dt.year
new_merchant_transactions['month'] = new_merchant_transactions['purchase_date'].dt.month
new_merchant_transactions['day'] = new_merchant_transactions['purchase_date'].dt.day
new_merchant_transactions['hour'] = new_merchant_transactions['purchase_date'].dt.hour
new_merchant_transactions['weekofyear'] = new_merchant_transactions['purchase_date'].dt.weekofyear
new_merchant_transactions['weekday'] = new_merchant_transactions['purchase_date'].dt.weekday
new_merchant_transactions['weekend'] = (new_merchant_transactions['purchase_date'].dt.weekday >= 5).astype(int)

new_merchant_transactions['price'] = new_merchant_transactions['purchase_amount'] / (new_merchant_transactions['installments'] + 1)

new_merchant_transactions['month_diff'] = ((datetime.date(2018, 5, 1) - new_merchant_transactions['purchase_date'].dt.date).dt.days) // 30
new_merchant_transactions['month_diff'] += new_merchant_transactions['month_lag']

new_merchant_transactions['duration'] = new_merchant_transactions['purchase_amount'] * new_merchant_transactions['month_diff']
new_merchant_transactions['amount_month_ratio'] = new_merchant_transactions['purchase_amount'] / (new_merchant_transactions['month_diff'] + 1)

new_merchant_transactions = utils.reduce_mem_usage(new_merchant_transactions)

# =============================================================================
#
# =============================================================================

def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = new_merchant_transactions.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    for c in ['new_purchase_date_max', 'new_purchase_date_min']:
        agg[c] = pd.to_datetime(agg[c]) 

    agg['new_purchase_date_diff'] = (agg['new_purchase_date_max'].dt.date - agg['new_purchase_date_min'].dt.date).dt.days
    agg['new_purchase_date_average'] = agg['new_purchase_date_diff'] / agg['new_card_id_count']
    agg['new_purchase_date_uptonow'] = (datetime.date(2018, 5, 1) - agg['new_purchase_date_max'].dt.date).dt.days
    agg['new_purchase_date_uptomin'] = (datetime.date(2018, 5, 1) - agg['new_purchase_date_min'].dt.date).dt.days

    agg.to_pickle(f'../feature/{PREF}.pkl')

    return

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {
            'prefix': 'new_',
            'key': 'card_id',
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
                'authorized_flag': ['sum', 'mean', 'std', 'skew'],
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
