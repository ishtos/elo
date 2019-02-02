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
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f102'

SUMMARY = 30

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew']

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'))
historical_transactions['purchase_amount'] = np.log1p(historical_transactions['purchase_amount'] - historical_transactions['purchase_amount'].min())

historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
# historical_transactions['year'] = historical_transactions['purchase_date'].dt.year
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

historical_transactions = utils.reduce_mem_usage(historical_transactions)

# =============================================================================
#
# =============================================================================

def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = historical_transactions.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    for c in ['hist_purchase_date_max', 'hist_purchase_date_min']:
        agg[c] = pd.to_datetime(agg[c]) 
    agg['hist_purchase_date_diff'] = (agg['hist_purchase_date_max'].dt.date - agg['hist_purchase_date_min'].dt.date).dt.days
    agg['hist_purchase_date_average'] = agg['hist_purchase_date_diff'] / agg['hist_card_id_size']
    agg['hist_purchase_date_uptonow'] = (datetime.date(2018, 4, 30) - agg['hist_purchase_date_max'].dt.date).dt.days
    agg['hist_purchase_date_uptomin'] = (datetime.date(2018, 4, 30) - agg['hist_purchase_date_min'].dt.date).dt.days

    agg.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

    return

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {   
            'prefix': 'hist_',
            'key': 'card_id',
            'num_aggregations': {
                'subsector_id': ['nunique'],
                'merchant_id': ['nunique'],
                'merchant_category_id': ['nunique'],

                # 'year': ['nunique'],
                'month': ['nunique', 'mean', 'var'],
                'hour': ['nunique', 'mean', 'min', 'max'],
                'weekofyear': ['nunique', 'mean', 'min', 'max'],
                'day': ['nunique', 'mean'],
                'weekday': ['mean'],
                'weekend': ['mean'],

                'purchase_amount': ['sum', 'max', 'min', 'mean', 'var', 'skew'],
                'installments': ['max', 'mean', 'var', 'skew'], # 'sum'
                'purchase_date': ['max', 'min'],
                'month_lag': ['max', 'min', 'mean', 'var', 'skew'], # 'max', 'min', 
                'month_diff': ['max', 'min', 'mean', 'var', 'skew'], # 'max', 'min'
                'authorized_flag': ['sum', 'mean'],
                'category_1': ['mean'],
                'category_2': ['nunique'], # 'mean'
                'category_3': ['nunique'], # 'mean'
                'card_id': ['size', 'count'],
                'price': ['sum', 'mean', 'max', 'min', 'var'],
              
                'duration': ['mean','min','max','var','skew'],
                'amount_month_ratio': ['mean','min','max','var','skew'],
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
