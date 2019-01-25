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
PATH = os.path.join('..', 'remove_outlier_data')

new_merchant_transactions = pd.read_csv(os.path.join(PATH, 'new_merchant_transactions.csv'))

new_merchant_transactions['purchase_date'] = pd.to_datetime(new_merchant_transactions['purchase_date'])
new_merchant_transactions['purchase_month'] = new_merchant_transactions['purchase_date'].dt.month
new_merchant_transactions['month_diff'] = (datetime.date(2018, 2, 1) - new_merchant_transactions['purchase_date'].dt.date).dt.days // SUMMARY  # TODO: change today
new_merchant_transactions['month_diff'] += new_merchant_transactions['month_lag']
new_merchant_transactions['installments'] = new_merchant_transactions['installments'].astype(int)
# new_merchant_transactions.loc[:, 'purchase_date'] = pd.DatetimeIndex(new_merchant_transactions['purchase_date']).astype(np.int64) * 1e-9

# =============================================================================
#
# =============================================================================


def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = new_merchant_transactions.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    df = new_merchant_transactions.groupby('card_id').size().reset_index(name='{}transactions_count'.format(prefix))
    df = pd.merge(df, agg, on='card_id', how='left')
    df['new_purchase_date_diff'] = (df['new_purchase_date_max'] - df['new_purchase_date_min']).dt.days
    df['new_purchase_date_average'] = df['new_purchase_date_diff'] / df['new_transactions_count']
    df['new_purchase_date_uptonow'] = (datetime.date(2018, 3, 1) - df['new_purchase_date_max'].dt.date).dt.days

    df.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

    return

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {
            'prefix': 'new_',
            'key': ['card_id'],
            'num_aggregations': {
                'category_1': ['sum', 'mean'],
                'category_2': ['nunique'],
                'category_3': ['nunique'],

                'merchant_id': ['nunique'],
                'state_id': ['nunique'],
                'subsector_id': ['nunique'],
                'city_id': ['nunique'],
                'merchant_category_id': ['nunique'],

                'installments': ['nunique', 'mean', 'std'],

                'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
                'purchase_month': ['median', 'max', 'min', 'std'],
                'purchase_date': ['max', 'min'],
                'month_diff': ['median', 'max', 'min', 'std'],
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
