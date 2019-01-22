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
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f206'

SUMMARY = 30

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', 'count']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
train = pd.read_csv('../data/train.csv')[[KEY]]
test = pd.read_csv('../data/test.csv')[[KEY]]

new_merchant_transactions = pd.read_csv(
    '../data/new_merchant_transactions.csv')

# TODO: check feature about time
new_merchant_transactions['purchase_date'] = pd.to_datetime(
    new_merchant_transactions['purchase_date'])
new_merchant_transactions['purchase_month'] = new_merchant_transactions['purchase_date'].dt.month
new_merchant_transactions['year'] = new_merchant_transactions['purchase_date'].dt.year
new_merchant_transactions['weekofyear'] = new_merchant_transactions['purchase_date'].dt.weekofyear
new_merchant_transactions['month'] = new_merchant_transactions['purchase_date'].dt.month
new_merchant_transactions['dayofweek'] = new_merchant_transactions['purchase_date'].dt.dayofweek
new_merchant_transactions['weekend'] = (
    new_merchant_transactions['purchase_date'].dt.weekday >= 5).astype(int)
new_merchant_transactions['hour'] = new_merchant_transactions['purchase_date'].dt.hour
new_merchant_transactions['month_diff'] = (datetime.today(
) - new_merchant_transactions['purchase_date']).dt.days // SUMMARY  # TODO: change today
new_merchant_transactions['month_diff'] += new_merchant_transactions['month_lag']
new_merchant_transactions['installments'] = new_merchant_transactions['installments'].astype(
    int)

new_merchant_transactions.loc[:, 'purchase_date'] = pd.DatetimeIndex(
    new_merchant_transactions['purchase_date']).astype(np.int64) * 1e-9
new_merchant_transactions = pd.get_dummies(
    new_merchant_transactions, columns=['category_2', 'category_3'])
new_merchant_transactions['installments'] = new_merchant_transactions['installments'].astype(
    int)

# =============================================================================
#
# =============================================================================


def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = new_merchant_transactions.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip()
                   for col in agg.columns.values]
    agg.reset_index(inplace=True)

    df = new_merchant_transactions.groupby('card_id').size(
    ).reset_index(name='{}transactions_count'.format(prefix))

    df = pd.merge(df, agg, on='card_id', how='left')

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
            'prefix': 'new_',
            'key': ['card_id', 'authorized_flag'],
            'num_aggregations': {
                # 'card_id': ['size'],
                'category_1': ['sum', 'mean'],
                'category_2_1.0': ['sum', 'mean'],
                'category_2_2.0': ['sum', 'mean'],
                'category_2_3.0': ['sum', 'mean'],
                'category_2_4.0': ['sum', 'mean'],
                'category_2_5.0': ['sum', 'mean'],
                'category_3_0': ['sum', 'mean'],
                'category_3_1': ['sum', 'mean'],
                'category_3_2': ['sum', 'mean'],
                'category_3_3': ['sum', 'mean'],
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
