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

PREF = 'f411'

SUMMARY = 30

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

union = pd.read_csv(
    os.path.join(PATH, 'union.csv'))

union['purchase_date'] = pd.to_datetime(
    union['purchase_date'])
union['purchase_month'] = union['purchase_date'].dt.month
union['month_diff'] = (datetime.date(2018, 3, 1) - union['purchase_date'].dt.date).dt.days // SUMMARY  # TODO: change today
union['month_diff'] += union['month_lag']
union['installments'] = union['installments'].astype(
    int)
# union.loc[:, 'purchase_date'] = pd.DatetimeIndex(union['purchase_date']).astype(np.int64) * 1e-9

# =============================================================================
#
# =============================================================================


def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = union.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    df = union.groupby('card_id').size().reset_index(name='{}transactions_count'.format(prefix))
    df = pd.merge(df, agg, on='card_id', how='left')

    df_Y = df[df.authorized_flag == 1].add_prefix('Y_')
    df_Y = df_Y.rename(columns={'Y_card_id': 'card_id'})
    del df_Y['Y_authorized_flag'], df_Y['Y_union_auth_transactions_count']

    df_N = df[df.authorized_flag == 0].add_prefix('N_')
    df_N = df_N.rename(columns={'N_card_id': 'card_id'})
    del df_N['N_authorized_flag'], df_N['N_union_auth_transactions_count']

    df_Y.to_pickle(f'../remove_outlier_feature/{PREF}_Y.pkl')
    df_N.to_pickle(f'../remove_outlier_feature/{PREF}_N.pkl')

    return


# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {
            'prefix': 'union_auth_',
            'key': ['card_id', 'authorized_flag'],
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

                'purchase_month': ['mean'],
                'month_diff': ['mean', 'std'],
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
