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

PREF = 'f212'

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'std']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

new_merchant_transactions = pd.read_csv(os.path.join(PATH, 'new_merchant_transactions.csv'))
new_merchant_transactions['installments'] = new_merchant_transactions['installments'].astype(int)
new_merchant_transactions = new_merchant_transactions.query('0 <= installments and installments <= 12')

# =============================================================================
#
# =============================================================================
def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    grouped = new_merchant_transactions.groupby(key)
    agg = grouped.agg(num_aggregations)
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    agg_Y = agg[agg.authorized_flag == 1].add_prefix('Y_')
    agg_Y = agg_Y.rename(columns={'Y_card_id': 'card_id'})
    del agg_Y['Y_authorized_flag']

    agg_Y = agg_Y.groupby('card_id').agg(['mean', 'std'])
    agg_Y.columns = ['new_'+'_'.join(col).strip()
                     for col in agg_Y.columns.values]
    agg_Y = agg_Y.reset_index()
    agg_Y = agg_Y.rename(columns={'hist_card_id': 'card_id'})

    agg_N = agg[agg.authorized_flag == 0].add_prefix('N_')
    agg_N = agg_N.rename(columns={'N_card_id': 'card_id'})
    del agg_N['N_authorized_flag']

    agg_N = agg_N.groupby('card_id').agg(['mean', 'std'])
    agg_N.columns = ['new_'+'_'.join(col).strip() for col in agg_N.columns.values]
    agg_N = agg_N.reset_index()
    agg_N = agg_N.rename(columns={'hist_card_id': 'card_id'})

    agg_Y.to_pickle(f'../remove_outlier_feature/{PREF}_Y.pkl')
    agg_N.to_pickle(f'../remove_outlier_feature/{PREF}_N.pkl')

    return


# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {
            'prefix': 'new_auth_',
            'key': ['card_id', 'month_lag', 'authorized_flag'],
            'num_aggregations': {
                'purchase_amount': stats,
                'installments': ['mean', 'sum', 'std']
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
