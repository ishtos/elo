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
# from datetime import datetime, date
import datetime
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f131'

SUMMARY = 30

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew']

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'))
historical_transactions['purchase_amount'] = np.log1p(historical_transactions['purchase_amount'] - historical_transactions['purchase_amount'].min())

# =============================================================================
#
# =============================================================================


def aggregate(args):
    prefix, index, columns, values = args['prefix'], args['index'], args['columns'], args['values']

    pt = historical_transactions.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc=['sum', 'count'])

    pt = pt.fillna(0).reset_index()
    pt.columns = [f'{c[0]}_{c[1]}_{c[2]}'.strip('_').replace('-', '') for c in pt.columns]
    pt = pt.add_prefix(prefix)
    pt = pt.rename(columns={prefix+KEY: KEY})

    use_cols = ['card_id']
    
    cols = [
        'hist_sum_purchase_amount_13', 'hist_sum_purchase_amount_12',
        'hist_sum_purchase_amount_11', 'hist_sum_purchase_amount_10',
        'hist_sum_purchase_amount_9', 'hist_sum_purchase_amount_8',
        'hist_sum_purchase_amount_7', 'hist_sum_purchase_amount_6',
        'hist_sum_purchase_amount_5', 'hist_sum_purchase_amount_4',
        'hist_sum_purchase_amount_3', 'hist_sum_purchase_amount_2',
        'hist_sum_purchase_amount_1', 'hist_sum_purchase_amount_0',
    ]
    cumsum_cols = []
    for e, c in enumerate(cols):
        cumsum_cols.append(c)
        pt['hist_cumusum_sum_purchase_amount' + str(e)] = pt[cumsum_cols].apply(np.sum, axis=1)
        use_cols.append('hist_cumusum_sum_purchase_amount'+str(e))

    cols = [
        'hist_count_purchase_amount_13', 'hist_count_purchase_amount_12',
        'hist_count_purchase_amount_11', 'hist_count_purchase_amount_10',
        'hist_count_purchase_amount_9', 'hist_count_purchase_amount_8',
        'hist_count_purchase_amount_7', 'hist_count_purchase_amount_6',
        'hist_count_purchase_amount_5', 'hist_count_purchase_amount_4',
        'hist_count_purchase_amount_3', 'hist_count_purchase_amount_2',
        'hist_count_purchase_amount_1', 'hist_count_purchase_amount_0'
    ]
    cumsum_cols = []
    for e, c in enumerate(cols):
        cumsum_cols.append(c)
        pt['hist_cumsum_count_purchase_amount' + str(e)] = pt[cumsum_cols].apply(np.sum, axis=1)
        use_cols.append('hist_cumsum_count_purchase_amount'+str(e))

    pt = pt[use_cols]
    pt.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

    return

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        { 
            'prefix': 'hist_',
            'index': 'card_id',
            'columns': 'month_lag',
            'values': ['purchase_amount']
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
