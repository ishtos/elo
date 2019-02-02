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

PREF = 'f108'

SUMMARY = 30

KEY = 'card_id'

# stats = ['max', 'mean', 'var']
stats = ['mean']

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'))
historical_transactions['purchase_amount'] = np.log1p(historical_transactions['purchase_amount'] - historical_transactions['purchase_amount'].min())
historical_transactions = utils.reduce_mem_usage(historical_transactions)

for col in ['category_2', 'subsector_id', 'merchant_id', 'merchant_category_id']:
    historical_transactions[col + '_mean'] = historical_transactions.groupby([col])['purchase_amount'].transform('mean')
    # historical_transactions[col + '_min'] = historical_transactions.groupby([col])['purchase_amount'].transform('min')
    # historical_transactions[col + '_max'] = historical_transactions.groupby([col])['purchase_amount'].transform('max')
    # historical_transactions[col + '_sum'] = historical_transactions.groupby([col])['purchase_amount'].transform('sum')

# =============================================================================
#
# =============================================================================

def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = historical_transactions.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)
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
                'category_2_mean': stats,
                # 'category_2_min': ['min'],
                # 'category_2_max': ['max'],
                # 'category_2_sum': ['sum'],

                'category_3_mean': stats,
                # 'category_3_min': ['min'],
                # 'category_3_max': ['max'],
                # 'category_3_sum': ['sum'],

                'subsector_id_mean': stats,
                # 'subsector_id_min': stats,
                # 'subsector_id_max': stats,
                # 'subsector_id_sum': stats,

                'merchant_id_mean': stats,
                # 'merchant_id_min': stats,
                # 'merchant_id_max': stats,
                # 'merchant_id_sum': stats, 

                'merchant_category_id_mean': stats,
                # 'merchant_category_id_min': stats,
                # 'merchant_category_id_max': stats,
                # 'merchant_category_id_sum': stats,
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
