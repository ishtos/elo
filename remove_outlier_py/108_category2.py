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

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew']

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'))
historical_transactions['purchase_amount'] = np.log1p(historical_transactions['purchase_amount'] - historical_transactions['purchase_amount'].min())
historical_transactions = utils.reduce_mem_usage(historical_transactions)

for col in ['category_2','category_3']:
    historical_transactions[col + '_mean'] = historical_transactions.groupby([col])['purchase_amount'].transform('mean')
    historical_transactions[col + '_min'] = historical_transactions.groupby([col])['purchase_amount'].transform('min')
    historical_transactions[col + '_max'] = historical_transactions.groupby([col])['purchase_amount'].transform('max')
    historical_transactions[col + '_sum'] = historical_transactions.groupby([col])['purchase_amount'].transform('sum')

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
                'category_2_mean': ['mean'],
                # 'category_2_min': ['min'],
                # 'category_2_max': ['max'],
                # 'category_2_sum': ['sum'],

                'category_3_mean': ['mean'],
                # 'category_3_min': ['min'],
                # 'category_3_max': ['max'],
                # 'category_3_sum': ['sum'],
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
