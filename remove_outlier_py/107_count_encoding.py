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

PREF = 'f107'

SUMMARY = 30

KEY = 'card_id'

stats = ['sum', 'mean', 'std']

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

categorical_columns = ['city_id', 'merchant_category_id', 'merchant_id', 'state_id', 'subsector_id']

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'), usecols=categorical_columns+['card_id'])
historical_transactions['agg_flag'] = 1

for c in categorical_columns:
    count_rank = historical_transactions.groupby(c)['agg_flag'].count().rank(ascending=False)
    historical_transactions[c + '_count'] = historical_transactions[c].map(count_rank)

historical_transactions = utils.reduce_mem_usage(historical_transactions)

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
                'city_id_count': stats,
                'merchant_category_id_count': stats,
                'merchant_id_count': stats,
                'state_id_count': stats,
                'subsector_id_count': stats,
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)