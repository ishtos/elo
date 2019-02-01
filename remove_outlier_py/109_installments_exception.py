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

PREF = 'f109'

SUMMARY = 30

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew']

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'), usecols=['card_id', 'installments'])
historical_transactions = utils.reduce_mem_usage(historical_transactions)

historical_transactions['-1_installments'] = historical_transactions['installments'].apply(lambda x: np.where(x == -1, 1, 0))
historical_transactions['999_installments'] = historical_transactions['installments'].apply(lambda x: np.where(x == 999, 1, 0))
historical_transactions['exception_installments'] = historical_transactions['installments'].apply(lambda x: np.where(x == 999 or x == -1, 1, 0))

# =============================================================================
#
# =============================================================================

def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = historical_transactions.groupby(key).agg(num_aggregations)
    agg.columns = [prefix + '_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace = True)
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
                # '-1_installments': ['sum'],
                # '999_installments': ['sum'], 
                'exception_installments': ['sum'],
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
