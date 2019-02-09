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

PREF = 'f206'

SUMMARY = 30

KEY = 'card_id'

stats = ['mean', 'sum']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

new_merchant_transactions = pd.read_csv(os.path.join(PATH, 'new_merchant_transactions.csv'), usecols=['card_id', 'installments'])
new_merchant_transactions = utils.reduce_mem_usage(new_merchant_transactions)


new_merchant_transactions['-1_installments'] = new_merchant_transactions['installments'].apply(lambda x: np.where(x == -1, 1, 0))
new_merchant_transactions['999_installments'] = new_merchant_transactions['installments'].apply(lambda x: np.where(x == 999, 1, 0))
new_merchant_transactions['exception_installments'] = new_merchant_transactions['installments'].apply(lambda x: np.where(x == 999 or x == -1, 1, 0))


# =============================================================================
#
# =============================================================================

def aggregate(args):
    prefix, key, num_aggregations = args['prefix'], args['key'], args['num_aggregations']

    agg = new_merchant_transactions.groupby(key).agg(num_aggregations)
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
            'prefix': 'new_',
            'key': 'card_id',
            'num_aggregations': {
                # '-1_installments': stats,
                '999_installments': stats, 
                # 'exception_installments': stats,
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
