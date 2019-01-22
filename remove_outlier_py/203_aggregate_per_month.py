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

PREF = 'f203'

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
    prefix, key, num_aggregations = args['prefix'] ,args['key'], args['num_aggregations']
    
    grouped = new_merchant_transactions.groupby(key)
    agg = grouped.agg(num_aggregations)
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg.reset_index(inplace=True)

    agg = agg.groupby('card_id').agg(['mean', 'std'])
    agg.columns = [prefix+'_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()
    agg = agg.rename(columns={prefix+KEY: KEY})
    
    agg.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

    return 

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {   
            'prefix': 'new_', 
            'key': ['card_id', 'month_lag'],
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
