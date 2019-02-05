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

PREF = 'f203'

KEY = 'card_id'

stats = ['sum', 'mean', 'var']

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

# train = pd.read_csv(os.path.join(PATH, 'train.csv.gz'))[[KEY]]
# test = pd.read_csv(os.path.join(PATH, 'test.csv.gz'))[[KEY]]


PATH = os.path.join('..', 'remove_outlier_data')

new_merchant_transactions = pd.read_csv(os.path.join(PATH, 'new_merchant_transactions.csv'))
new_merchant_transactions['purchase_date'] = pd.to_datetime(new_merchant_transactions['purchase_date'])

RANGE = 7
new_merchant_transactions['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: 1 if x >= -RANGE and x <= RANGE else 0)
new_merchant_transactions['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: 1 if x >= -RANGE and x <= RANGE else 0)
new_merchant_transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: 1 if x >= -RANGE and x <= RANGE else 0)
new_merchant_transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: 1 if x >= -RANGE and x <= RANGE else 0)
new_merchant_transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: 1 if x >= -RANGE and x <= RANGE else 0)
new_merchant_transactions['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: 1 if x >= -RANGE and x <= RANGE else 0)

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
                'Christmas_Day_2017': stats,
                'Mothers_Day_2017': stats,
                'fathers_day_2017': stats,
                'Children_day_2017': stats,
                'Valentine_Day_2017': stats,
                'Black_Friday_2017': stats, 
            }
        }
    ]
    
    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
