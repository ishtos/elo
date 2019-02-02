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
import warnings
import random
import glob
import datetime

import numpy as np
import pandas as pd

from tqdm import tqdm
from attrdict import AttrDict
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from multiprocessing import cpu_count, Pool
from functools import reduce, partial
from scipy.stats import skew, kurtosis, iqr

utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f205'

SUMMARY = 30

KEY = 'card_id'

stats = ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr']

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

new_merchant_transactions = pd.read_csv(os.path.join(PATH, 'new_merchant_transactions.csv'), usecols=['card_id', 'category_2', 'category_3'])
new_merchant_transactions = pd.get_dummies(new_merchant_transactions, columns=['category_2', 'category_3'], dummy_na=True)

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
                'category_2_1.0': ['sum'], # ['sum', 'mean'], 
                'category_2_2.0': ['sum'], # ['sum', 'mean'],  
                'category_2_3.0': ['sum'], # ['sum', 'mean'], 
                'category_2_4.0': ['sum'], # ['sum', 'mean'],
                'category_2_5.0': ['sum'], # ['sum', 'mean'],
                'category_2_nan': ['sum'], # ['sum', 'mean'], 
               
                'category_3_0.0': ['sum'], # ['sum', 'mean'],
                'category_3_1.0': ['sum'], # ['sum', 'mean'], 
                'category_3_2.0': ['sum'], # ['sum', 'mean']
                'category_3_nan': ['sum'], # ['sum', 'mean']
            }
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)






