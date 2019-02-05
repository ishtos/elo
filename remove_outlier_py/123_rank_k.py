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

PREF = 'f123'

SUMMARY = 30

KEY = 'card_id'

stats = ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr']

# =============================================================================
# def
# =============================================================================
def last_k_instalment_features(gr, periods):
    gr_ = gr.copy()
    features = {}

    for period in periods:
        gr_period = gr_[gr_['days'] <= period]

        features = utils.add_features_in_group(
            features,gr_period, 
            'installments', 
            ['max', 'mean', 'var', 'skew'],
            'hist_last_{}_'.format(period))
        
        features = utils.add_features_in_group(
            features,gr_period, 
            'purchase_amount', 
            ['sum', 'max', 'min', 'mean', 'var', 'skew'],
            'hist_last_{}_'.format(period))
    return features

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'))

historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
historical_transactions['days'] = (datetime.date(2018, 2, 28) - historical_transactions['purchase_date'].dt.date).dt.days 
historical_transactions = historical_transactions.query('0 <= installments and installments <= 12')

# =============================================================================
#
# =============================================================================

groupby = historical_transactions.groupby('card_id')

func = partial(last_k_instalment_features, periods=[30, 90, 180])

g = utils.parallel_apply(groupby, func, index_name='card_id',num_workers=4, chunk_size=10000).reset_index()
g = g.fillna(0)
g.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

#==============================================================================
utils.end(__file__)
