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

PREF = 'f105'

SUMMARY = 30

KEY = 'card_id'

stats = ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr']

# =============================================================================
# def
# =============================================================================
def last_k_instalment_features_with_fractions(gr, periods, fraction_periods):
    gr_ = gr.copy()
    
    features = {}
    features_temp = {}

    for period in periods:
        gr_period = gr_[gr_['days'] <= period]

        features_temp = utils.add_features_in_group(
            features_temp,gr_period, 
            'installments', 
            ['mean', 'var', 'skew', 'kurt','iqr'], 
            'last_{}_'.format(period))
        
        features_temp = utils.add_features_in_group(
            features_temp,gr_period, 
            'purchase_amount', 
            ['sum', 'max', 'mean', 'var', 'skew', 'kurt','iqr'],
            'last_{}_'.format(period))
    
    for short_period, long_period in fraction_periods:
        short_feature_names = utils._get_feature_names(features_temp, short_period)
        long_feature_names = utils._get_feature_names(features_temp, long_period)
        
        for short_feature, long_feature in zip(short_feature_names, long_feature_names):
            old_name_chunk = '_{}_'.format(short_period)
            new_name_chunk ='_{}by{}_fraction_'.format(short_period, long_period)
            fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
            features[fraction_feature_name] = utils.safe_div(features_temp[short_feature], features_temp[long_feature])
    return pd.Series(features)


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

func = partial(
    last_k_instalment_features_with_fractions, 
    periods=[60, 180, 360, 540], 
    fraction_periods=[(60, 180),(60, 360),(180, 540),(360, 540)])

g = utils.parallel_apply(groupby, func, index_name='card_id',num_workers=4, chunk_size=10000).reset_index()
g.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

#==============================================================================
utils.end(__file__)
