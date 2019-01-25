#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 2018

@author: toshiki.ishikawa
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import gc
import utils
import numpy as np
import pandas as pd

from tqdm import tqdm
from time import time, sleep
# from datetime import datetime
import datetime
from itertools import combinations
from multiprocessing import cpu_count, Pool


utils.start(__file__)

#==============================================================================

PATH = os.path.join('..', 'input')
os.makedirs(os.path.join('..', 'remove_outlier_data'), exist_ok=True)

train = pd.read_csv('../input/train.csv', parse_dates=['first_active_month'])
test = pd.read_csv('../input/test.csv', parse_dates=['first_active_month'])
# historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'), usecols=['card_id', 'purchase_date'])
# historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
# group = historical_transactions.groupby('card_id')['purchase_date'].max().reset_index()

# train = pd.merge(train, group, on='card_id', how='left')
# test = pd.merge(test, group, on='card_id', how='left')

#==============================================================================
# outliers
#==============================================================================
train['outliers'] = 0
train.loc[train['target'] < -30, 'outliers'] = 1

feature_cols = ['feature_1', 'feature_2', 'feature_3']
feature_label = train.groupby(feature_cols).agg({'outliers': ['mean']}).reset_index()
feature_label.columns = [f'{c[0]}_{c[1]}'.strip('_') for c in feature_label.columns]
train = pd.merge(train, feature_label, on=feature_cols, how='left')
test = pd.merge(test, feature_label, on=feature_cols, how='left')

train = train.drop('outliers', axis=1)

#==============================================================================
# train
#==============================================================================
columns = ['year', 'weekday', 'month', 'weekofyear', 'quarter']
for c in columns:
    nc = 'first_active_month' + '_' + c
    train[nc] = getattr(train['first_active_month'].dt, c).astype(int)

max_date = train['first_active_month'].dt.date.max()
train['elapsed_time'] = (max_date - train['first_active_month'].dt.date).dt.days
# train['elapsed_time'] = (train['purchase_date'].dt.date - train['first_active_month'].dt.date).dt.days

train['days_feature_1'] = train['feature_1'] * train['elapsed_time']
train['days_feature_2'] = train['feature_2'] * train['elapsed_time']
train['days_feature_3'] = train['feature_3'] * train['elapsed_time']

train['feature_1'] = train['feature_1'].astype('category')
train['feature_2'] = train['feature_2'].astype('category')
train['feature_3'] = train['feature_3'].astype('category')

# del train['purchase_date'], train['first_active_month']

train.to_csv(os.path.join('..', 'remove_outlier_data', 'train.csv'), index=False)

#==============================================================================
# test
#==============================================================================

test.loc[
    test['first_active_month'].isna(), 'first_active_month'] = test.loc[
        (test['feature_1'] == 5) & 
        (test['feature_2'] == 2) & 
        (test['feature_3'] == 1), 'first_active_month'].min()

for c in columns:
    nc = 'first_active_month' + '_' + c
    test[nc] = getattr(test['first_active_month'].dt, c).astype(int)

test['elapsed_time'] = (max_date - test['first_active_month'].dt.date).dt.days
# test['elapsed_time'] = (test['purchase_date'].dt.date - test['first_active_month'].dt.date).dt.days

test['days_feature_1'] = test['feature_1'] * test['elapsed_time']
test['days_feature_2'] = test['feature_2'] * test['elapsed_time']
test['days_feature_3'] = test['feature_3'] * test['elapsed_time']

test['feature_1'] = test['feature_1'].astype('category')
test['feature_2'] = test['feature_2'].astype('category')
test['feature_3'] = test['feature_3'].astype('category')

# del test['purchase_date'], test['first_active_month']

test.to_csv(os.path.join('..', 'remove_outlier_data', 'test.csv'), index=False)

#==============================================================================

utils.end(__file__)
