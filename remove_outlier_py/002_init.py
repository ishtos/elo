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
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from time import time, sleep
from itertools import combinations
from multiprocessing import cpu_count, Pool


utils.start(__file__)

#==============================================================================

PATH = os.path.join('..', 'input')
# os.makedirs(os.path.join('..', 'remove_outlier_data'), exist_ok=True)

train = pd.read_csv('../input/train.csv', parse_dates=['first_active_month'])
test = pd.read_csv('../input/test.csv', parse_dates=['first_active_month'])
# historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'), usecols=['card_id', 'purchase_date'])
# historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
# group = historical_transactions.groupby('card_id')['purchase_date'].max().reset_index()

# train = pd.merge(train, group, on='card_id', how='left')
# test = pd.merge(test, group, on='card_id', how='left')

#==============================================================================
# df
#==============================================================================
train['outliers'] = 0
train.loc[train['target'] < -30, 'outliers'] = 1

test['target'] = np.nan

df = pd.concat([train, test], axis=0)

del train, test
gc.collect()

df['first_active_month'] = pd.to_datetime(df['first_active_month'])

# df['quarter'] = df['first_active_month'].dt.quarter
df['elapsed_time'] = (datetime.date(2018, 5, 1) - df['first_active_month'].dt.date).dt.days

# df['days_feature1'] = df['elapsed_time'] * df['feature_1']
# df['days_feature2'] = df['elapsed_time'] * df['feature_2']
# df['days_feature3'] = df['elapsed_time'] * df['feature_3']

# df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
# df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
# df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']

# df, cols = utils.one_hot_encoder(df, nan_as_category=False)

features = ['feature_1', 'feature_2', 'feature_3']
for f in features:
    map_mean = df.groupby(f)['outliers'].mean()
    map_sum = df.groupby(f)['outliers'].sum()
    df[f + '_outliers_mean'] = df[f].map(map_mean)
    df[f + '_outliers_sum'] = df[f].map(map_sum)

# features = ['feature_1_outliers_mean', 'feature_2_outliers_mean', 'feature_3_outliers_mean']
# df['outliers_mean'] = df[features].sum(axis=1)
# features = ['feature_1_outliers_sum', 'feature_2_outliers_sum', 'feature_3_outliers_sum']
# df['outliers_sum'] = df[features].sum(axis=1)


# df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
# df['feature_mean'] = df['feature_sum'] / 3

# t = df[features]
# df['feature_max'] = t.max(axis=1)
# df['feature_min'] = t.min(axis=1)
# df['feature_var'] = t.std(axis=1)

train = df[df['target'].notnull()]
test = df[df['target'].isnull()]
del df
gc.collect()

train.to_csv(os.path.join('..', 'remove_outlier_data', 'train.csv'), index=False)
test.to_csv(os.path.join('..', 'remove_outlier_data', 'test.csv'), index=False)

#==============================================================================
# outliers
#==============================================================================
# train_df['outliers'] = 0 
# train_df.loc[train_df['target'] < -30, 'outliers'] = 1


# feature_cols = ['feature_1', 'feature_2', 'feature_3']
# feature_label = train.groupby(feature_cols).agg({'outliers': ['mean']}).reset_index()
# feature_label.columns = [f'{c[0]}_{c[1]}'.strip('_') for c in feature_label.columns]
# train = pd.merge(train, feature_label, on=feature_cols, how='left')
# test = pd.merge(test, feature_label, on=feature_cols, how='left')

# train = train.drop('outliers', axis=1)

#==============================================================================
# train
#==============================================================================
# columns = ['year', 'weekday', 'month', 'weekofyear', 'quarter']
# for c in columns:
#     nc = 'first_active_month' + '_' + c
#     train[nc] = getattr(train['first_active_month'].dt, c).astype(int)

# max_date = train['first_active_month'].dt.date.max()
# train['elapsed_time'] = (max_date - train['first_active_month'].dt.date).dt.days
# # train['elapsed_time'] = (train['purchase_date'].dt.date - train['first_active_month'].dt.date).dt.days

# train['days_feature_1'] = train['feature_1'] * train['elapsed_time']
# train['days_feature_2'] = train['feature_2'] * train['elapsed_time']
# train['days_feature_3'] = train['feature_3'] * train['elapsed_time']

# train['feature_1'] = train['feature_1'].astype('category')
# train['feature_2'] = train['feature_2'].astype('category')
# train['feature_3'] = train['feature_3'].astype('category')

# # del train['purchase_date'], train['first_active_month']

# train.to_csv(os.path.join('..', 'remove_outlier_data', 'train.csv'), index=False)

#==============================================================================
# test
#==============================================================================

# test.loc[
#     test['first_active_month'].isna(), 'first_active_month'] = test.loc[
#         (test['feature_1'] == 5) & 
#         (test['feature_2'] == 2) & 
#         (test['feature_3'] == 1), 'first_active_month'].min()

# for c in columns:
#     nc = 'first_active_month' + '_' + c
#     test[nc] = getattr(test['first_active_month'].dt, c).astype(int)

# test['elapsed_time'] = (max_date - test['first_active_month'].dt.date).dt.days
# # test['elapsed_time'] = (test['purchase_date'].dt.date - test['first_active_month'].dt.date).dt.days

# test['days_feature_1'] = test['feature_1'] * test['elapsed_time']
# test['days_feature_2'] = test['feature_2'] * test['elapsed_time']
# test['days_feature_3'] = test['feature_3'] * test['elapsed_time']

# test['feature_1'] = test['feature_1'].astype('category')
# test['feature_2'] = test['feature_2'].astype('category')
# test['feature_3'] = test['feature_3'].astype('category')

# # del test['purchase_date'], test['first_active_month']

# test.to_csv(os.path.join('..', 'remove_outlier_data', 'test.csv'), index=False)

#==============================================================================

utils.end(__file__)
