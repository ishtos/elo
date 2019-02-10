#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 2018

@author: toshiki.ishikawa
"""

import os
import gc
import sys
import utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from datetime import datetime, date
from collections import defaultdict
from multiprocessing import cpu_count, Pool

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

import warnings
warnings.simplefilter('ignore')

utils.start(__file__)
#==============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

KEY = 'card_id'

SEED = 18
# SEED = np.random.randint(9999)

NTHREAD = cpu_count()

NFOLD = 4

# params1 = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting': 'gbdt',
#     'learning_rate': 0.018545526395058548,
#     'max_depth': 15,
#     'num_leaves': 54,

#     'min_child_weight': 5.343384366323818,
#     'min_data_in_leaf': 79,
#     'reg_alpha': 1.1302650970728192,
#     'reg_lambda': 0.3603427518866501,

#     'feature_fraction': 0.8354507676881442,
#     'nthread': NTHREAD,
#     'bagging_freq': 3,
#     'bagging_fraction': 0.8126672064208567,
#     'verbose': -1,
#     'seed': SEED
# }

# params2 = {
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'boosting': 'rf',
#     'learning_rate': 0.018545526395058548,
#     'max_depth': 15,
#     'num_leaves': 54,

#     'min_child_weight': 5.343384366323818,
#     'min_data_in_leaf': 79,
#     'reg_alpha': 1.1302650970728192,
#     'reg_lambda': 0.3603427518866501,

#     'feature_fraction': 0.8354507676881442,
#     'nthread': NTHREAD,
#     'bagging_freq': 3,
#     'bagging_fraction': 0.8126672064208567,
#     'verbose': -1,
#     'seed': SEED
# }

params1 = {
    'num_leaves': 31,
    'min_data_in_leaf': 30,
    'objective': 'regression',
    'max_depth': -1,
    'learning_rate': 0.01,
    'boosting': 'gbdt',
    'feature_fraction': 0.9,
    'bagging_freq': 1,
    'bagging_fraction': 0.9,
    'bagging_seed': 11,
    'metric': 'rmse',
    'lambda_l1': 0.1,
    'verbosity': -1,
    'nthread': NTHREAD,
    'random_state': SEED
}

params2 = {
    'num_leaves': 31,
    'min_data_in_leaf': 30,
    'objective': 'binary',
    'max_depth': 6,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_freq': 1,
    'bagging_fraction': 0.9,
    'bagging_seed': 11,
    'metric': 'binary_logloss',
    'lambda_l1': 0.1,
    'verbosity': -1,
    'random_state': SEED
}

# =============================================================================
# features
# =============================================================================

features = []

features += [f'f10{i}.pkl' for i in (2, 4, 6, 7)]
# features += [f'f11{i}_{j}.pkl' for i in (1,) 
#                                for j in ('Y', 'N')]
# features += [f'f12{i}.pkl' for i in (1,)]
# features += [f'f13{i}.pkl' for i in (1,)]
features += [f'f14{i}.pkl' for i in (1,)]

features += [f'f20{i}.pkl' for i in (2, 4)]
# features += [f'f23{i}.pkl' for i in (1, 2)]

features += [f'f30{i}.pkl' for i in (2,)]

# =============================================================================
# read data and features
# =============================================================================
train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

for f in tqdm(features):
    t = pd.read_pickle(os.path.join('..', 'remove_outlier_feature', f))
    train = pd.merge(train, t, on=KEY, how='left')
    test = pd.merge(test, t, on=KEY, how='left')

# =============================================================================
# add handcrafted features
# =============================================================================
df = pd.concat([train, test], axis=0)
df['first_active_month'] = pd.to_datetime(df['first_active_month'])

df['hist_first_buy'] = (df['hist_purchase_date_min'].dt.date - df['first_active_month'].dt.date).dt.days
df['hist_last_buy'] = (df['hist_purchase_date_max'].dt.date - df['first_active_month'].dt.date).dt.days
df['new_first_buy'] = (df['new_purchase_date_min'].dt.date - df['first_active_month'].dt.date).dt.days
df['new_last_buy'] = (df['new_purchase_date_max'].dt.date - df['first_active_month'].dt.date).dt.days

# df['hist_first_buy'] = (datetime.date(2018, 4, 30) - df['hist_purchase_date_min'].dt.date).dt.days
# df['hist_last_buy'] = (datetime.date(2018, 4, 30) - df['hist_purchase_date_max'].dt.date).dt.days
# df['new_first_buy'] = (datetime.date(2018, 4, 30) - df['new_purchase_date_min'].dt.date).dt.days
# df['new_last_buy'] = (datetime.date(2018, 4, 30) - df['new_purchase_date_max'].dt.date).dt.days

date_features = [
    'hist_purchase_date_max','hist_purchase_date_min',
    'new_purchase_date_max', 'new_purchase_date_min',
    # 'Y_hist_auth_purchase_date_max', 'Y_hist_auth_purchase_date_min', 
    # 'N_hist_auth_purchase_date_max', 'N_hist_auth_purchase_date_min'
]

for f in date_features:
    df[f] = df[f].astype(np.int64) * 1e-9

df['card_id_total'] = df['new_card_id_size'] + df['hist_card_id_size']
df['card_id_cnt_total'] = df['new_card_id_count'] + df['hist_card_id_count']
df['card_id_cnt_ratio'] = df['new_card_id_count'] / df['hist_card_id_count']
df['purchase_amount_total'] = df['new_purchase_amount_sum'] + df['hist_purchase_amount_sum']
df['purchase_amount_mean'] = df['new_purchase_amount_mean'] + df['hist_purchase_amount_mean']
df['purchase_amount_max'] = df['new_purchase_amount_max'] + df['hist_purchase_amount_max']
df['purchase_amount_min'] = df['new_purchase_amount_min'] + df['hist_purchase_amount_min']
df['purchase_amount_ratio'] = df['new_purchase_amount_sum'] / df['hist_purchase_amount_sum']
# df['month_diff_mean'] = df['new_month_diff_mean'] + df['hist_month_diff_mean']
# df['month_diff_ratio'] = df['new_month_diff_mean'] / df['hist_month_diff_mean']
# df['month_lag_mean'] = df['new_month_lag_mean'] + df['hist_month_lag_mean']
# df['month_lag_max'] = df['new_month_lag_max'] + df['hist_month_lag_max']
# df['month_lag_min'] = df['new_month_lag_min'] + df['hist_month_lag_min']
# df['category_1_mean'] = df['new_category_1_mean'] + df['hist_category_1_mean']
# # df['installments_total'] = df['new_installments_sum'] + df['hist_installments_sum']
# df['installments_mean'] = df['new_installments_mean'] + df['hist_installments_mean']
# df['installments_max'] = df['new_installments_max'] + df['hist_installments_max']
# df['installments_ratio'] = df['new_installments_sum'] / df['hist_installments_sum']
# df['price_total'] = df['purchase_amount_total'] / df['installments_total']
# df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
# df['price_max'] = df['purchase_amount_max'] / df['installments_max']
df['duration_mean'] = df['new_duration_mean'] + df['hist_duration_mean']
df['duration_min'] = df['new_duration_min'] + df['hist_duration_min']
df['duration_max'] = df['new_duration_max'] + df['hist_duration_max']
df['amount_month_ratio_mean'] = df['new_amount_month_ratio_mean'] + df['hist_amount_month_ratio_mean']
df['amount_month_ratio_min'] = df['new_amount_month_ratio_min'] + df['hist_amount_month_ratio_min']
df['amount_month_ratio_max'] = df['new_amount_month_ratio_max'] + df['hist_amount_month_ratio_max']
df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
# df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']

train = df[df['target'].notnull()]
test = df[df['target'].isnull()]

del df
gc.collect()

# =============================================================================
# target encoding
# =============================================================================

# =============================================================================
# drop same values
# =============================================================================

# =============================================================================
# preprocess
# =============================================================================
train['nan_count'] = train.isnull().sum(axis=1)
test['nan_count'] = test.isnull().sum(axis=1)

y = train['target']

col_not_to_use = [
    'first_active_month', 'card_id', 'target', 'outliers',
    'hist_card_id_size', 'new_card_id_size'
    # 'hist_purchase_date_max', 'new_purchase_date_max',
    # 'hist_purchase_date_min', 'new_purchase_date_min', 
    # 'hist_month_lag_max', 'hist_month_lag_min', 
    # 'hist_month_lag_mean', 'hist_month_lag_var' ,'hist_month_lag_skew'
]

col_not_to_use += [c for c in train.columns if ('duration' in c) or ('amount_month_ratio' in c) or ('skew' in c)]
col_to_use = [c for c in train.columns if c not in col_not_to_use]


gc.collect()

# X = pd.get_dummies(X, columns=categorical_features, drop_first=True, dummy_na=True)
# X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True, dummy_na=True)

# =============================================================================
# cv
# =============================================================================
folds = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

X = train[train['outliers'] == 0]
X = X[col_to_use]
X_test = test[col_to_use]

categorical_features = [
    'feature_1', 'feature_2', 'feature_3',
]

for c in categorical_features:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')

oof = np.zeros(len(X))
prediction = np.zeros(len(X_test))

scores = []

feature_importance = pd.DataFrame()

for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    # dtrain = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index], categorical_feature=categorical_features)
    # dvalid = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index], categorical_feature=categorical_features)

    dtrain = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index])
    dvalid = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index])

    params = {
        'boosting': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'subsample': 0.9855232997390695,
        'max_depth': 7,
        'top_rate': 0.9064148448434349,
        'num_leaves': 63,
        'min_child_weight': 41.9612869171337,
        'other_rate': 0.0721768246018207,
        'reg_alpha': 9.677537745007898,
        'colsample_bytree': 0.5665320670155495,
        'min_split_gain': 9.820197773625843,
        'reg_lambda': 8.2532317400459,
        'min_data_in_leaf': 21,
        'verbose': -1,
        'seed':int(2**fold_n),
        'bagging_seed':int(2**fold_n),
        'drop_seed':int(2**fold_n)
    }

    model = lgb.train(
        params,
        dtrain,
        20000,
        valid_sets=[dtrain, dvalid],
        verbose_eval=200,
        early_stopping_rounds=200)

    y_pred_valid = model.predict(X.iloc[valid_index], num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    oof[valid_index] = y_pred_valid.reshape(-1,)
    scores.append(np.sqrt(mean_squared_error(y.iloc[valid_index], y_pred_valid)))

    prediction += y_pred / folds.n_splits

print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

model_without_outliers = pd.DataFrame({'card_id': test['card_id'].values})
model_without_outliers['target'] = prediction

# =============================================================================
# classifier
# =============================================================================
y = train['outliers']

col_not_to_use = ['first_active_month', 'card_id', 'target', 'outliers']
col_to_use = [c for c in train.columns if c not in col_not_to_use]

# =============================================================================
# cv
# =============================================================================
folds = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

X = train[col_to_use]
X_test = test[col_to_use]

categorical_features = [
    'feature_1', 'feature_2', 'feature_3',
]

for c in categorical_features:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')

oof = np.zeros(len(X))
prediction = np.zeros(len(X_test))

scores = []

feature_importance = pd.DataFrame()

for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    dtrain = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index])
    dvalid = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index])

    model = lgb.train(
        params2,
        dtrain,
        20000,
        valid_sets=[dtrain, dvalid],
        verbose_eval=2000,
        early_stopping_rounds=200)

    y_pred_valid = model.predict(X.iloc[valid_index], num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    oof[valid_index] = y_pred_valid.reshape(-1,)
    scores.append(np.sqrt(mean_squared_error(y.iloc[valid_index], y_pred_valid)))

    prediction += y_pred / folds.n_splits

print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

outlier_prob = pd.DataFrame({'card_id': test['card_id'].values})
outlier_prob['target'] = prediction

# =============================================================================
# Combining
# =============================================================================
outlier_id = pd.DataFrame(outlier_prob.sort_values(by='target', ascending=False).head(10000)['card_id'])
best_submission = pd.read_csv('../submission/lightgbm_outlier_20190202_temp.csv')
most_likely_liers = pd.merge(best_submission, outlier_id, on='card_id', how='right')

ix1 = model_without_outliers['card_id'].isin(outlier_id['card_id'].values)
ix2 = best_submission['card_id'].isin(outlier_id['card_id'].values)
model_without_outliers.loc[ix2, 'target'] = best_submission.loc[ix1. 'target']
model_without_outliers.to_csv('../submission/combining_submission.csv', index=False)

#==============================================================================
utils.end(__file__)
