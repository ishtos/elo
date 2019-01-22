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
PATH = os.path.join('..', 'data')

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
# all data
# =============================================================================
train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

train['outliers'] = 0
train.loc[train['target'] < -30, 'outliers'] = 1

features = ['f103.pkl', 'f105.pkl', 'f109.pkl']
features += ['f107_N.pkl', 'f107_Y.pkl', 'f108_N.pkl', 'f108_Y.pkl']
features += ['f203.pkl', 'f205.pkl', 'f209.pkl']
features += ['f207_N.pkl', 'f207_Y.pkl', 'f208_N.pkl', 'f208_Y.pkl']
features += ['f302.pkl', 'f303.pkl', 'f304.pkl', 'f305.pkl', 'f306.pkl']
features += ['f403.pkl', 'f404.pkl', 'f409.pkl']
features += ['f406_N.pkl', 'f406_Y.pkl', 'f408_N.pkl', 'f408_Y.pkl']

for f in features:
    print(f)
    train = pd.merge(train, pd.read_pickle(
        os.path.join('..', 'feature', f)), on=KEY, how='left')
    test = pd.merge(test, pd.read_pickle(
        os.path.join('..', 'feature', f)), on=KEY, how='left')

for f in ['hist_purchase_date_max', 'hist_purchase_date_min',
          'N_hist_auth_purchase_date_max', 'N_hist_auth_purchase_date_min',
          'Y_hist_auth_purchase_date_max', 'Y_hist_auth_purchase_date_min',
          'new_purchase_date_max', 'new_purchase_date_min',
          'N_new_auth_purchase_date_max', 'N_new_auth_purchase_date_min',
          'Y_new_auth_purchase_date_max', 'Y_new_auth_purchase_date_min',
          'union_purchase_date_max', 'union_purchase_date_min',
          'N_union_auth_purchase_date_max', 'N_union_auth_purchase_date_min',
          'Y_union_auth_purchase_date_max', 'Y_union_auth_purchase_date_min']:
    train[f] = train[f].astype(np.int64) * 1e-9
    test[f] = test[f].astype(np.int64) * 1e-9

train = train.drop(['N_authorized_flag_x', 'Y_authorized_flag_x',
                    'N_authorized_flag_y', 'Y_authorized_flag_y',
                    'union_transactions_count_x', 'union_transactions_count_y'], axis=1)
test = test.drop(['N_authorized_flag_x', 'Y_authorized_flag_x',
                  'N_authorized_flag_y', 'Y_authorized_flag_y',
                  'union_transactions_count_x', 'union_transactions_count_y'], axis=1)

y = train['target']

train['feature_3'] = train['feature_3'].astype(int)
test['feature_3'] = test['feature_3'].astype(int)

categorical_features = ['feature_1', 'feature_2', 'feature_3']

for col in categorical_features:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) +
            list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))

col_not_to_use = ['first_active_month', 'card_id', 'target', 'outliers']
col_to_use = [c for c in train.columns if c not in col_not_to_use]

gc.collect()

# =============================================================================
# cv
# =============================================================================
folds = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

X = train[train['outliers'] == 0]
X = X[col_to_use]
X_test = test[col_to_use]

oof = np.zeros(len(X))
prediction = np.zeros(len(X_test))

scores = []

feature_importance = pd.DataFrame()

for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    # dtrain = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index], categorical_feature=categorical_features)
    # dvalid = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index], categorical_feature=categorical_features)

    dtrain = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index])
    dvalid = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index])

    model = lgb.train(
        params1,
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

oof = np.zeros(len(X))
prediction = np.zeros(len(X_test))

scores = []

feature_importance = pd.DataFrame()

for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    # dtrain = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index],
    #                      categorical_feature=categorical_features)
    # dvalid = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index],
    #                      categorical_feature=categorical_features)

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
outlier_id = pd.DataFrame(outlier_prob.sort_values(by='target', ascending=False).head(25000)['card_id'])
best_submission = pd.read_csv('../submission/3692.csv')
most_likely_liers = pd.merge(best_submission, outlier_id, on='card_id', how='right')

ix1 = model_without_outliers['card_id'].isin(outlier_id['card_id'].values)
ix2 = best_submission['card_id'].isin(outlier_id['card_id'].values)
model_without_outliers.loc[ix2, 'target'] = best_submission[ix1]['target']
model_without_outliers.to_csv('../submission/combining_submission.csv', index=False)

#==============================================================================
utils.end(__file__)
