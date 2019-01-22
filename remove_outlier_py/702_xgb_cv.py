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

from glob import glob
from tqdm import tqdm
from datetime import datetime, date
from collections import defaultdict
from multiprocessing import cpu_count, Pool

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error

import xgboost as xgb

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

params = {
    'eta': 0.005, 
    'max_depth': 10, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    'objective': 'reg:linear', 
    'eval_metric': 'rmse', 
    'silent': True
}

# =============================================================================
# all data
# =============================================================================
train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

# features =  ['f102.pkl', 'f103.pkl', 'f105.pkl']
# features += ['f106_N.pkl', 'f106_Y.pkl', 'f107_N.pkl', 'f107_Y.pkl']
# features += ['f202.pkl', 'f203.pkl', 'f205.pkl']
# features += ['f206_N.pkl', 'f206_Y.pkl', 'f207_N.pkl', 'f207_Y.pkl']
# features += ['f302.pkl', 'f303.pkl', 'f304.pkl', 'f305.pkl', 'f306.pkl']
# features += ['f402.pkl', 'f403.pkl', 'f404.pkl']
# features += ['f405_N.pkl', 'f405_Y.pkl', 'f406_N.pkl', 'f406_Y.pkl']
# features += ['f901.pkl']

features = ['f103.pkl', 'f105.pkl', 'f109.pkl']
features += ['f107_N.pkl', 'f107_Y.pkl', 'f108_N.pkl', 'f108_Y.pkl']
features += ['f203.pkl', 'f205.pkl', 'f209.pkl']
features += ['f207_N.pkl', 'f207_Y.pkl', 'f208_N.pkl', 'f208_Y.pkl']
features += ['f302.pkl', 'f303.pkl', 'f304.pkl', 'f305.pkl', 'f306.pkl']
features += ['f403.pkl', 'f404.pkl', 'f409.pkl', 'f411.pkl']
features += ['f406_N.pkl', 'f406_Y.pkl', 'f408_N.pkl', 'f408_Y.pkl']

for f in features:
    print(f'Merge: {f}')
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

# for col in train.columns:
#     if train[col].isna().any():
#         train[col] = train[col].fillna(0)

# for col in test.columns:
#     if test[col].isna().any():
#         test[col] = test[col].fillna(0)

train['nan_count'] = train.isnull().sum(axis=1)
test['nan_count'] = test.isnull().sum(axis=1)

y = train['target']

col_not_to_use = ['first_active_month', 'card_id', 'target']
col_to_use = [c for c in train.columns if c not in col_not_to_use]

train = train[col_to_use]
test = test[col_to_use]

train['feature_3'] = train['feature_3'].astype(int)
test['feature_3'] = test['feature_3'].astype(int)

categorical_features = ['feature_1', 'feature_2', 'feature_3']

for col in categorical_features:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) +
            list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))

# for col in ['hist_purchase_amount_max', 'hist_purchase_date_max', 'new_purchase_amount_max', 'new_purchase_date_max']:
#     train[col + '_to_mean'] = train[col] / train[col].mean()
#     test[col + '_to_mean'] = test[col] / test[col].mean()

gc.collect()

# =============================================================================
# feature selection
# =============================================================================
# feature = pd.read_csv('../py/20190109_1_IMP.csv')
# g = feature.groupby(['feature'])['importance'].mean().reset_index()
# g = g.sort_values('importance', ascending=False).reset_index(drop=True)
# g = g[g.importance > 0]
# g = g.feature.values

# X = train[g]
# X_test = test[g]

X = train
X_test = test

# ========= ====================================================================
# cv
# =============================================================================
folds = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

oof = np.zeros(len(X))
prediction = np.zeros(len(X_test))

scores = []

feature_importance = pd.DataFrame()

for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    dtrain = xgb.DMatrix(data=X.iloc[train_index], label=y.iloc[train_index])
    dvalid = xgb.DMatrix(data=X.iloc[valid_index], label=y.iloc[valid_index])

    model = xgb.train(
        params,
        dtrain,
        20000,          
        [(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=200,
        verbose_eval=2000)
    
    y_pred_valid = model.predict(xgb.DMatrix(data=X.iloc[valid_index]), ntree_limit=model.best_ntree_limit+100)
    y_pred = model.predict(xgb.DMatrix(data=X_test), ntree_limit=model.best_ntree_limit+100)
    
    oof[valid_index] = y_pred_valid.reshape(-1,)
    scores.append(np.sqrt(mean_squared_error(y.iloc[valid_index], y_pred_valid)))
    
    prediction += y_pred / folds.n_splits   

np.save(os.path.join('stacking', 'oof_xgb'), oof)
np.save(os.path.join('stacking', 'prediction_xgb'), prediction)

print('shape:', X.shape)
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
print(features)

submission = pd.read_csv(os.path.join('..', 'input', 'sample_submission.csv'))
submission['target'] = prediction
submission.to_csv(os.path.join('..', 'submission', 'xgboost_{}.csv'.format(str(date.today()).replace('-', ''))), index=False)

#==============================================================================
utils.end(__file__)
