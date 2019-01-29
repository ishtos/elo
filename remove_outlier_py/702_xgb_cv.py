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
    'eta': 0.01, 
    'max_depth': 0, 
    'subsample': 0.6, 
    'colsample_bytree': 0.6, 
    'objective': 'reg:linear', 
    'eval_metric': 'rmse', 
    'silent': True
}

# =============================================================================
# features
# =============================================================================

features = []

features += [f'f10{i}.pkl' for i in (2, 4,)]
features += [f'f11{i}_{j}.pkl' for i in (1, 2) 
                               for j in ('Y', 'N')]
features += [f'f12{i}.pkl' for i in (1,)]
features += [f'f13{i}.pkl' for i in (1, 2)]

features += [f'f20{i}.pkl' for i in (2,)]
features += [f'f23{i}.pkl' for i in (1, 2)]

features += [f'f30{i}.pkl' for i in (2, 3, 4,)]

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
# change date to int
# =============================================================================
cols = train.columns.values
for f in [
    'new_purchase_date_max', 'new_purchase_date_min',
    'hist_purchase_date_max', 'hist_purchase_date_min', 
    'Y_hist_auth_purchase_date_max', 'Y_hist_auth_purchase_date_min', 
    'N_hist_auth_purchase_date_max', 'N_hist_auth_purchase_date_min',
    'Y_new_auth_purchase_date_max', 'Y_new_auth_purchase_date_min', 
    'N_new_auth_purchase_date_max', 'N_new_auth_purchase_date_min',
]:
    if f in cols:
        train[f] = train[f].astype(np.int64) * 1e-9
        test[f] = test[f].astype(np.int64) * 1e-9

# train['outlier'] = 0
# train.loc[train.target < -30, 'outlier'] = 1

# =============================================================================
# drop same values
# =============================================================================
# ffm_cols = pd.read_csv('./ffm/ffm_cols.csv')

drop_cols = [
    'hist_cumsum_count_purchase_amount13', 'Y_new_auth_purchase_date_max',
    'Y_new_auth_purchase_date_min', 'N_new_auth_purchase_date_min'
]
# drop_cols += list(ffm_cols['ffm_cols'].values)

for d in drop_cols:
    if f in cols:
        train.drop(d, axis=1, inplace=True)
        test.drop(d, axis=1, inplace=True)

# drop_cols = pd.read_csv('duplicated_columns.csv')

# for d in drop_cols['duplicated_columns'].values:
#     if d in cols:
#         train.drop(d, axis=1, inplace=True)
#         test.drop(d, axis=1, inplace=True)

# =============================================================================
# preprocess
# =============================================================================
train['nan_count'] = train.isnull().sum(axis=1)
test['nan_count'] = test.isnull().sum(axis=1)

train = train.fillna(0)
test = test.fillna(0)

y = train['target']

col_not_to_use = ['first_active_month', 'card_id', 'target']
col_to_use = [c for c in train.columns if c not in col_not_to_use]

train = train[col_to_use]
test = test[col_to_use]

train['feature_3'] = train['feature_3'].astype(int)
test['feature_3'] = test['feature_3'].astype(int)

categorical_features = ['feature_1', 'feature_2', 'feature_3']

# for col in categorical_features:
#     lbl = LabelEncoder()
#     lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
#     train[col] = lbl.transform(list(train[col].values.astype('str')))
#     test[col] = lbl.transform(list(test[col].values.astype('str')))

gc.collect()

# =============================================================================
# feature selection
# =============================================================================
# feature = pd.read_csv('IMP_csv/20190115_IMP.csv')
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
