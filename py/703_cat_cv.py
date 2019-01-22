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

import catboost as cat

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

features =  ['f102.pkl', 'f103.pkl', 'f105.pkl']
features += ['f106_N.pkl', 'f106_Y.pkl', 'f107_N.pkl', 'f107_Y.pkl']
features += ['f202.pkl', 'f203.pkl', 'f205.pkl']
features += ['f206_N.pkl', 'f206_Y.pkl', 'f207_N.pkl', 'f207_Y.pkl']
features += ['f302.pkl', 'f303.pkl', 'f304.pkl', 'f305.pkl', 'f306.pkl']
features += ['f402.pkl', 'f403.pkl', 'f404.pkl']
features += ['f405_N.pkl', 'f405_Y.pkl', 'f406_N.pkl', 'f406_Y.pkl']
features += ['f901.pkl']

for f in features:
    print(f'Merge: {f}')
    train = pd.merge(train, pd.read_pickle(os.path.join('..', 'feature', f)), on=KEY, how='left')
    test = pd.merge(test, pd.read_pickle(os.path.join('..', 'feature', f)), on=KEY, how='left')

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
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))

gc.collect()

# ========= ====================================================================
# cv
# =============================================================================
folds = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

X = train
X_test = test

oof = np.zeros(len(X))
prediction = np.zeros(len(X_test))

scores = []

feature_importance = pd.DataFrame()

for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    dtrain = xgb.DMatrix(X.iloc[train_index], label=y.iloc[train_index])
    dvalid = xgb.DMatrix(X.iloc[valid_index], label=y.iloc[valid_index])

    model = xgb.train(
        params,
        dtrain,
        20000,          
        [dtrain, dvalid],
        early_stopping_rounds=200,
        verbose_eval=2000)
    
    y_pred_valid = model.predict(xgb.DMatrix(X.iloc[val_idx]), ntree_limit=model.best_ntree_limit+100)
    y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_ntree_limit+100)
    
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
