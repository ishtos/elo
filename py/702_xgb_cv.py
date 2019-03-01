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
import datetime
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import cpu_count, Pool

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

import xgboost as xgb

import warnings
warnings.simplefilter('ignore')

utils.start(__file__)
#==============================================================================
# Logger
#==============================================================================
from logging import getLogger, FileHandler, Formatter, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)

file_handler = FileHandler(os.path.join('logs', 'log_{}'.format(str(datetime.datetime.today().date()).replace('-', ''))))
formatter = Formatter('%(message)s')
file_handler.setFormatter(formatter)
file_handler.setLevel(DEBUG)

logger.addHandler(file_handler)
logger.propagate = False

#==============================================================================
PATH = os.path.join('..', 'data')

KEY = 'card_id'

SEED = 18
# SEED = np.random.randint(9999)

NTHREAD = cpu_count()

NFOLD = 11

# =============================================================================
# features
# =============================================================================
features = []

features += [f'f10{i}.pkl' for i in (2, 3, 4, 5, 6, 7, 8)]
features += [f'f11{i}_{j}.pkl' for i in (1,) 
                               for j in ('Y', 'N')]
features += [f'f13{i}.pkl' for i in (1, 3, 4)]

features += [f'f20{i}.pkl' for i in (2, 3, 4, 5, 6, 7, 8)]
# features += [f'f23{i}.pkl' for i in (1, 3)]

features += [f'f30{i}.pkl' for i in (2, )]

# =============================================================================
# all data
# =============================================================================
train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

for f in tqdm(features):
    t = pd.read_pickle(os.path.join('..', 'feature', f))
    train = pd.merge(train, t, on=KEY, how='left')
    test = pd.merge(test, t, on=KEY, how='left')

df = pd.concat([train, test], axis=0)
df['first_active_month'] = pd.to_datetime(df['first_active_month'])

date_features = [
    'hist_purchase_date_max','hist_purchase_date_min',
    'new_purchase_date_max', 'new_purchase_date_min',
    'Y_hist_auth_purchase_date_max', 'N_hist_auth_purchase_date_max',
    'Y_hist_auth_purchase_date_min', 'N_hist_auth_purchase_date_min'
]

for f in date_features:
    df[f] = pd.to_datetime(df[f])

df['hist_first_buy'] = (df['hist_purchase_date_min'].dt.date - df['first_active_month'].dt.date).dt.days
df['hist_last_buy'] = (df['hist_purchase_date_max'].dt.date - df['first_active_month'].dt.date).dt.days
df['new_first_buy'] = (df['new_purchase_date_min'].dt.date - df['first_active_month'].dt.date).dt.days
df['new_last_buy'] = (df['new_purchase_date_max'].dt.date - df['first_active_month'].dt.date).dt.days

for f in date_features:
    df[f] = df[f].astype(np.int64) * 1e-9

df['card_id_cnt_total'] = df['new_card_id_count'] + df['hist_card_id_count']
df['purchase_amount_total'] = df['new_purchase_amount_sum'] + df['hist_purchase_amount_sum']
df['purchase_amount_mean'] = df['new_purchase_amount_mean'] + df['hist_purchase_amount_mean']
df['purchase_amount_max'] = df['new_purchase_amount_max'] + df['hist_purchase_amount_max']
df['purchase_amount_min'] = df['new_purchase_amount_min'] + df['hist_purchase_amount_min']
df['sum_new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
df['sum_hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
df['sum_CLV_ratio'] = df['sum_new_CLV'] / df['sum_hist_CLV']

df['nans'] = df.isnull().sum(axis=1)

train = df[df['target'].notnull()]
test = df[df['target'].isnull()]
y = train['target']

categorical_features = ['feature_1', 'feature_2', 'feature_3']
pca = PCA(n_components=1)
pca.fit(train[categorical_features])
pca_train_values = pca.transform(train[categorical_features])
pca_test_values = pca.transform(test[categorical_features])

pca_train_values = np.transpose(pca_train_values, (1, 0))
pca_test_values = np.transpose(pca_test_values, (1, 0))

for e, (pca_train, pca_test) in enumerate(zip(pca_train_values, pca_test_values)):
    train[f'pca_feature_{e}'] = pca_train
    test[f'pca_feature_{e}'] = pca_test

del df
gc.collect()

col_to_use = [
    'feature_1', 
    'feature_2', 
    'elapsed_time', 
    'feature_1_outliers_mean', 
    'feature_1_outliers_sum', 
    'feature_2_outliers_mean',
    'feature_2_outliers_sum',
    'hist_subsector_id_nunique',
    'hist_year_nunique',
    'hist_month_nunique',
    'hist_month_mean',
    'hist_month_min',
    'hist_month_max',
    'hist_hour_nunique',
    'hist_hour_min',
    'hist_hour_max',
    'hist_weekofyear_nunique',
    'hist_weekofyear_min',
    'hist_day_nunique',
    'hist_weekday_mean',
    'hist_weekday_min',
    'hist_weekday_max',
    'hist_purchase_amount_sum',
    'hist_purchase_amount_min',
    'hist_purchase_amount_mean',
    'hist_installments_sum',
    'hist_installments_max',
    'feature_3',
    'hist_merchant_id_nunique',
    'hist_purchase_amount_skew',
    'hist_purchase_date_max',
    'hist_month_lag_max',
    'hist_month_lag_min',
    'hist_month_lag_mean',
    'hist_month_diff_mean',
    'hist_month_diff_std',
    'hist_authorized_flag_sum',
    'hist_authorized_flag_mean',
    'hist_authorized_flag_std',
    'hist_authorized_flag_skew',
    'hist_category_1_mean',
    'hist_category_2_nunique',
    'hist_category_2_4_mean',
    'hist_category_3_0_mean',
    'hist_category_3_1_mean',
    'hist_merchant_category_id_count_sum',
    'hist_merchant_id_count_sum',
    'hist_merchant_id_count_mean',
    'hist_merchant_id_count_std',
    'new_city_minus_one_sum',
    'new_city_minus_one_mean',
    'new_city_minus_one_std',
    'Y_hist_auth_subsector_id_nunique',
    'Y_hist_auth_merchant_id_nunique',
    'Y_hist_auth_merchant_category_id_nunique',
    'Y_hist_auth_month_nunique',
    'Y_hist_auth_day_nunique',
    'Y_hist_auth_weekday_max',
    'Y_hist_auth_purchase_amount_sum',
    'Y_hist_auth_purchase_date_min',
    'Y_hist_auth_month_lag_min',
    'Y_hist_auth_month_lag_std',
    'Y_hist_auth_month_diff_mean',
    'Y_hist_auth_category_1_mean',
    'Y_hist_auth_category_2_mean',
    'Y_hist_auth_price_mean',
    'Y_hist_auth_duration_mean',
    'N_hist_auth_subsector_id_nunique',
    'N_hist_auth_merchant_id_nunique',
    'N_hist_auth_merchant_category_id_nunique',
    'new_subsector_id_nunique',
    'new_merchant_id_nunique',
    'new_merchant_category_id_nunique',
    'new_year_nunique',
    'new_month_nunique',
    'new_month_mean',
    'new_month_min',
    'new_hour_mean',
    'new_weekofyear_mean',
    'new_weekofyear_min',
    'new_weekofyear_max',
    'new_day_mean',
    'new_day_max',
    'new_purchase_amount_sum',
    'new_purchase_amount_max',
    'new_purchase_amount_min',
    'new_installments_skew',
    'new_purchase_date_max',
    'new_purchase_date_min',
    'new_month_lag_max',
    'new_month_lag_mean',
    'new_month_lag_std',
    'new_month_diff_mean',
    'new_category_1_mean',
    'new_price_min',
    'new_duration_skew',
    'new_purchase_date_diff',
    'new_purchase_date_average',
    'new_purchase_date_uptonow',
    'new_Black_Friday_2017_mean',
    'new_category_2_1_mean',
    'new_category_2_2_sum',
    'new_category_2_2_mean',
]

X = train[col_to_use]
X_test = test[col_to_use]

for i, j in zip(X.columns, X.dtypes):
    print(i, j)

# ========= ===================================================================
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

    params = {
        'gpu_id': 0, 
        'objective': 'reg:linear', 
        'eval_metric': 'rmse', 
        'silent': True, 
        'booster': 'gbtree', 
        'n_jobs': 4, 
        'n_estimators': 2500, 
        'tree_method': 'hist', 
        'grow_policy': 'lossguide', 
        'max_depth': 12, 
        'seed': int(2**fold_n), 
        'colsample_bylevel': 0.9, 
        'colsample_bytree': 0.8, 
        'gamma': 0.0001, 
        'learning_rate': 0.006150886706231842, 
        'max_bin': 128, 
        'max_leaves': 47, 
        'min_child_weight': 40, 
        'reg_alpha': 10.0, 
        'reg_lambda': 10.0, 
        'subsample': 0.9
    }

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


np.save(os.path.join('stacking', '{}_oof_xgb'.format(str(datetime.datetime.today().date()).replace('-', ''))), oof)
np.save(os.path.join('stacking', '{}_prediction_xgb'.format(str(datetime.datetime.today().date()).replace('-', ''))), prediction)

print('shape:', X.shape)
print('CV {0:} mean score: {1:.4f}, std: {2:.4f}, max: {3:.4f}, min: {4:.4f}.'.format(NFOLD, np.mean(scores), np.std(scores), np.max(scores), np.min(scores)))
print(features)

logger.info('''
# ============================================================================= 
# SUMMARY                                                     
# =============================================================================
''')
logger.info('shape: {}'.format(X.shape))
logger.info('CV {0:} mean score: {1:.4f}, std: {2:.4f}, max: {3:.4f}, min: {4:.4f}.'.format(NFOLD, np.mean(scores), np.std(scores), np.max(scores), np.min(scores)))
logger.info('{}'.format(features))
logger.info('''
# ============================================================================= 
# END                                              
# =============================================================================
''')

submission = pd.read_csv(os.path.join('..', 'input', 'sample_submission.csv'))
submission['target'] = prediction
submission.to_csv(os.path.join('..', 'submission', 'xgboost_{}.csv'.format(str(datetime.datetime.today().date()).replace('-', ''))), index=False)
#==============================================================================
utils.end(__file__)
