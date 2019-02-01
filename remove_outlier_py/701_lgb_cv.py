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
# Logger
#==============================================================================
from logging import getLogger, FileHandler, Formatter, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)

file_handler = FileHandler('log_outlier_{}'.format(str(date.today()).replace('-', '')))
formatter = Formatter('%(message)s')
file_handler.setFormatter(formatter)
file_handler.setLevel(DEBUG)

logger.addHandler(file_handler)
logger.propagate = False

#==============================================================================

PATH = os.path.join('..', 'remove_outlier_data')

KEY = 'card_id'

SEED = 6
# SEED = np.random.randint(9999)

NTHREAD = cpu_count()

NFOLD = 5

# params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting': 'gbdt',
#     'learning_rate': 0.01,
#     'max_depth': 8,
#     'num_leaves': 12,

#     'min_data_in_leaf': 27,
#     'reg_alpha': 0.3,
#     'reg_lambda': 0.3,
    
#     'feature_fraction': 0.9,
#     'nthread': NTHREAD,
#     'bagging_freq': 1,
#     'bagging_fraction': 0.8,
#     'verbose': -1,
#     'seed': SEED
# }

params = {
    'num_leaves': 31,
    'min_data_in_leaf': 30, 
    'objective':'regression',
    'max_depth': -1,
    'learning_rate': 0.01,
    'boosting': 'gbdt',
    'feature_fraction': 0.6,
    'bagging_freq': 1,
    'bagging_fraction': 0.8,
    'bagging_seed': 11,
    'metric': 'rmse',
    'lambda_l1': 0.1,
    'verbosity': -1,
    'nthread': NTHREAD,
    'random_state': SEED
}

# =============================================================================
# best feature
# =============================================================================

# features =  ['f103.pkl', 'f105.pkl', 'f109.pkl']
# features += ['f107_N.pkl', 'f107_Y.pkl', 
#              'f108_N.pkl', 'f108_Y.pkl']
# features += ['f203.pkl', 'f205.pkl', 'f209.pkl']
# features += ['f207_N.pkl', 'f207_Y.pkl', 
#              'f208_N.pkl', 'f208_Y.pkl']
# features += ['f302.pkl', 'f303.pkl', 'f304.pkl', 'f305.pkl', 'f306.pkl']
# features += ['f403.pkl', 'f404.pkl', 'f409.pkl', 'f411.pkl']
# features += ['f406_N.pkl', 'f406_Y.pkl', 
#              'f408_N.pkl', 'f408_Y.pkl']

# =============================================================================
# features
# =============================================================================

features = []

features += [f'f10{i}.pkl' for i in (2, 7, 8)]
# features += [f'f11{i}_{j}.pkl' for i in (1, 2) 
#                                for j in ('Y', 'N')]
# features += [f'f12{i}.pkl' for i in (1,)]
# features += [f'f13{i}.pkl' for i in (1,)]

features += [f'f20{i}.pkl' for i in (2, 5, 6)]
# features += [f'f23{i}.pkl' for i in (1, 2)]

# features += [f'f30{i}.pkl' for i in (2,)]

# features += [f'f40{i}.pkl' for i in (2, 3)]
# features += [f'f41{i}_{j}.pkl' for i in (1, 2)
#                                for j in ('Y', 'N')]
# features += [f'f42{i}.pkl' for i in (1, 2)]

# features += [f'f50{i}.pkl' for i in (2, )]

# features = os.listdir('../remove_outlier_feature')

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

date_features=[
    'hist_purchase_date_max','hist_purchase_date_min',
    'new_purchase_date_max', 'new_purchase_date_min'
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
df['month_diff_mean'] = df['new_month_diff_mean'] + df['hist_month_diff_mean']
df['month_diff_ratio'] = df['new_month_diff_mean'] / df['hist_month_diff_mean']
df['month_lag_mean'] = df['new_month_lag_mean'] + df['hist_month_lag_mean']
# df['month_lag_max'] = df['new_month_lag_max'] + df['hist_month_lag_max']
# df['month_lag_min'] = df['new_month_lag_min'] + df['hist_month_lag_min']
df['category_1_mean'] = df['new_category_1_mean'] + df['hist_category_1_mean']
# df['installments_total'] = df['new_installments_sum'] + df['hist_installments_sum']
df['installments_mean'] = df['new_installments_mean'] + df['hist_installments_mean']
# df['installments_max'] = df['new_installments_max'] + df['hist_installments_max']
# df['installments_ratio'] = df['new_installments_sum'] / df['hist_installments_sum']
# df['price_total'] = df['purchase_amount_total'] / df['installments_total']
df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
# df['price_max'] = df['purchase_amount_max'] / df['installments_max']
df['duration_mean'] = df['new_duration_mean'] + df['hist_duration_mean']
df['duration_min'] = df['new_duration_min'] + df['hist_duration_min']
df['duration_max'] = df['new_duration_max'] + df['hist_duration_max']
df['amount_month_ratio_mean'] = df['new_amount_month_ratio_mean'] + df['hist_amount_month_ratio_mean']
df['amount_month_ratio_min'] = df['new_amount_month_ratio_min'] + df['hist_amount_month_ratio_min']
df['amount_month_ratio_max'] = df['new_amount_month_ratio_max'] + df['hist_amount_month_ratio_max']
df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']

# hist_category = [
#     'hist_category_2_1_sum', 'hist_category_2_2_sum',
#     'hist_category_2_3_sum', 'hist_category_2_4_sum',
#     'hist_category_2_5_sum', 'hist_category_3_0_sum',
#     'hist_category_3_1_sum', 'hist_category_3_2_sum'
# ]

# new_category = [
#     'new_category_2_1_sum', 'new_category_2_2_sum',
#     'new_category_2_3_sum', 'new_category_2_4_sum', 
#     'new_category_2_5_sum', 'new_category_3_0_sum', 
#     'new_category_3_1_sum', 'new_category_3_2_sum'
# ]

# for i, j in zip(hist_category, new_category):
#     df[i[5:]] = df[i] / df[j]

train = df[df['target'].notnull()]
test = df[df['target'].isnull()]
del df
gc.collect()

# =============================================================================
# change date to int
# =============================================================================
# cols = train.columns.values
# for f in [
#     'new_purchase_date_max', 'new_purchase_date_min',
#     'hist_purchase_date_max', 'hist_purchase_date_min', 
#     'Y_hist_auth_purchase_date_max', 'Y_hist_auth_purchase_date_min', 
#     'N_hist_auth_purchase_date_max', 'N_hist_auth_purchase_date_min',
#     'Y_new_auth_purchase_date_max', 'Y_new_auth_purchase_date_min', 
#     'N_new_auth_purchase_date_max', 'N_new_auth_purchase_date_min',
# ]:
#     if f in cols:
#         train[f] = train[f].astype(np.int64) * 1e-9
#         test[f] = test[f].astype(np.int64) * 1e-9

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
    'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
    'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size'
]
col_to_use = [c for c in train.columns if c not in col_not_to_use]


gc.collect()

X = train[col_to_use]
X_test = test[col_to_use]

# ========= ====================================================================
# cv
# =============================================================================
folds = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

oof = np.zeros(len(X))
prediction = np.zeros(len(X_test))

scores = []

feature_importance = pd.DataFrame()

for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    dtrain = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index])
    dvalid = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index])

    params = {
        'task': 'train',
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

    fold_importance = pd.DataFrame()
    fold_importance['feature'] = X.columns
    fold_importance['importance'] = np.log1p(model.feature_importance(importance_type='gain', iteration=model.best_iteration))
    fold_importance['fold'] = fold_n + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

np.save(os.path.join('stacking', '{}_oof_lgb'.format(str(date.today()).replace('-', ''))), oof)
np.save(os.path.join('stacking', '{}_prediction_lgb'.format(str(date.today()).replace('-', ''))), prediction)

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
submission.to_csv(os.path.join('..', 'submission', 'lightgbm_outlier_{}.csv'.format(str(date.today()).replace('-', ''))), index=False)

feature_importance['importance'] /= NFOLD
cols = feature_importance[['feature', 'importance']].groupby('feature').mean().sort_values(by='importance', ascending=False)[:50].index

best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
best_features = best_features.sort_values(by='importance', ascending=False)
feature_importance.sort_values(by='importance', ascending=False).to_csv('./IMP_csv/{}_IMP.csv'.format(str(date.today()).replace('-', '')), index=False)

plt.figure(figsize=(14, 25))
plt.title('LGB Features (avg over folds)')
plot = sns.barplot(x='importance', y='feature', data=best_features)
fig = plot.get_figure()
fig.savefig('./IMP_png/{}_IMP.png'.format(str(date.today()).replace('-', '')), bbox_inches='tight')

#==============================================================================
utils.end(__file__)
