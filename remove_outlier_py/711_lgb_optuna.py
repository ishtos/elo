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
import optuna
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
#     'num_leaves': 31,
#     'min_data_in_leaf': 30, 
#     'objective':'regression',
#     'max_depth': -1,
#     'learning_rate': 0.01,
#     'boosting': 'gbdt',
#     'feature_fraction': 0.6,
#     'bagging_freq': 1,
#     'bagging_fraction': 0.8,
#     'bagging_seed': 11,
#     'metric': 'rmse',
#     'lambda_l1': 0.1,
#     'verbosity': -1,
#     'nthread': NTHREAD,
#     'random_state': SEED
# }

# =============================================================================
# features
# =============================================================================

features = []

features += [f'f10{i}.pkl' for i in (2,)]
# features += [f'f11{i}_{j}.pkl' for i in (1, 2) 
#                                for j in ('Y', 'N')]
features += [f'f12{i}.pkl' for i in (1, 2)]
# features += [f'f13{i}.pkl' for i in (1, 2)]

features += [f'f20{i}.pkl' for i in (2,)]
# features += [f'f23{i}.pkl' for i in (1, 2)]

features += [f'f30{i}.pkl' for i in (3, 4)]

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
# hand crafted 
# =============================================================================
date_features=['hist_purchase_date_max','hist_purchase_date_min', 'new_purchase_date_max', 'new_purchase_date_min']

for df in tqdm((train, test)):
    for f in date_features + ['first_active_month']:
        df[f] = pd.to_datetime(df[f])

    df['hist_first_buy'] = (df['hist_purchase_date_min'].dt.date - df['first_active_month'].dt.date).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'].dt.date - df['first_active_month'].dt.date).dt.days
    df['new_first_buy'] = (df['new_purchase_date_min'].dt.date - df['first_active_month'].dt.date).dt.days
    df['new_last_buy'] = (df['new_purchase_date_max'].dt.date - df['first_active_month'].dt.date).dt.days

    for f in date_features:
        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_transactions_count'] + df['hist_transactions_count']
    # df['card_id_cnt_total'] = df['new_card_id_count'] + df['hist_card_id_count']
    # df['card_id_cnt_ratio'] = df['new_card_id_count'] / df['hist_card_id_count']
    df['purchase_amount_total'] = df['new_purchase_amount_sum'] + df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_purchase_amount_mean'] + df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_purchase_amount_max'] + df['hist_purchase_amount_max']
    df['purchase_amount_min'] = df['new_purchase_amount_min'] + df['hist_purchase_amount_min']
    df['purchase_amount_ratio'] = df['new_purchase_amount_sum'] / df['hist_purchase_amount_sum']
    df['month_diff_mean'] = df['new_month_diff_mean'] + df['hist_month_diff_mean']
    df['month_diff_ratio'] = df['new_month_diff_mean'] / df['hist_month_diff_mean']
    # df['month_lag_mean'] = df['new_month_lag_mean'] + df['hist_month_lag_mean']
    # df['month_lag_max'] = df['new_month_lag_max'] + df['hist_month_lag_max']
    # df['month_lag_min'] = df['new_month_lag_min'] + df['hist_month_lag_min']
    df['category_1_mean'] = df['new_category_1_mean'] + df['hist_category_1_mean']
    df['installments_total'] = df['new_installments_sum'] + df['hist_installments_sum']
    df['installments_mean'] = df['new_installments_mean'] + df['hist_installments_mean']
    # df['installments_max'] = df['new_installments_max'] + df['hist_installments_max']
    df['installments_ratio'] = df['new_installments_sum'] / df['hist_installments_sum']
    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    # df['price_max'] = df['purchase_amount_max'] / df['installments_max']
    df['duration_mean'] = df['new_duration_mean'] + df['hist_duration_mean']
    df['duration_min'] = df['new_duration_min'] + df['hist_duration_min']
    df['duration_max'] = df['new_duration_max'] + df['hist_duration_max']
    df['amount_month_ratio_mean']= df['new_amount_month_ratio_mean'] + df['hist_amount_month_ratio_mean']
    df['amount_month_ratio_min']= df['new_amount_month_ratio_min'] + df['hist_amount_month_ratio_min']
    df['amount_month_ratio_max']= df['new_amount_month_ratio_max'] + df['hist_amount_month_ratio_max']
    df['new_CLV'] = df['new_transactions_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
    df['hist_CLV'] = df['hist_transactions_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']

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


col_not_to_use = ['first_active_month', 'card_id', 'target']
col_to_use = [c for c in train.columns if c not in col_not_to_use]

X = train[col_to_use]
X_test = test[col_to_use]

# ========= ====================================================================
# cv
# =============================================================================
def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True

NFOLDS = 5
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    num_leaves = trial.suggest_int('num_leaves', 10, 256)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 30, 300)
    bagging_freq = trial.suggest_int('bagging_freq', 1, 5) 
    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.5, 1.0)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0.4, 0.6)
    lambda_l1 = trial.suggest_uniform('lambda_l1', 0.0, 1.0)
    lambda_l2 = trial.suggest_uniform('lambda_l2', 0.0, 1.0)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': -1,
        'verbosity': -1,
        'boosting_type': 'gbdt',

        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'bagging_freq': bagging_freq,
        'bagging_fraction': bagging_fraction,
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,

        'bagging_seed': 11,
        'nthread': NTHREAD,
        'random_state': SEED
    }

    x_score = []
    final_cv_train = np.zeros(len(X))
    final_cv_pred = np.zeros(len(X_test))

    cv_train = np.zeros(len(X))
    cv_pred = np.zeros(len(X_test))

    params['seed'] = 0

    folds = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)

    best_trees = []
    fold_scores = []

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        dtrain = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index])
        dvalid = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index])

        model = lgb.train(
            params,
            dtrain,
            20000,          
            valid_sets=[dtrain, dvalid],
            verbose_eval=200,
            early_stopping_rounds=20)
       
        best_trees.append(model.best_iteration)
        cv_pred += model.predict(X_test, num_iteration=model.best_iteration)
        cv_train[valid_index] += model.predict(X.iloc[valid_index])

        score = Gini(y.iloc[valid_index], cv_train[valid_index])
        print(score)
        fold_scores.append(score)

    cv_pred /= NFOLDS
    final_cv_train += cv_train
    final_cv_pred += cv_pred

    print("cv score:")
    print(Gini(y, cv_train))
    print("current score:", Gini(y, final_cv_train / (fold_n + 1.)), fold_n + 1)
    print(fold_scores)
    print(best_trees, np.mean(best_trees))

    x_score.append(Gini(y, cv_train))
    print(x_score)

    pd.DataFrame({'card_id': test['card_id'], 'target': final_cv_pred / 16.}).to_csv(os.path.join('optuna', 'lgbm3_pred_avg_2.csv'), index=False)
    pd.DataFrame({'card_id': train['card_id'], 'target': final_cv_train / 16.}).to_csv(os.path.join('optuna', 'lgbm3_cv_avg_2.csv'), index=False)

    return (1 - x_score[0])

study = optuna.create_study()
study.optimize(objective, n_trials=100)

# print('shape:', X.shape)
# print('CV {0:} mean score: {1:.4f}, std: {2:.4f}, max: {3:.4f}, min: {4:.4f}.'.format(NFOLD, np.mean(scores), np.std(scores), np.max(scores), np.min(scores)))
# print(features)

# logger.info('''
# # ============================================================================= 
# # SUMMARY                                                     
# # =============================================================================
# ''')
# logger.info('shape: {}'.format(X.shape))
# logger.info('CV {0:} mean score: {1:.4f}, std: {2:.4f}, max: {3:.4f}, min: {4:.4f}.'.format(NFOLD, np.mean(scores), np.std(scores), np.max(scores), np.min(scores)))
# logger.info('{}'.format(features))
# logger.info('''
# # ============================================================================= 
# # END                                              
# # =============================================================================
# ''')

# submission = pd.read_csv(os.path.join('..', 'input', 'sample_submission.csv'))
# submission['target'] = prediction
# submission.to_csv(os.path.join('..', 'submission', 'lightgbm_outlier_{}.csv'.format(str(date.today()).replace('-', ''))), index=False)

# feature_importance['importance'] /= NFOLD
# cols = feature_importance[['feature', 'importance']].groupby('feature').mean().sort_values(by='importance', ascending=False)[:50].index

# best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
# best_features = best_features.sort_values(by='importance', ascending=False)
# feature_importance.sort_values(by='importance', ascending=False).to_csv('./IMP_csv/{}_IMP.csv'.format(str(date.today()).replace('-', '')), index=False)

# plt.figure(figsize=(14, 25))
# plt.title('LGB Features (avg over folds)')
# plot = sns.barplot(x='importance', y='feature', data=best_features)
# fig = plot.get_figure()
# fig.savefig('./IMP_png/{}_IMP.png'.format(str(date.today()).replace('-', '')), bbox_inches='tight')

# #==============================================================================
# utils.end(__file__)
