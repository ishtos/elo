#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 2018

@author: toshiki.ishikawa
"""

import os
import sys
import gc
import utils
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f141'

KEY = 'card_id'

PATH = os.path.join('..', 'remove_outlier_data')

# =============================================================================
#
# =============================================================================
historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'), usecols=['card_id', 'purchase_amount', 'month_lag', 'purchase_date'])
historical_transactions['purchase_amount'] = np.log1p(historical_transactions['purchase_amount'] - historical_transactions['purchase_amount'].min())

historical_transactions = historical_transactions.sort_values(by=['card_id', 'purchase_date'])[['card_id', 'month_lag', 'purchase_amount']]

num_aggregations = {
    'purchase_amount': ['sum', 'mean']
}

historical_transactions = historical_transactions.groupby(['card_id', 'month_lag']).agg(num_aggregations).reset_index()
historical_transactions.columns = [f'{c[0]}_{c[1]}'.strip('_') for c in historical_transactions.columns]

columns = [c for c in historical_transactions.columns if c != 'card_id']

# =============================================================================
#
# =============================================================================

# historical_transactions = historical_transactions.sort_values(by=['card_id', 'purchase_date'])[['card_id', 'month_lag', 'purchase_amount']]
# card_ids = historical_transactions['card_id'].unique()
# out_dfs = [None] * len(card_ids)

# num_aggregations = {
#     'purchase_amount': [
#         'sum', 'mean', 
#         # 'max', 'min',
#         # 'var', 'skew'
#     ]
# }

# historical_transactions = historical_transactions.groupby(['card_id', 'month_lag']).agg(num_aggregations).reset_index()
# historical_transactions.columns = [f'{c[0]}_{c[1]}'.strip('_') for c in historical_transactions.columns]

# columns = [
#     'purchase_amount_sum', 'purchase_amount_mean',
#     # 'purchase_amount_max', 'purchase_amount_min',
#     # 'purchase_amount_var', 'purchase_amount_skew'
# ]

# for i, card_id in enumerate(tqdm(card_ids)):
#     out_dfs[i] = {'card_id': card_id}
#     x = list(historical_transactions.loc[historical_transactions.card_id == card_id].index) 
    
#     for c in columns: 
#         y = list(historical_transactions.loc[historical_transactions.card_id == card_id][c]) 

#         model = sm.OLS(y, sm.add_constant(x)).fit()
#         out_dfs[i].update({'coef_'+c[16:]: model.params[1], 'intercept_'+c[16:]: model.params[0]}) 

# # reg = LinearRegression(n_jobs=-1)
# # for i, card_id in enumerate(tqdm(card_ids)):
# #     out_dfs[i] = {'card_id': card_id}
# #     x = list(historical_transactions.loc[historical_transactions.card_id == card_id].index)
# #     x = [[x] for x in x]
# #     for c in columns: 
# #         y = list(historical_transactions.loc[historical_transactions.card_id == card_id][c]) 
# #         y = [[y] for y in y]
        
# #         pred = reg.fit(x, y)
# #         out_dfs[i].update({'coef_'+c[16:]: pred.coef_[0][0], 'intercept_'+c[16:]: pred.intercept_[0]})
    
# columns_names = [
#     'card_id', 
#     'coef_sum', 'intercept_sum',
#     'coef_mean', 'intercept_mean', 
#     # 'coef_max', 'intercept_max', 
#     # 'coef_min', 'intercept_min', 
#     # 'coef_var', 'intercept_var', 
#     # 'coef_skew', 'intercept_skew'
# ]

# reg_df = pd.concat([pd.DataFrame([out_dfs[i]], columns=columns_names) for i in tqdm(range(len(list(filter(None, out_dfs)))))], ignore_index=True)
# reg_df.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

# =============================================================================
#
# =============================================================================

def coef_and_intercept(card_id):
    res = {'card_id': card_id}
    x = list(historical_transactions.loc[historical_transactions.card_id == card_id].index)
    for c in columns: 
        y = list(historical_transactions.loc[historical_transactions.card_id == card_id][c]) 
        model = sm.OLS(y, sm.add_constant(x)).fit()
        res.update({'coef_'+c[16:]: model.params[1], 'intercept_'+c[16:]: model.params[0]})
    
    return res

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    card_ids = historical_transactions['card_id'].unique()

    pool = Pool(NTHREAD)
    result = pool.map(coef_and_intercept, card_ids)
    pool.close()

#==============================================================================
utils.end(__file__)

