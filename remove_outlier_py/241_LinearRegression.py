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

PREF = 'f241'

KEY = 'card_id'

AGG = 'sum'

PATH = os.path.join('..', 'remove_outlier_data')

# =============================================================================
#
# =============================================================================
new_merchant_transactions = pd.read_csv(os.path.join(PATH, 'new_merchant_transactions.csv'), usecols=['card_id', 'purchase_amount', 'month_lag', 'purchase_date'])
new_merchant_transactions['purchase_amount'] = np.log1p(new_merchant_transactions['purchase_amount'] - new_merchant_transactions['purchase_amount'].min())

new_merchant_transactions = new_merchant_transactions.sort_values(by=['card_id', 'purchase_date'])[['card_id', 'month_lag', 'purchase_amount']]

num_aggregations = {
    'purchase_amount': AGG
}

new_merchant_transactions = new_merchant_transactions.groupby(['card_id', 'month_lag']).agg(num_aggregations).reset_index()
new_merchant_transactions.columns = [f'{c[0]}_{c[1]}'.strip('_') for c in new_merchant_transactions.columns]

columns = [c for c in new_merchant_transactions.columns if c != 'card_id']

# =============================================================================
#
# =============================================================================

def coef_and_intercept(card_id):
    x = new_merchant_transactions.loc[new_merchant_transactions.card_id == card_id].index
    y = new_merchant_transactions.loc[new_merchant_transactions.card_id == card_id][c]
    
    model = sm.OLS(y, sm.add_constant(x)).fit()
    
    res = {
        'card_id': card_id,
         'coef_'+AGG: model.params[1], 
         'intercept_'+AGG: model.params[0]
    }
    
    return res

# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    card_ids = new_merchant_transactions['card_id'].unique()

    pool = Pool(NTHREAD)
    result = pool.map(coef_and_intercept, card_ids)
    pool.close()

    df = pd.DataFrame(result)
    df.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

#==============================================================================
utils.end(__file__)