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
import datetime

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f101'

KEY = 'card_id'

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
# common
# =============================================================================
historical_transactions = pd.read_csv('../input/historical_transactions.csv')
historical_transactions = historical_transactions.query('purchase_amount < 6000000')
historical_transactions.loc[historical_transactions.installments == 999, 'installments'] = -1

# historical_transactions['installments'] = historical_transactions['installments'].astype('category')

# =============================================================================
# normal
# =============================================================================
# historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].apply(lambda x: np.where(x == 'Y', 1, 0))
# map_dict = {'Y': 0, 'N': 1}
# historical_transactions['category_1'] = historical_transactions['category_1'].apply(lambda x: map_dict[x]).astype('category')
# map_dict = {'A': 0, 'B': 1, 'C': 2, 'nan': 3}
# historical_transactions['category_3'] = historical_transactions['category_3'].apply(lambda x: map_dict[str(x)]).astype('category')

# =============================================================================
# fillna
# =============================================================================
# historical_transactions['category_2'].fillna(1.0,inplace=True)
# historical_transactions['category_3'].fillna('A',inplace=True)
# historical_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

# historical_transactions['purchase_amount'] = historical_transactions['purchase_amount'].apply(lambda x: min(x, 0.8))

historical_transactions['authorized_flag'] = historical_transactions['authorized_flag'].map({'Y': 1, 'N': 0})
historical_transactions['category_1'] = historical_transactions['category_1'].map({'Y': 1, 'N': 0})
historical_transactions['category_3'] = historical_transactions['category_3'].map({'A': 0, 'B': 1, 'C': 2, 'nan': 3})

historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
historical_transactions['month'] = historical_transactions['purchase_date'].dt.month
historical_transactions['day'] = historical_transactions['purchase_date'].dt.day
historical_transactions['hour'] = historical_transactions['purchase_date'].dt.hour
historical_transactions['weekofyear'] = historical_transactions['purchase_date'].dt.weekofyear
historical_transactions['weekday'] = historical_transactions['purchase_date'].dt.weekday
historical_transactions['weekend'] = (historical_transactions['purchase_date'].dt.weekday >= 5)

historical_transactions['price'] = historical_transactions['purchase_amount'] / historical_transactions['installments']

historical_transactions['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
historical_transactions['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
historical_transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
historical_transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
historical_transactions['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
historical_transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
historical_transactions['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

historical_transactions['month_diff'] = (datetime.date(2018, 2, 28) - historical_transactions['purchase_date'].dt.date).dt.days // 30
historical_transactions['month_diff'] += historical_transactions['month_lag']

historical_transactions['duration'] = historical_transactions['purchase_amount'] * historical_transactions['month_diff']
historical_transactions['amount_month_ratio'] = historical_transactions['purchase_amount'] / historical_transactions['month_diff']

# for col in ['category_2','category_3']:
    # historical_transactions[col+'_mean'] = historical_transactions.groupby([col])['purchase_amount'].transform('mean')
    # historical_transactions[col+'_sum'] = historical_transactions.groupby([col])['purchase_amount'].transform('sum')

# =============================================================================
# 
# =============================================================================
historical_transactions = utils.reduce_mem_usage(historical_transactions)
historical_transactions.to_csv(os.path.join('..', 'remove_outlier_data', 'historical_transactions.csv'), index=False)

# =============================================================================
utils.end(__file__)
