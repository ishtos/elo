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

PREF = 'f201'

KEY = 'card_id'

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
# common
# =============================================================================
new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')
new_merchant_transactions = new_merchant_transactions.query('purchase_amount < 80')
new_merchant_transactions.loc[new_merchant_transactions.installments == 999, 'installments'] = -1

# new_merchant_transactions['installments'] = new_merchant_transactions['installments'].astype('category')

# =============================================================================
# normal
# =============================================================================
# new_merchant_transactions['authorized_flag'] = new_merchant_transactions['authorized_flag'].apply(lambda x: np.where(x == 'Y', 1, 0))
# map_dict = {'Y': 0, 'N': 1}
# new_merchant_transactions['category_1'] = new_merchant_transactions['category_1'].apply(lambda x: map_dict[x]).astype('category')
# map_dict = {'A': 0, 'B': 1, 'C': 2, 'nan': 3}
# new_merchant_transactions['category_3'] = new_merchant_transactions['category_3'].apply(lambda x: map_dict[str(x)]).astype('category')

# =============================================================================
# normal
# =============================================================================
# new_merchant_transactions['category_2'].fillna(1.0,inplace=True)
# new_merchant_transactions['category_3'].fillna('A',inplace=True)
# new_merchant_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

# new_merchant_transactions['purchase_amount'] = new_merchant_transactions['purchase_amount'].apply(lambda x: min(x, 0.8))

new_merchant_transactions['authorized_flag'] = new_merchant_transactions['authorized_flag'].map({'Y': 1, 'N': 0})
new_merchant_transactions['category_1'] = new_merchant_transactions['category_1'].map({'Y': 1, 'N': 0})
new_merchant_transactions['category_3'] = new_merchant_transactions['category_3'].map({'A': 0, 'B': 1, 'C': 2, 'nan': 3})

new_merchant_transactions['purchase_date'] = pd.to_datetime(new_merchant_transactions['purchase_date'])
new_merchant_transactions['month'] = new_merchant_transactions['purchase_date'].dt.month
new_merchant_transactions['day'] = new_merchant_transactions['purchase_date'].dt.day
new_merchant_transactions['hour'] = new_merchant_transactions['purchase_date'].dt.hour
new_merchant_transactions['weekofyear'] = new_merchant_transactions['purchase_date'].dt.weekofyear
new_merchant_transactions['weekday'] = new_merchant_transactions['purchase_date'].dt.weekday
new_merchant_transactions['weekend'] = (new_merchant_transactions['purchase_date'].dt.weekday >= 5)

new_merchant_transactions['price'] = new_merchant_transactions['purchase_amount'] / new_merchant_transactions['installments']

new_merchant_transactions['Christmas_Day_2017']=(pd.to_datetime('2017-12-25') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_merchant_transactions['Mothers_Day_2017']=(pd.to_datetime('2017-06-04') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_merchant_transactions['fathers_day_2017']=(pd.to_datetime('2017-08-13') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_merchant_transactions['Children_day_2017']=(pd.to_datetime('2017-10-12') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_merchant_transactions['Valentine_Day_2017']=(pd.to_datetime('2017-06-12') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_merchant_transactions['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_merchant_transactions['Mothers_Day_2018']=(pd.to_datetime('2018-05-13') - new_merchant_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

new_merchant_transactions['month_diff'] = (datetime.date(2018, 2, 28) - new_merchant_transactions['purchase_date'].dt.date).dt.days // 30
new_merchant_transactions['month_diff'] += new_merchant_transactions['month_lag']

new_merchant_transactions['duration'] = new_merchant_transactions['purchase_amount'] * new_merchant_transactions['month_diff']
new_merchant_transactions['amount_month_ratio'] = new_merchant_transactions['purchase_amount'] / new_merchant_transactions['month_diff']

# for col in ['category_2','category_3']:
#     new_merchant_transactions[col+'_mean'] = new_merchant_transactions.groupby([col])['purchase_amount'].transform('mean')
#     new_merchant_transactions[col+'_sum'] = new_merchant_transactions.groupby([col])['purchase_amount'].transform('sum')

# =============================================================================
# normal
# =============================================================================
new_merchant_transactions = utils.reduce_mem_usage(new_merchant_transactions )
new_merchant_transactions.to_csv(os.path.join('..', 'remove_outlier_data', 'new_merchant_transactions.csv'), index=False)

# =============================================================================
utils.end(__file__)
