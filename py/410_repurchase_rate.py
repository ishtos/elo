#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 2018

@author: toshiki.ishikawa
"""

import os
import sys
import gc
import utils
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f410'

KEY = 'card_id'

# stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', 'count']
stats = ['min', 'max', 'mean', 'count']


# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'data')

union = pd.read_csv(os.path.join(PATH, 'union.csv'))

union.purchase_date = pd.to_datetime(union.purchase_date)
union['month'] = union.purchase_date.dt.month
union['year'] = union.purchase_date.dt.year
union = union[['card_id', 'merchant_id', 'month', 'year']]
union['cnt'] = 1

ug = union.groupby(['card_id', 'merchant_id', 'month', 'year'])['cnt'].count().reset_index()
ugg = ug.groupby(['card_id', 'month', 'year']).agg({'cnt': stats})
del ug
gc.collect()

ugg = ugg.reset_index()
ugg.columns = [f'{c[0]}_{c[1]}' for c in ugg.columns]
ugg = ugg.rename(columns={'card_id_': 'card_id', 
                          'month_': 'month', 
                          'year_': 'year'})
pt = ugg.pivot_table(index='card_id', 
                     columns=['month', 'year'],
                     values=['cnt_min', 'cnt_max',
                             'cnt_mean', 'cnt_count'])
                #      values=['cnt_min', 'cnt_max', 
                #              'cnt_mean', 'cnt_median', 
                #              'cnt_std', 'cnt_var',
                #              'cnt_skew', 'cnt_count'])
del ugg
gc.collect()

pt.columns = [f'{c[0]}_{c[1]}_{c[2]}' for c in pt.columns]
pt = pt.reset_index()
pt = pt.fillna(0)

pt.to_pickle(f'../feature/{PREF}.pkl')

#==============================================================================
utils.end(__file__)
