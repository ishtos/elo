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

PREF = 'f421'

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew', 'count']

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

union = pd.read_csv(os.path.join(PATH, 'union.csv'), usecols=['card_id', 'purchase_date', 'authorized_flag'])

union.purchase_date = pd.to_datetime(union.purchase_date)
union['month'] = union.purchase_date.dt.month
union['year'] = union.purchase_date.dt.year

union = union[['authorized_flag', 'card_id', 'month', 'year']]
union['cnt'] = 1

ug = union.groupby(['authorized_flag', 'card_id', 'month', 'year'])['cnt'].count().reset_index()
ug_y = ug[ug.authorized_flag == 1].reset_index(drop=True)
ug_n = ug[ug.authorized_flag == 0].reset_index(drop=True)

pt_y = ug_y.pivot_table(index='card_id', columns=['month', 'year'], values='cnt').reset_index().fillna(0)
pt_n = ug_n.pivot_table(index='card_id', columns=['month', 'year'], values='cnt').reset_index().fillna(0)
del ug, ug_y, ug_n
gc.collect()

pt_y.columns = [f'{c[0]}_{c[1]}' for c in pt_y.columns]
pt_y = pt_y.rename(columns={'card_id_': 'card_id'})

pt_n.columns = [f'{c[0]}_{c[1]}' for c in pt_n.columns] 
pt_n = pt_n.rename(columns={'card_id_': 'card_id'})

pt = pd.merge(pt_y, pt_n, on='card_id', how='left')
del pt_y, pt_n
gc.collect()

columns = [
    '1_2017_', '2_2017_', '3_2017_', '4_2017_', 
    '5_2017_', '6_2017_', '7_2017_', '8_2017_',
    '9_2017_', '10_2017_', '11_2017_', '12_2017_',
    '1_2018_', '2_2018_',
]

pt = pt.fillna(0)
for e, c in tqdm(enumerate(columns)):
    pt[c+'rate'] = pt[c+'x'] / (pt[c+'x'] + pt[c+'y'] + 1e-19)
pt['3_2018_rate'] = 1.0 
pt['4_2018_rate'] = 1.0 

columns = [
    '1_2017_rate', '2_2017_rate', '3_2017_rate', '4_2017_rate', 
    '5_2017_rate', '6_2017_rate', '7_2017_rate', '8_2017_rate',
    '9_2017_rate', '10_2017_rate', '11_2017_rate', '12_2017_rate',
    '1_2018_rate', '2_2018_rate', '3_2018_rate', '4_2018_rate'
]

pt = pt[['card_id'] + columns]

for c1, c2 in zip(columns, columns[1:]):
    if (c1 == 'card_id') or (c2 == 'card_id'):
        continue
    pt[c2] = pt[c1] + pt[c2]

pt = pt.add_prefix('union_')
pt = pt.rename(columns={'union_card_id': 'card_id'})

pt.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')
#==============================================================================
utils.end(__file__)
