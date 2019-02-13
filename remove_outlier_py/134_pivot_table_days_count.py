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

from tqdm import tqdm
# from datetime import datetime, date
import datetime
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f134'

SUMMARY = 30

KEY = 'card_id'

stats = 'count'

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'), usecols=['purchase_date', 'card_id'])

grouped = historical_transactions.groupby('card_id')['purchase_date'].min().reset_index()
grouped.rename(columns={'purchase_date': 'purchase_date_min'}, inplace=True)

historical_transactions = pd.merge(historical_transactions, grouped, on='card_id', how='left')
historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
historical_transactions['purchase_date_min'] = pd.to_datetime(historical_transactions['purchase_date_min'])
historical_transactions['days'] = (historical_transactions['purchase_date'].dt.date - historical_transactions['purchase_date_min'].dt.date).dt.days
historical_transactions['round_days'] = (np.ceil(historical_transactions['days'] // 10) * 10).astype(int)
historical_transactions['agg_flag'] = 1

# =============================================================================
#
# =============================================================================


def aggregate(args):
    prefix, index, columns, values = args['prefix'], args['index'], args['columns'], args['values']

    pt = historical_transactions.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=stats).reset_index()
    pt.columns = [f'{c[0]}_{c[1]}'.strip('_') for c in pt.columns]
    pt = pt.add_prefix(prefix)
    pt = pt.rename(columns={prefix+KEY: KEY})

    pt.to_pickle(f'../remove_outlier_feature/{PREF}.pkl')

    return


# =============================================================================
#
# =============================================================================
if __name__ == '__main__':
    argss = [
        {
            'prefix': 'hist_',
            'index': 'card_id',
            'columns': ['round_days'],
            'values': ['agg_flag']
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
