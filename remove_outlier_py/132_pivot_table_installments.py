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

PREF = 'f132'

SUMMARY = 30

KEY = 'card_id'

stats = ['min', 'max', 'mean', 'median', 'std', 'var', 'skew']

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

historical_transactions = pd.read_csv(os.path.join(PATH, 'historical_transactions.csv'))
historical_transactions['installments'] = historical_transactions['installments'].astype(int)
historical_transactions = historical_transactions.query('0 <= installments and installments <= 12')
# =============================================================================
#
# =============================================================================


def aggregate(args):
    prefix, index, columns, values = args['prefix'], args['index'], args['columns'], args['values']

    pt = historical_transactions.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc=['mean'])

    pt = pt.fillna(0).reset_index()
    pt.columns = [f'{c[0]}_{c[1]}_{c[2]}'.strip('_').replace('-', '') for c in pt.columns]
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
            'columns': 'month_lag',
            'values': ['installments']
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
