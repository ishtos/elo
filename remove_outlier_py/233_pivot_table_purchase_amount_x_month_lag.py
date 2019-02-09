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

PREF = 'f233'

SUMMARY = 30

KEY = 'card_id'

stats = ['min', 'max', 'mean']

# =============================================================================
#
# =============================================================================
PATH = os.path.join('..', 'remove_outlier_data')

new_merchant_transactions = pd.read_csv(os.path.join(PATH, 'new_merchant_transactions.csv'))
new_merchant_transactions['installments'].replace(-1, np.nan, inplace=True)
new_merchant_transactions['installments'].replace(999, np.nan, inplace=True)

new_merchant_transactions['purchase_amount'] = np.round(new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)

# =============================================================================
#
# =============================================================================


def aggregate(args):
    prefix, index, columns, values = args['prefix'], args['index'], args['columns'], args['values']

    pt = new_merchant_transactions.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=stats).reset_index()
    pt.columns = [f'{c[0]}_{c[1]}_{c[2]}'.strip('_') for c in pt.columns]
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
            'columns': 'installments',
            'values': ['purchase_amount']
        }
    ]

    pool = Pool(NTHREAD)
    callback = pool.map(aggregate, argss)
    pool.close()

#==============================================================================
utils.end(__file__)
