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
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

PREF = 'f401_'

KEY = 'card_id'

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
historical_transactions = pd.read_csv('../data/historical_transactions.csv')
new_merchant_transactions = pd.read_csv('../data/new_merchant_transactions.csv')

union = pd.concat([historical_transactions, new_merchant_transactions], axis=0)

union.to_csv(os.path.join('..', 'data', 'union.csv'), index=False)

# =============================================================================
utils.end(__file__)
