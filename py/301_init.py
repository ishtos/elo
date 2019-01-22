#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 2018

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

PREF = 'f301_'

KEY = 'card_id'

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

# =============================================================================
#
# =============================================================================
merchants = pd.read_csv('../input/merchants.csv')

map_dict = {'N': 0, 'Y': 1}
merchants['category_1'] = merchants['category_1'].apply(lambda x: map_dict[x])
merchants['category_4'] = merchants['category_4'].apply(lambda x: map_dict[x])
map_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
merchants['most_recent_sales_range'] = merchants['most_recent_sales_range'].apply(lambda x: map_dict[x])
merchants['most_recent_purchases_range'] = merchants['most_recent_purchases_range'].apply(lambda x: map_dict[x])

merchants = utils.reduce_mem_usage(merchants)
merchants.to_csv(os.path.join('..', 'data', 'merchants.csv'), index=False)

# =============================================================================
utils.end(__file__)
