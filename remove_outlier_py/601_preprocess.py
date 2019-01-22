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
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
NTHREAD = cpu_count()

# os.system(f'rm ../feature/{PREF}_train.pkl')
# os.system(f'rm ../feature/{PREF}_test.pkl')

PATH = os.path.join('..', 'remove_outlier_feature')

# =============================================================================
#
# =============================================================================
files = os.listdir(PATH)

for f in files:
    print(f)
    df = pd.read_pickle(os.path.join(PATH, f))
    df = utils.reduce_mem_usage(df)
    df.to_pickle(f'../remove_outlier_feature/{f}')    

# =============================================================================
utils.end(__file__)
