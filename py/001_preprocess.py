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

PATH = os.path.join('..', 'input')

# =============================================================================
#
# =============================================================================
files = os.listdir(PATH)

for f in files:
    print(f)
    if f.endswith('.xlsx'):
        continue
    df = pd.read_csv(os.path.join(PATH, f))
    df = utils.reduce_mem_usage(df)
    df.to_csv(f'../data/{f}', index=False)

# =============================================================================
utils.end(__file__)
