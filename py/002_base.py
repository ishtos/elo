#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 2018

@author: toshiki.ishikawa
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import gc
import utils
import numpy as np
import pandas as pd

from tqdm import tqdm
from time import time, sleep
from datetime import datetime
from itertools import combinations
from multiprocessing import cpu_count, Pool


utils.start(__file__)

#==============================================================================

PATH = os.path.join('..', 'input')

union = pd.read_csv('../data/union.csv', usecols=['card_id', 'merchant_id'])

#==============================================================================
# union
#==============================================================================

union = union.drop_duplicates(keep='first').reset_index()
union.to_csv(os.path.join('..', 'data', 'union_base.csv'), index=False)

#==============================================================================

utils.end(__file__)
