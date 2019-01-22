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
import xlearn as xl

from tqdm import tqdm
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from multiprocessing import cpu_count, Pool


utils.start(__file__)
#==============================================================================
param = {
    'task': 'reg', 
    'lr': 0.2, 
    'lambda': 0.002, 
    'metric': 'rmse'
}

#==============================================================================
#
#==============================================================================
ffm_model = xl.create_ffm()
ffm_model.setTrain("./ffm/alltrainffm.txt")

ffm_model.fit(param, "./ffm/model.out")

ffm_model.setTest("./ffm/alltrainffm.txt")
ffm_model.predict("./ffm/model.out", "./ffm/output.txt")
#==============================================================================
utils.end(__file__)