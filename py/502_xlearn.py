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
PATH = os.path.join('..', 'data')

PREF = 'f502'

param = {
    'task': 'binary', 
    'opt': 'adagrad',
    'lr': 0.2, 
    'lambda': 0.002, 
    'metric': 'auc', 
    'fold': 5,
    'epoch': 10,
}

CV = False
# CV = True

#==============================================================================
#
#==============================================================================
ffm_model = xl.create_ffm()
ffm_model.setTrain('./ffm/alltrainffm.txt')

if CV:
    ffm_model.cv(param)
    ftrain = 'cv_train'
    ftest = 'cv_test'
else:
    ffm_model.fit(param, './ffm/model.out')
    ftrain = 'train'
    ftest = 'test'

ffm_model.setTest('./ffm/alltestffm.txt')
ffm_model.predict('./ffm/model.out', f'./ffm/{ftest}.txt')

ffm_model.setTest('./ffm/alltrainffm.txt')
ffm_model.predict('./ffm/model.out', f'./ffm/{ftrain}.txt')

#==============================================================================
# to pkl
#==============================================================================
train = pd.read_csv(os.path.join(PATH, 'train.csv'), usecols=['card_id'])
test = pd.read_csv(os.path.join(PATH, 'test.csv'), usecols=['card_id'])
df = pd.concat([train, test], axis=0)

ffm_train = pd.read_csv(f'./ffm/{ftrain}.txt', header=None)
ffm_test = pd.read_csv(f'./ffm/{ftest}.txt', header=None)
ffm_df = pd.concat([ffm_train, ffm_test])

ffm_df['card_id'] = df['card_id']
ffm_df = ffm_df.rename(columns={0: 'outliers'})

ffm_df.to_pickle(f'../feature/{PREF}.pkl')

#==============================================================================
# submission
#==============================================================================
# outputs = pd.read_csv('output.txt', header=None)
# outputs.columns = ['target']

# sub.target = outputs.target.ravel()
# sub.to_csv('libffmsubmission.csv',index=False)

# sub = pd.read_csv('../input/sample_submission.csv')
# outputs = pd.read_csv('output.txt', header=None)
# outputs.columns = ['target']
# sub.target = outputs.target
# sub.to_csv('../submission/libffmsubmission.csv',index=False)

#==============================================================================
utils.end(__file__)
