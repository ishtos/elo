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
PATH = os.path.join('..', 'remove_outlier_data')

KEY = 'card_id'

#==============================================================================
# features
#==============================================================================
features = []

features += [f'f10{i}.pkl' for i in (2, 3)]
# features += [f'f11{i}_{j}.pkl' for i in (1, 2)
#                                for j in ('Y', 'N')]
# features += [f'f12{i}.pkl' for i in (1, 2)]


features += [f'f20{i}.pkl' for i in (2, 3)]
# features += [f'f21{i}_{j}.pkl' for i in (1, 2)
#                                for j in ('Y', 'N')]

features += [f'f40{i}.pkl' for i in (2, 3)]
# features += [f'f41{i}_{j}.pkl' for i in (1, 2)
#                                for j in ('Y', 'N')]
# features += [f'f42{i}.pkl' for i in (1, 2)]


# features = os.listdir('../remove_outlier_feature')

#==============================================================================
# train and test
#==============================================================================
train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

for f in tqdm(features):
    t = pd.read_pickle(os.path.join('..', 'remove_outlier_feature', f))
    train = pd.merge(train, t, on=KEY, how='left')
    test = pd.merge(test, t, on=KEY, how='left')

test.insert(1, 'outlier', 0)
train['outlier'] = 0
train.loc[train.target < -30, 'outlier'] = 1

#==============================================================================
# date to int
#==============================================================================
cols = train.columns.values
for f in [
    'new_purchase_date_max', 'new_purchase_date_min',
    'hist_purchase_date_max', 'hist_purchase_date_min',
    'N_hist_auth_purchase_date_max', 'N_hist_auth_purchase_date_min',
    'Y_hist_auth_purchase_date_max', 'Y_hist_auth_purchase_date_min',
    'Y_new_auth_purchase_date_max', 'Y_new_auth_purchase_date_min',
    'N_new_auth_purchase_date_max', 'N_new_auth_purchase_date_min',
    'Y_new_auth_purchase_date_max_x', 'Y_new_auth_purchase_date_min_x',
    'N_new_auth_purchase_date_max_x', 'N_new_auth_purchase_date_min_x',
    'Y_new_auth_purchase_date_max_y', 'Y_new_auth_purchase_date_min_y',
    'N_new_auth_purchase_date_max_y', 'N_new_auth_purchase_date_min_y'
]:
    if f in cols:
        train[f] = train[f].astype(np.int64) * 1e-9
        test[f] = test[f].astype(np.int64) * 1e-9

#==============================================================================
# to bins
#==============================================================================
# x = pd.concat([train, test])
# x = x.reset_index(drop=True)

# cut_cols = []
# for c in x.columns[2:]:
#     trainno = len(x.loc[:train.shape[0], c].unique())
#     testno = len(x.loc[train.shape[0]:, c].unique())
#     print(c, trainno, testno)
#     if (trainno > 1000) or (testno > 1000):
#         cut_cols.append(c)

# for c in cut_cols:
#     x[c] = pd.cut(x[c], 50, labels=False)

# train = x.loc[:train.shape[0]].copy()
# test = x.loc[train.shape[0]:].copy()

#==============================================================================
# transform dataformat
#==============================================================================
categories = [c for d, c in zip(train.dtypes, train.columns) if str(d).startswith('int')]
numerics = []

pd.DataFrame(data=categories, columns=['ffm_cols']).to_csv('./ffm/ffm_cols.csv', index=False)

currentcode = len(numerics)
catdict = {}
catcodes = {}
for x in numerics:
    catdict[x] = 0
for x in categories:
    catdict[x] = 1

noofrows = train.shape[0]
with open("./ffm/alltrainffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n % 100000) == 0):
            print('Row', n)
        datastring = ""
        datarow = train.iloc[r].to_dict()
        datastring += str(int(datarow['outlier']))

        for i, x in enumerate(catdict.keys()):
            if(catdict[x] == 0):
                datastring = datastring + " " + \
                    str(i)+":" + str(i)+":" + str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode += 1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode += 1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":" + str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)

noofrows = test.shape[0]
with open("./ffm/alltestffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n % 100000) == 0):
            print('Row', n)
        datastring = ""
        datarow = test.iloc[r].to_dict()
        datastring += str(int(datarow['outlier']))

        for i, x in enumerate(catdict.keys()):
            if(catdict[x] == 0):
                datastring = datastring + " " + \
                    str(i)+":" + str(i)+":" + str(datarow[x])
            else:
                if(x not in catcodes):
                    catcodes[x] = {}
                    currentcode += 1
                    catcodes[x][datarow[x]] = currentcode
                elif(datarow[x] not in catcodes[x]):
                    currentcode += 1
                    catcodes[x][datarow[x]] = currentcode

                code = catcodes[x][datarow[x]]
                datastring = datastring + " "+str(i)+":" + str(int(code))+":1"
        datastring += '\n'
        text_file.write(datastring)

#==============================================================================
utils.end(__file__)
