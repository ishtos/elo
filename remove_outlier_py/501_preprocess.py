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

features += [f'f10{i}.pkl' for i in (2, 4, 6, 7)]
# features += [f'f11{i}_{j}.pkl' for i in (1,) 
#                                for j in ('Y', 'N')]
# features += [f'f12{i}.pkl' for i in (1,)]
# features += [f'f13{i}.pkl' for i in (1,)]
features += [f'f14{i}.pkl' for i in (1,)]

features += [f'f20{i}.pkl' for i in (2, 4)]
# features += [f'f23{i}.pkl' for i in (1, 2)]

features += [f'f30{i}.pkl' for i in (2,)]

#==============================================================================
# train and test
#==============================================================================
# train = pd.read_csv(os.path.join(PATH, 'train.csv'))
# test = pd.read_csv(os.path.join(PATH, 'test.csv'))

# for f in tqdm(features):
#     t = pd.read_pickle(os.path.join('..', 'remove_outlier_feature', f))
#     train = pd.merge(train, t, on=KEY, how='left')
#     test = pd.merge(test, t, on=KEY, how='left')

train = pd.read_csv('../remove_outlier_data/historical_transactions.csv')
train['installments'].replace(-1, np.nan, inplace=True)
train['installments'].replace(999, np.nan, inplace=True)
train['purchase_amount'] = np.log1p(train['purchase_amount'] - train['purchase_amount'].min())

# new_merchant_transactions = pd.read_csv('../remove_outlier_data/new_merchant_transactions.csv')

#==============================================================================
# date to int
#==============================================================================
# cols = train.columns.values
# for f in [
#     'new_purchase_date_max', 'new_purchase_date_min',
#     'hist_purchase_date_max', 'hist_purchase_date_min',
# ]:
#     if f in cols:
#         train[f] = train[f].astype(np.int64) * 1e-9
#         test[f] = test[f].astype(np.int64) * 1e-9

# test['outliers'] = test['outliers'].fillna(0)

#==============================================================================
# transform dataformat
#==============================================================================
categories = [c for d, c in zip(train.dtypes, train.columns) if str(d).startswith('int')]
numerics = []
target = 'purchase_amount'
# numerics = [c for d, c in zip(train.dtypes, train.columns) if not str(d).startswith('int')]

pd.DataFrame(data=categories, columns=['ffm_cols']).to_csv('./ffm/ffm_cols.csv', index=False)

fname = "./ffm/historical_transactions.txt" 

currentcode = len(numerics)
catdict = {}
catcodes = {}
for x in numerics:
    catdict[x] = 0
for x in categories:
    catdict[x] = 1

noofrows = train.shape[0]
with open(fname, "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if((n % 100000) == 0):
            print('Row', n)
        datastring = ""
        datarow = train.iloc[r].to_dict()
        datastring += str(int(datarow[target]))

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
# with open("./ffm/alltestffm.txt", "w") as text_file:
#     for n, r in enumerate(range(noofrows)):
#         if((n % 100000) == 0):
#             print('Row', n)
#         datastring = ""
#         datarow = test.iloc[r].to_dict()
#         datastring += str(int(datarow[target]))

#         for i, x in enumerate(catdict.keys()):
#             if(catdict[x] == 0):
#                 datastring = datastring + " " + \
#                     str(i)+":" + str(i)+":" + str(datarow[x])
#             else:
#                 if(x not in catcodes):
#                     catcodes[x] = {}
#                     currentcode += 1
#                     catcodes[x][datarow[x]] = currentcode
#                 elif(datarow[x] not in catcodes[x]):
#                     currentcode += 1
#                     catcodes[x][datarow[x]] = currentcode

#                 code = catcodes[x][datarow[x]]
#                 datastring = datastring + " "+str(i)+":" + str(int(code))+":1"
#         datastring += '\n'
#         text_file.write(datastring)

#==============================================================================
utils.end(__file__)
