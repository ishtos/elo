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
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from time import time, sleep
from datetime import datetime
from multiprocessing import cpu_count, Pool
from sklearn.model_selection import KFold

# =============================================================================
# global variables
# =============================================================================

COMPETHITION_NAME = 'Elo'

SPLIT_SIZE = 20


# =============================================================================
# def
# =============================================================================

def start(fname):
    global st_time
    st_time = time()
    print("""
#==============================================================================
# START!!! {}    PID: {}    time: {}
#==============================================================================
""".format( fname, os.getpid(), datetime.today() ))
    return

def reset_time():
    global st_time
    st_time = time()
    return

def end(fname):
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( elapsed_minute() ))
    return

def elapsed_minute():
    return (time() - st_time)/60


def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

def to_feature(df, path):
    
    if df.columns.duplicated().sum() > 0:
        raise Exception(f'duplicated!: { df.columns[df.columns.duplicated()] }')
    df.reset_index(inplace=True, drop=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather(f'{path}_{c}.f')
    return

def read_feature(df, path):
    # TODO: implementaion
    return 

def to_pickles(df, path, split_size=3, inplace=True):
    """
    path = '../output/mydf'
    
    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    
    """
    print(f'shape: {df.shape}')
    
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    gc.collect()
    mkdir_p(path)
    
    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(f'{path}/{i:03d}.p')
    return

def read_pickles(path, col=None, use_tqdm=True):
    if col is None:
        if use_tqdm:
            df = pd.concat([ pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*'))) ])
        else:
            print(f'reading {path}')
            df = pd.concat([ pd.read_pickle(f) for f in sorted(glob(path+'/*')) ])
    else:
        df = pd.concat([ pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*'))) ])
    return df

def load_train(col=None):
    if col is None:
        return read_pickles('../data/train')
    else:
        return read_pickles('../data/train', col)

def load_test(col=None):
    if col is None:
        return read_pickles('../data/test')
    else:
        return read_pickles('../data/test', col)

def merge(df, col):
    trte = pd.concat([load_train(col=col),
                      load_test(col=col)])
    df_ = pd.merge(df, trte, on='SK_ID_CURR', how='left')
    return df_

def check_feature():
    
    sw = False
    files = sorted(glob('../feature/train*.f'))
    for f in files:
        path = f.replace('train_', 'test_')
        if not os.path.isfile(path):
            print(f)
            sw = True
    
    files = sorted(glob('../feature/test*.f'))
    for f in files:
        path = f.replace('test_', 'train_')
        if not os.path.isfile(path):
            print(f)
            sw = True
    
    if sw:
        raise Exception('Miising file :(')
    else:
        print('All files exist :)')

# =============================================================================
# 
# =============================================================================
def get_dummies(df):
    """
    binary would be drop_first
    """
    col = df.select_dtypes('O').columns.tolist()
    nunique = df[col].nunique()
    col_binary = nunique[nunique==2].index.tolist()
    [col.remove(c) for c in col_binary]
    df = pd.get_dummies(df, columns=col)
    df = pd.get_dummies(df, columns=col_binary, drop_first=True)
    df.columns = [c.replace(' ', '-') for c in df.columns]
    return df


# def reduce_mem_usage(df):
#     col_int8 = []
#     col_int16 = []
#     col_int32 = []
#     col_int64 = []
#     col_float16 = []
#     col_float32 = []
#     col_float64 = []
#     col_cat = []
#     for c in tqdm(df.columns, mininterval=20):
#         col_type = df[c].dtype

#         if col_type != object:
#             c_min = df[c].min()
#             c_max = df[c].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     col_int8.append(c)
                    
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     col_int16.append(c)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     col_int32.append(c)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     col_int64.append(c)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     col_float16.append(c)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     col_float32.append(c)
#                 else:
#                     col_float64.append(c)
#         else:
#             col_cat.append(c)
    
#     if len(col_int8)>0:
#         df[col_int8] = df[col_int8].astype(np.int8)
#     if len(col_int16)>0:
#         df[col_int16] = df[col_int16].astype(np.int16)
#     if len(col_int32)>0:
#         df[col_int32] = df[col_int32].astype(np.int32)
#     if len(col_int64)>0:
#         df[col_int64] = df[col_int64].astype(np.int64)
#     if len(col_float16)>0:
#         df[col_float16] = df[col_float16].astype(np.float16)
#     if len(col_float32)>0:
#         df[col_float32] = df[col_float32].astype(np.float32)
#     if len(col_float64)>0:
#         df[col_float64] = df[col_float64].astype(np.float64)
#     if len(col_cat)>0:
#         df[col_cat] = df[col_cat].astype('category')

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def to_pkl_gzip(df, path):
    df.to_pickle(path)
    os.system('gzip ' + path)
    os.system('rm ' + path)
    return

def check_var(df, var_limit=0, sample_size=None):
    if sample_size is not None:
        if df.shape[0]>sample_size:
            df_ = df.sample(sample_size, random_state=71)
        else:
            df_ = df
    else:
        df_ = df
        
    var = df_.var()
    col_var0 = var[var<=var_limit].index
    if len(col_var0)>0:
        print(f'remove var<={var_limit}: {col_var0}')
    return col_var0

def check_corr(df, corr_limit=1, sample_size=None):
    if sample_size is not None:
        if df.shape[0]>sample_size:
            df_ = df.sample(sample_size, random_state=71)
        else:
            raise Exception(f'df:{df.shape[0]} <= sample_size:{sample_size}')
    else:
        df_ = df
    
    corr = df_.corr('pearson').abs()
    a, b = np.where(corr>=corr_limit)
    col_corr1 = []
    for a_,b_ in zip(a, b):
        if a_ != b_ and a_ not in col_corr1:
            col_corr1.append(b_)
    if len(col_corr1)>0:
        col_corr1 = df.iloc[:,col_corr1].columns
        print(f'remove corr>={corr_limit}: {col_corr1}')
    return col_corr1

def remove_feature(df, var_limit=0, corr_limit=1, sample_size=None, only_var=True):
    col_var0 = check_var(df,  var_limit=var_limit, sample_size=sample_size)
    df.drop(col_var0, axis=1, inplace=True)
    if only_var==False:
        col_corr1 = check_corr(df, corr_limit=corr_limit, sample_size=sample_size)
        df.drop(col_corr1, axis=1, inplace=True)
    return

def __get_use_files__():
    
    return

def get_use_files(prefixes=[], is_train=True):
    unused_files = []
   
    # unused_files  = [f.split('/')[-1] for f in sorted(glob('../feature_unused/*.pkl'))]
    # unused_files += [f.split('/')[-1] for f in sorted(glob('../feature_var0/*.pkl'))]
    # unused_files += [f.split('/')[-1] for f in sorted(glob('../feature_corr1/*.pkl'))]
   
    all_files = sorted(glob('../feature/*.pkl'))
    # if is_train:
    #     all_files = sorted(glob('../feature/train*.pkl'))
    #     unused_files = ['../feature/train_'+f for f in unused_files]
    # else:
    #     all_files = sorted(glob('../feature/test*.pkl'))
    #     unused_files = ['../feature/test_'+f for f in unused_files]
    
    if len(prefixes)>0:
        use_files = []
        for prefix in prefixes:
            use_files += glob(f'../feature/*{prefix}*')
        all_files = (set(all_files) & set(use_files)) - set(unused_files)
        
    else:
        for f in unused_files:
            if f in all_files:
                all_files.remove(f)
    
    all_files = sorted(all_files)
    
    print(f'got {len(all_files)}')
    return all_files

