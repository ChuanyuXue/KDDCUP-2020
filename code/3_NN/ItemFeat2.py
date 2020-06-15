# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:01:01 2020
@author: hcb
"""

import pandas as pd
import os
from config import config


def get_feat(now_phase=3, base_path=None):
    
    # if base_path is None:
    #     train_path = 'underexpose_train'
    #     test_path = 'underexpose_test'
    # else:
    #     train_path = os.path.join(base_path, 'underexpose_train')
    #     test_path = os.path.join(base_path, 'underexpose_test')
    train_path = config.train_path
    test_path = config.test_path  
    click_train = pd.DataFrame()
    click_test = pd.DataFrame()
    for c in range(now_phase + 1):
        click_tmp = pd.read_csv(train_path + f'/underexpose_train_click-{c}.csv', header=None,
                                names=['user_id', 'item_id', 'time'])
        click_tmp['user_id'] = '1_{}_'.format(c) + click_tmp['user_id'].astype(str)
        click_test_tmp = pd.read_csv(test_path + f'/underexpose_test_click-{c}.csv', header=None,
                                     names=['user_id', 'item_id', 'time'])
        click_test_tmp['user_id'] = '0_{}_'.format(c) + click_test_tmp['user_id'].astype(str)
        click_train = click_train.append(click_tmp)
        click_test = click_test.append(click_test_tmp)
    all_click = click_train.append(click_test)
    print(all_click['item_id'].nunique())
    item_df = all_click.groupby('item_id')['time'].count().reset_index()
    item_df.columns = ['item_id', 'degree']
    
    feat = pd.read_csv('./data/underexpose_train/underexpose_item_feat.csv', header=None)
    feat[1] = feat[1].apply(lambda x:x[1:]).astype(float)
    feat[128] = feat[128].apply(lambda x:x[:-1]).astype(float)
    feat[129] = feat[129].apply(lambda x:x[1:]).astype(float)
    feat[256] = feat[256].apply(lambda x:x[:-1]).astype(float)
    feat.columns = ['item_id'] + ['feat'+str(i) for i in range(256)]
    
    item_df = item_df.merge(feat, on='item_id', how='left')
    print(item_df['item_id'].nunique())
    def transform(x):
        if x > 150 and x <400:
            x = (x-150) // 25 * 25 +150
        elif x>=400:
            x = 400
        return x 
    
    item_df['degree'] = item_df['degree'].apply(lambda x: transform(x))
    degree_df = item_df.groupby('degree')[['feat'+str(i) for i in range(256)]].mean().reset_index()
    na_df = item_df[item_df['feat0'].isna()][['item_id', 'degree']].merge(degree_df, on='degree', how='left')
    item_df.dropna(inplace=True)
    item_df = pd.concat((item_df, na_df))
    
    item_df.to_csv('item_feat.csv', index=None)
    
if __name__ == '__main__':
    get_feat(now_phase=9)