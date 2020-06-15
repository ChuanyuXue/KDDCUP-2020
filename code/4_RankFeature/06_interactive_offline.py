#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from gensim.models import KeyedVectors
import gc


# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


train_path = './user_data/offline/'
test_path = './user_data/offline/'
header = 'offline'
now_phase = 9


# In[3]:


file_name = 'recall_0531_addsim_addAA_RA_additemtime_addcount_addnn_addtxt'
df = pd.read_csv(train_path + 'new_recall/' + file_name + '.csv')


# In[ ]:





# In[4]:


click_trn = []
click_tst = []
#qtime_tst = []
for p in tqdm(range(0, now_phase+1)):
    tmp = pd.read_csv(train_path + header + f'_train_click-{p}.csv', header=None, names=['user_id', 'item_id', 'time'])
    tmp['phrase'] = p
    click_trn.append(tmp)
    tmp = pd.read_csv(test_path + header + f'_test_click-{p}.csv', header=None, names=['user_id', 'item_id', 'time'])
    tmp['phrase'] = p
    click_tst.append(tmp)
    #tmp = pd.read_csv(test_path + header + f'_test_qtime-{p}.csv', header=None, names=['user_id', 'item_id', 'query_time'])
    #tmp['phrase'] = p
    #qtime_tst.append(tmp)
    
click_trn = pd.concat(click_trn, axis=0, ignore_index=True)
click_tst = pd.concat(click_tst, axis=0, ignore_index=True)
#qtime_tst = pd.concat(qtime_tst, axis=0, ignore_index=True)


# In[5]:


click_df = pd.concat([click_trn, click_tst], axis=0, ignore_index=True)
click_df['item_count'] = click_df.groupby(['item_id', 'phrase'])['user_id'].transform('count')


# In[6]:


click_df.shape


# In[7]:


click_df.head()


# In[8]:


count_map = click_df[['item_id', 'phrase', 'item_count']].drop_duplicates()


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


def gen_add_1(df):
    group_df = df.groupby(['user_id', 'phrase', 'item_count'])['time'].agg([['user_item_count_cnt', 'count'],
                                                                            ['user_item_count_max_time', 'max'],
                                                                            ['user_item_count_min_time', 'min']]).reset_index()
    group_df['sum'] = group_df.groupby(['user_id', 'phrase'])['user_item_count_cnt'].transform('sum')
    group_df['user_item_count_ratio'] = group_df['user_item_count_cnt'] / group_df['sum']
    group_df['user_item_count_timedelta'] = group_df['user_item_count_max_time'] - group_df['user_item_count_min_time']
    del group_df['sum']
    return group_df


# In[10]:


df = df.merge(count_map, on=['item_id', 'phrase'], how='left')


# In[11]:


df.shape


# In[12]:


train_add1 = gen_add_1(click_df)


# In[13]:


train_add1.shape


# In[ ]:





# In[14]:


df = df.merge(train_add1, on=['user_id', 'phrase', 'item_count'], how='left')


# In[15]:


df.shape


# In[25]:


df[df['phrase'] == 7]['img_feature_1']


# In[16]:


df.to_csv(train_path + 'new_recall/' + file_name + '_interactive.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




