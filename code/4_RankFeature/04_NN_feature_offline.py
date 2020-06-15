#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict  
import math  
import json
from sys import stdout
import pickle


# In[2]:


train_path = './user_data/offline/'
test_path = './user_data/offline/'
header = 'offline'
now_phase = 9


# In[3]:


nn = pd.read_csv(train_path + 'nn/nn_' + header + '.csv')


# In[4]:


result = pd.DataFrame()
result['item'] = nn['item']
result['item'] = result['item'].apply(lambda x: x[1:])
result['item'] = result['item'].apply(lambda x: x[:-1])
result['item'] = result['item'].apply(lambda x: x.split(','))
result['item'] = result['item'].apply(lambda x: [int(y) for y in x]) 
result['user_id'] = nn['user']



result['score'] = nn['score']
result['score'] = result['score'].apply(lambda x: x[1:])
result['score'] = result['score'].apply(lambda x: x[:-1])
result['score'] = result['score'].apply(lambda x: x.split(','))
result['score'] = result['score'].apply(lambda x: [float(y) for y in x]) 

result['score'] = result['score'].apply(lambda x: [1/(1+np.exp(-y)) for y in x])

recom_item = []

for i,row in tqdm(result.iterrows()):
    tmp_list = row['item']
    score_list = row['score']
    for j in range(len(score_list)):
        recom_item.append([ row['user_id'],tmp_list[j],score_list[j]])

recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'nn']) 


# In[5]:


file_name = 'recall_0531_addsim_addAA_RA_additemtime_addcount'
recall = pd.read_csv(train_path + 'new_recall/' + file_name + '.csv')


# In[6]:


recall = pd.merge(left=recall,
                 right=recom_df,
                 how='left',
                 on=['user_id','item_id'])


# In[7]:


recall.to_csv(train_path + 'new_recall/' + file_name + '_addnn.csv',index=False)

