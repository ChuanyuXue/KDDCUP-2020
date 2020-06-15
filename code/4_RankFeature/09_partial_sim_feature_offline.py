#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train_path = './user_data/offline/'
test_path = './user_data/offline/'
header = 'offline'
now_phase = 9


# In[3]:


sim_partial = pd.read_csv(train_path + 'new_recall/recall_partial.csv')


# In[4]:


sim_partial = sim_partial[['user_id','item_id','sim','feature_0','feature_1','feature_2','feature_3']]


# In[5]:


sim_partial.to_csv(train_path + 'new_recall/recall_partial.csv', index=False)


# In[6]:


file_name = 'recall_0531_addsim_addAA_RA_additemtime_addcount_addnn_addtxt_interactive_countdetail_userfeature'
df = pd.read_csv(train_path + 'new_recall/' + file_name + '.csv')


# In[7]:


sim_partial.columns = ['user_id','item_id','sim_partial','feature_0_partial','feature_1_partial','feature_2_partial','feature_3_partial']


# In[8]:


df = pd.merge(left = df,
             right = sim_partial,
             how = 'left',
             on = ['user_id','item_id'])


# In[9]:


df.to_csv(train_path + 'new_recall/' + file_name + '_partialsim.csv', index=False)

