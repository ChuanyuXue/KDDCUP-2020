#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train_path = './user_data/model_1/'
test_path = './user_data/model_1/'
header = 'model_1'
now_phase = 9


# In[6]:


sim_partial = pd.read_csv(train_path + 'new_recall/recall_partial.csv')


# In[9]:


sim_partial = sim_partial[['user_id','item_id','sim','feature_0','feature_1','feature_2','feature_3']]


# In[10]:


sim_partial.to_csv(train_path + 'new_recall/recall_partial.csv', index=False)


# In[12]:


file_name = 'recall_0531_addsim_addAA_RA_additemtime_addcount_addnn_addtxt_interactive_countdetail_userfeature'
df = pd.read_csv(train_path + 'new_recall/' + file_name + '.csv')


# In[14]:


sim_partial.columns = ['user_id','item_id','sim_partial','feature_0_partial','feature_1_partial','feature_2_partial','feature_3_partial']


# In[15]:


df = pd.merge(left = df,
             right = sim_partial,
             how = 'left',
             on = ['user_id','item_id'])


# In[19]:


df.to_csv(train_path + 'new_recall/' + file_name + '_partialsim.csv', index=False)

