#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm


# In[2]:


train_path = './user_data/offline/'
test_path = './user_data/offline/'
header = 'offline'
now_phase = 9


# In[3]:


item_count_dict = {}
user_phase_dict = {}

whole_click = pd.DataFrame() 
for c in range(now_phase + 1):  
    #print('phase:', c)  
    click_train = pd.read_csv(train_path + header + '_train_click_{}_time.csv'.format(c))  
    click_test = pd.read_csv(test_path + header + '_test_click_{}_time.csv'.format(c))  
    click_query = pd.read_csv(test_path + header + '_test_qtime_{}_time.csv'.format(c)) 

    all_click = click_train.append(click_test)  
    whole_click = whole_click.append(all_click)  
    df_item_count = whole_click.groupby(['item_id'],as_index=False)['user_id'].agg({'count':'count'}) 
    
    for i, row in df_item_count.iterrows():
        item_count_dict.setdefault(row['item_id'],list(np.zeros(now_phase+1)))
        item_count_dict[row['item_id']][c] = row['count']   
        
    
    for i, row in click_query.iterrows():
        user_phase_dict[row['user_id']] = c


# In[4]:


item_count_df = pd.DataFrame({'item_id': list(item_count_dict.keys()), 'count': list(item_count_dict.values())})


# In[5]:


item_count_list = []

for i,row in item_count_df.iterrows():
    for j in range(now_phase + 1):
        item_count_list.append([row['item_id'],j,row['count'][j]])

item_count_df = pd.DataFrame(item_count_list, columns=['item_id', 'phrase', 'count'])


# In[6]:


# 与上下阶段的差

item_count_df_list = []
for i,x in tqdm(item_count_df.groupby('item_id')):
    x['diff_from_last'] = x['count'].diff(1)
    x['diff_from_next'] = x['count'].diff(-1)
    item_count_df_list.append(x)

item_count_df = pd.concat(item_count_df_list)


# 最大值与当前phase的差

item_count_df['max'] = item_count_df.groupby('item_id')['count'].transform('max')
item_count_df['diff_from_max'] = item_count_df['max'] - item_count_df['count']
item_count_df = item_count_df.drop(columns = 'max')


# In[7]:


file_name = 'recall_0531_addsim_addAA_RA_additemtime'

df = pd.read_csv('./user_data/offline/new_recall/' + file_name + '.csv')


# In[8]:


df['phrase'] = df['user_id'].apply(lambda x:user_phase_dict[x])


# In[9]:


df = pd.merge(left = df,
             right = item_count_df,
             how = 'left',
             on = ['item_id','phrase'])


# In[10]:


df.shape


# In[ ]:





# In[11]:


df.to_csv('./user_data/offline/new_recall/' + file_name + '_addcount.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




