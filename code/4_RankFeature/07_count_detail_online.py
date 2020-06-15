#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train_path = './user_data/dataset/'
test_path = './user_data/dataset/'
header = 'underexpose'
now_phase = 9


# In[3]:


whole_click = pd.DataFrame() 
for c in range(now_phase + 1):  
    #print('phase:', c)  
    click_train = pd.read_csv(train_path + header + '_train_click_{}_time.csv'.format(c))  
    click_test = pd.read_csv(test_path + header + '_test_click_{}_time.csv'.format(c))  
    click_query = pd.read_csv(test_path + header + '_test_qtime_{}_time.csv'.format(c)) 

    all_click = click_train.append(click_test)  
    all_click['phrase'] = c
    whole_click = whole_click.append(all_click)  


# In[4]:


whole_click['item_count'] = whole_click.groupby(['item_id','phrase'])['time'].transform('count')


# In[5]:


whole_click['item_max_time_in_phrase'] = whole_click.groupby(['item_id','phrase'])['time'].transform('max')


# 比如当前阶段count是否是峰值，当前阶段的count和最小值的差，当前阶段count在当前阶段所有商品的rank，当前阶段count在当前商品即使阶段的tank

# In[6]:


## 当前阶段是否是峰值
climax = whole_click.sort_values('item_count').reset_index(drop=True).drop_duplicates(['item_id'], keep='last')
climax['is_climix'] = 1
whole_click = pd.merge(left = whole_click,
         right = climax[['item_id', 'phrase','is_climix']],
         how = 'left',
         on = ['item_id','phrase'])
whole_click.loc[pd.isna(whole_click['is_climix']), 'is_climix'] = 0


# In[7]:


# 当前阶段是否是波谷
valley = whole_click.sort_values('item_count').reset_index(drop=True).drop_duplicates(['item_id'], keep='first')
valley['is_lowest_point'] = 1
whole_click = pd.merge(left = whole_click,
         right = valley[['item_id', 'phrase','is_lowest_point']],
         how = 'left',
         on = ['item_id','phrase'])
whole_click.loc[pd.isna(whole_click['is_lowest_point']), 'is_lowest_point'] = 0


# In[8]:


# 当前阶段count与最小值和均值的差
whole_click['item_count_min'] = whole_click.groupby('item_id')['item_count'].transform('min')
whole_click['item_count_mean'] = whole_click.groupby('item_id')['item_count'].transform('mean')
whole_click['item_diff_from_min'] = whole_click['item_count'] - whole_click['item_count_min']
whole_click['item_diff_from_mean'] = whole_click['item_count'] - whole_click['item_count_mean']


# In[9]:


# 当前阶段count在当前阶段所有商品的rank
whole_click['item_count_rankin_phrase'] = whole_click.groupby('phrase')['item_count'].rank(method='dense')


# In[10]:


# 当前阶段count在商品历史阶段
whole_click['item_count_rankin_history'] = whole_click.groupby('item_id')['item_count'].rank(method='dense')


# In[11]:


file_name = 'recall_0531_addsim_addAA_RA_additemtime_addcount_addnn_addtxt_interactive'
df = pd.read_csv(train_path + 'new_recall/' + file_name + '.csv')


# In[12]:


df = pd.merge(left = df,
        right = whole_click[['item_id', 'phrase'
            ,'item_max_time_in_phrase',
                            'is_climix',
                            'is_lowest_point',
                            'item_diff_from_min',
                            'item_diff_from_mean',
                            'item_count_rankin_phrase',
                            'item_count_rankin_history'
                            ]].drop_duplicates(['item_id','phrase']),
        how = 'left',
        on = ['item_id','phrase'])


# In[13]:


df.loc[pd.isna(df['item_count']), 'item_never_in_phrase'] = 1
df.loc[~pd.isna(df['item_count']), 'item_never_in_phrase'] = 0


# In[14]:


df.to_csv(train_path + 'new_recall/' + file_name + '_countdetail.csv', index=False)

