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


file_name = 'recall_0531_addsim_addAA_RA_additemtime_addcount_addnn_addtxt_interactive_countdetail'
df = pd.read_csv(train_path + 'new_recall/' + file_name + '.csv')


# In[6]:


whole_click['user_mean_count'] = whole_click.groupby(['user_id','phrase'])['item_count'].transform('mean')

whole_click['user_max_count'] = whole_click.groupby(['user_id','phrase'])['item_count'].transform('max')

whole_click['user_min_count'] = whole_click.groupby(['user_id','phrase'])['item_count'].transform('min')


# In[7]:


whole_click['user_count'] = whole_click.groupby(['user_id','phrase'])['time'].transform('count')


# In[8]:


temp = whole_click[['user_id','phrase','user_count']].sort_values('user_count').reset_index(drop=True).drop_duplicates(['user_id'], keep='last')

temp['is_user_count_climax'] = 1
whole_click = pd.merge(left=whole_click,
         right = temp[['user_id','phrase','is_user_count_climax']],
         how = 'left',
         on = ['user_id','phrase'])


# In[9]:


temp = whole_click[['user_id','phrase','user_count']].sort_values('user_count').reset_index(drop=True).drop_duplicates(['user_id'], keep='first')

temp['is_user_count_lowerpoint'] = 1
whole_click = pd.merge(left=whole_click,
         right = temp[['user_id','phrase','is_user_count_lowerpoint']],
         how = 'left',
         on = ['user_id','phrase'])


# In[10]:


df = pd.merge(left = df,
        right = whole_click[['user_id', 'phrase'
            ,'user_mean_count','user_max_count','user_min_count','is_user_count_climax','is_user_count_lowerpoint'
                            ]].drop_duplicates(['user_id','phrase']),
        how = 'left',
        on = ['user_id','phrase'])


# In[11]:


df.loc[pd.isna(df['is_user_count_climax']), 'is_user_count_climax'] = 0


# In[12]:


df.loc[pd.isna(df['is_user_count_lowerpoint']), 'is_user_count_lowerpoint'] = 0


# In[13]:


for i in ['user_mean_count','user_max_count','user_min_count']:
    df[i] = df['item_count'].fillna(0) / df[i]


# In[14]:


df['is_user_count_climax'] = df['is_user_count_climax'] * df['is_climix'].fillna(0)


# In[15]:


df['is_user_count_lowerpoint'] = df['is_user_count_lowerpoint'] * df['is_user_count_lowerpoint'].fillna(0)


# In[16]:


df.to_csv(train_path + 'new_recall/' + file_name + '_userfeature.csv', index=False)

