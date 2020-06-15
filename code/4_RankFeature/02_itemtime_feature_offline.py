#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def extractItemCount(df, df_qTime, df_click, intervals, col_name):
    
    
    
    df_click = getTimeInterval(df_click,intervals)
    
    if 'time_interval' not in df.columns:
        df_qTime = getTimeInterval(df_qTime,intervals)
        df = df.merge(df_qTime[['user_id','time_interval']])
    
    df_click_sta = df_click[['user_id','item_id','time_interval']].groupby(by=['item_id','time_interval'],as_index=False).count()
    df_click_sta.columns = ['item_id','time_interval',col_name]
    
    df = df.merge(df_click_sta,on=['item_id','time_interval'],how='left')
    
    return df
    


# In[3]:


def getTimeInterval(df,intervals):
    df['hour_minute'] = (df['datetime'].dt.hour + df['datetime'].dt.minute/60)/24

    time_interval_list = np.linspace(0,1,intervals)

    df['time_interval'] = df['hour_minute'].apply(lambda x: np.where(x<time_interval_list)[0][0]-1 )
    df['time_interval'] = (df['datetime'].dt.day - min(df['datetime'].dt.day))*intervals + df['time_interval']
    return df


# In[4]:


train_path = './user_data/offline/'
test_path = './user_data/offline/'
header = 'offline'

now_phase = 9
file_name = 'recall_0531_addsim_addAA_RA'

df = pd.read_csv('./user_data/offline/new_recall/' + file_name + '.csv')


# In[5]:


whole_qTime = pd.DataFrame() 

for c in range(now_phase + 1):  
    #print('phase:', c)  
    click_query = pd.read_csv(test_path + header + '_test_qtime_{}_time.csv'.format(c))  
    whole_qTime = whole_qTime.append(click_query)  
    
whole_qTime = whole_qTime.reset_index(drop=True)
whole_qTime['datetime'] = pd.to_datetime(whole_qTime['datetime'])


# In[6]:


whole_click = pd.DataFrame() 
for c in range(now_phase + 1):  
    #print('phase:', c)  
    click_train = pd.read_csv(train_path + header + '_train_click_{}_time.csv'.format(c))  
    click_test = pd.read_csv(test_path + header + '_test_click_{}_time.csv'.format(c))  

    all_click = click_train.append(click_test)  
    whole_click = whole_click.append(all_click)  
    
    
whole_click =  whole_click.drop_duplicates(subset=['user_id','item_id','time'],keep='last')
whole_click = whole_click.sort_values('time')
whole_click = whole_click.reset_index(drop=True)
whole_click['datetime'] = pd.to_datetime(whole_click['datetime'])


# In[7]:


df = extractItemCount(df,whole_qTime,whole_click,2,'item_count_12h')
df = extractItemCount(df,whole_qTime,whole_click,4,'item_count_6h')
df = extractItemCount(df,whole_qTime,whole_click,6,'item_count_4h')
df = extractItemCount(df,whole_qTime,whole_click,12,'item_count_2h')
df = extractItemCount(df,whole_qTime,whole_click,24,'item_count_1h')


# In[8]:


df.to_csv('./user_data/offline/new_recall/' + file_name + '_additemtime.csv',index=False)

