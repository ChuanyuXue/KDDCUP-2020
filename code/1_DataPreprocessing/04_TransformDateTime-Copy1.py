#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
from tqdm import tqdm  
from collections import defaultdict  
import math  
import numpy as np
import datetime


# In[2]:


random_number_1 = 41152582
random_number_2 = 1570909091


# In[3]:


train_path = './user_data/offline/'
test_path = './user_data/offline/'

now_phase = 9
for c in range(now_phase + 1):  
    print('phase:', c)  
    click_train = pd.read_csv(train_path + '/offline_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
    click_test = pd.read_csv(test_path + '/offline_test_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
    click_query = pd.read_csv(test_path + '/offline_test_qtime-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time']) 
    
    click_train['unix_time'] = click_train['time'].apply(lambda x: x * random_number_2 + random_number_1)
    click_train['datetime'] = click_train['unix_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    click_train.to_csv(train_path+'/offline_train_click_{}_time.csv'.format(c),index=False)
    
    click_test['unix_time'] = click_test['time'].apply(lambda x: x * random_number_2 + random_number_1)
    click_test['datetime'] = click_test['unix_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    click_test.to_csv(test_path+'/offline_test_click_{}_time.csv'.format(c),index=False)
    
    click_query['unix_time'] = click_query['time'].apply(lambda x: x * random_number_2 + random_number_1)
    click_query['datetime'] = click_query['unix_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    click_query.to_csv(test_path+'/offline_test_qtime_{}_time.csv'.format(c),index=False)   
    


# In[4]:


num = 1
train_path = './user_data/model_'+str(num)
test_path = './user_data/model_'+str(num)

now_phase = 9
for c in range(now_phase + 1):  
    print('phase:', c)  
    click_train = pd.read_csv(train_path + '/model_'+str(num)+'_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
    click_test = pd.read_csv(test_path + '/model_'+str(num)+'_test_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
    click_query = pd.read_csv(test_path + '/model_'+str(num)+'_test_qtime-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time']) 
    
    click_train['unix_time'] = click_train['time'].apply(lambda x: x * random_number_2 + random_number_1)
    click_train['datetime'] = click_train['unix_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    click_train.to_csv(train_path+'/model_'+str(num)+'_train_click_{}_time.csv'.format(c),index=False)
    
    click_test['unix_time'] = click_test['time'].apply(lambda x: x * random_number_2 + random_number_1)
    click_test['datetime'] = click_test['unix_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    click_test.to_csv(test_path+'/model_'+str(num)+'_test_click_{}_time.csv'.format(c),index=False)
    
    click_query['unix_time'] = click_query['time'].apply(lambda x: x * random_number_2 + random_number_1)
    click_query['datetime'] = click_query['unix_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    click_query.to_csv(test_path+'/model_'+str(num)+'_test_qtime_{}_time.csv'.format(c),index=False)   
    


# In[5]:


train_path = './user_data/dataset'  
test_path = './user_data/dataset'

now_phase = 9
for c in range(now_phase + 1):  
    print('phase:', c)  
    click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
    click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
    click_query = pd.read_csv(test_path + '/underexpose_test_qtime-{}.csv'.format(c), header=None,  names=['user_id', 'time']) 
    
    click_train['unix_time'] = click_train['time'].apply(lambda x: x * random_number_2 + random_number_1)
    click_train['datetime'] = click_train['unix_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    click_train.to_csv(train_path+'/underexpose_train_click_{}_time.csv'.format(c),index=False)
    
    click_test['unix_time'] = click_test['time'].apply(lambda x: x * random_number_2 + random_number_1)
    click_test['datetime'] = click_test['unix_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    click_test.to_csv(test_path+'/underexpose_test_click_{}_time.csv'.format(c),index=False)
    
    click_query['unix_time'] = click_query['time'].apply(lambda x: x * random_number_2 + random_number_1)
    click_query['datetime'] = click_query['unix_time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    click_query.to_csv(test_path+'/underexpose_test_qtime_{}_time.csv'.format(c),index=False)   
    

