#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings 
warnings.filterwarnings("ignore") 


# In[30]:


current_stage = 9
path = './data/'
output_path = './user_data/offline/'
input_header = 'underexpose_'
output_header = 'offline_'

#path = 'offline/'
#input_header = 'offline_'
#output_header = 'model1/model_1_'


# In[31]:


df_train_list = [pd.read_csv(path+'underexpose_train/'+input_header+'train_click-%d.csv'%x,
                             header=None,
                             names=['user_id', 'item_id', 'time']) for x in range(current_stage + 1)]
for x, df_train in enumerate(df_train_list):
    df_train.to_csv('./user_data/dataset/' + input_header + 'train_click-%d.csv'%x, index=False,header=None)

df_train = pd.concat(df_train_list)
df_train = df_train.drop_duplicates(subset=['user_id','item_id','time'],keep='last')
df_train = df_train.reset_index(drop=True)


# In[32]:


df_test_list = [pd.read_csv(path+'underexpose_test/'+input_header+'test_click-%d.csv'%x,
                             header=None,
                             names=['user_id', 'item_id', 'time']) for x in range(current_stage + 1)]
for x, df_test in enumerate(df_test_list):
    df_test.to_csv('./user_data/dataset/' + input_header + 'test_click-%d.csv'%x, index=False,header=None)
df_test = pd.concat(df_test_list)
df_test = df_test.drop_duplicates(subset=['user_id','item_id','time'],keep='last')
df_test = df_test.reset_index(drop=True)


# In[33]:


df = pd.concat([df_train,df_test])
df = df.drop_duplicates(subset=['user_id','item_id','time'],keep='last')
df = df.reset_index(drop=True)


# In[34]:


# if you are generating the offline dataset please use the comment sentense

# df_pred_list = [pd.read_csv(path+input_header+'test_qtime-%d.csv'%x,
#                              header=None,
#                              names=['user_id','item_id','time']) for x in range(current_stage + 1)]

#online
df_pred_list = [pd.read_csv(path+'underexpose_test/'+input_header+'test_qtime-%d.csv'%x,
                             header=None,
                             names=['user_id','time']) for x in range(current_stage + 1)]

for x, df_pred in enumerate(df_pred_list):
    df_pred.to_csv('./user_data/dataset/' + input_header + 'test_qtime-%d.csv'%x, index=False,header=None)


# In[35]:


for i in range(current_stage + 1):
    if 'item_id' in df_pred_list[i].columns:
        df_pred_list[i] = df_pred_list[i][['user_id','time']]


# In[36]:


df_list = []

for i in range(current_stage + 1):
    df_0 = pd.concat([df_train_list[i], df_test_list[i],df_pred_list[i]])
    df_0 = df_0.sort_values(by=['time'])
    df_0 = df_0.reset_index(drop=True)
    df_list.append(df_0)


# In[37]:


for i in range(current_stage + 1):
    count_log = []
    for index, row in df_pred_list[i].iterrows():
        count_log.append(sum((df_list[i]['user_id']==row['user_id']) & (df_list[i]['time']<row['time']) ))
    df_pred_list[i]['count_log'] = count_log


# In[38]:


list_train_list = [[] for x in range(current_stage + 1)]
list_test_list = [[] for x in range(current_stage + 1)]

for each_stage_out in range(current_stage + 1):
    
    fout = open(output_path + output_header + 'test_qtime-%d.csv'%each_stage_out,'w')
    
    for i, row in df_pred_list[each_stage_out].iterrows():
        if row['count_log'] < 3:
            continue    

        df_tmp = df_list[each_stage_out][df_list[each_stage_out]['user_id']==row['user_id']]
        
        if sum(df_tmp['time']==max(df_tmp['time'])) > 1:
            row_tmp = df_list[each_stage_out].loc[df_tmp[ (df_tmp['time']==max(df_tmp['time']) ) & (~np.isnan(df_tmp['item_id'] )) ].index[0]]
            user_id_tmp = row_tmp['user_id']
            item_id_tmp = row_tmp['item_id']
            time_tmp = row_tmp['time']
            fout.write(str(int(user_id_tmp)) + ',' + str(int(item_id_tmp)) + ',' + str(time_tmp) + '\n')
        else:
            row_tmp = df_list[each_stage_out].loc[df_tmp.index[-2]]
            user_id_tmp = row_tmp['user_id']
            item_id_tmp = row_tmp['item_id']
            time_tmp = row_tmp['time']            
            fout.write(str(int(user_id_tmp)) + ',' + str(int(item_id_tmp)) + ',' + str(time_tmp) + '\n')
        
        for each_stage_in in range(current_stage + 1):
            list_train_list[each_stage_in] += list(df_train_list[each_stage_in][(df_train_list[each_stage_in]['user_id']==row['user_id'])
                                       &(df_train_list[each_stage_in]['item_id']==item_id_tmp)].index)

            list_test_list[each_stage_in] += list(df_test_list[each_stage_in][(df_test_list[each_stage_in]['user_id']==row['user_id'])
                                     &(df_test_list[each_stage_in]['item_id']==item_id_tmp)].index)
    fout.close()


# In[ ]:





# In[39]:


df_train_list = [x.drop(labels=list_train_list[i],axis=0) for i,x in enumerate(df_train_list)]


# In[40]:


df_test_list = [x.drop(labels=list_test_list[i],axis=0) for i,x in enumerate(df_test_list)]


# In[41]:


df_train_list = [x.reset_index(drop=True) for x in df_train_list]
df_test_list = [x.reset_index(drop=True) for x in df_test_list]


# In[42]:


for i in range(current_stage + 1):
    df_train_list[i].to_csv(output_path + output_header+'train_click-%d.csv'%i,index=False,header=None)
    df_test_list[i].to_csv(output_path + output_header+'test_click-%d.csv'%i,index=False,header=None)

# In[ ]:




