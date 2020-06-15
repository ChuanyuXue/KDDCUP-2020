#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict  
import math  
from sys import stdout
import pickle
from evaulation import evaluate


# In[ ]:





# In[2]:


def get_predict(df, pred_col, top_fill, ranknum):  
    top_fill = [int(t) for t in top_fill.split(',')]  
    scores = [-1 * i for i in range(1, len(top_fill) + 1)]  
    ids = list(df['user_id'].unique())  
    fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])  
    fill_df.sort_values('user_id', inplace=True)  
    fill_df['item_id'] = top_fill * len(ids)  
    fill_df[pred_col] = scores * len(ids)  
    df = df.append(fill_df)  
    df.sort_values(pred_col, ascending=False, inplace=True)  
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')  
    df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)  
    df = df[df['rank'] <= ranknum]  
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',', expand=True).reset_index()  
    return df 


# In[3]:


def merge_label(train, label):
    tmp = pd.merge(left = train,
            right = label[['user_id','item_id','future_click']],
            how = 'left',
            on = ['user_id','item_id'])
    tmp.loc[~pd.isna(tmp['future_click']), 'future_click'] = 1
    tmp.loc[pd.isna(tmp['future_click']), 'future_click'] = 0
    return tmp


# In[ ]:





# In[4]:


model1_train = pd.read_csv('./user_data/model_1/new_recall/recall_0531_addsim_addAA_RA_additemtime_addcount_addnn_addtxt_interactive_countdetail_userfeature_partialsim_emergency.csv')
model1_label = pd.read_csv('./user_data/model_1/model_1_debias_track_answer.csv', 
                           names = ['phase','user_id','item_id','future_click'])
model1_train = merge_label(model1_train, model1_label)


# In[5]:


model1_train.shape


# In[ ]:





# In[ ]:





# In[6]:


offline_train = pd.read_csv('./user_data/offline/new_recall/recall_0531_addsim_addAA_RA_additemtime_addcount_addnn_addtxt_interactive_countdetail_userfeature_partialsim_emergency.csv')
offline_label = pd.read_csv('./user_data/offline/offline_debias_track_answer.csv', 
                            names = ['phase','user_id','item_id','future_click'])
offline_train = merge_label(offline_train, offline_label)


# In[7]:


offline_train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


col_sel = [x for x in offline_train.columns if x not in ['user_item_count_max_time','user_item_count_min_time',
                                                        'time_interval','item_count_4h','phrase','item_count_6h',
                                                        'is_user_count_climax','item_count_2h','is_user_count_lowerpoint',
                                                        'item_count_1h']]


# In[9]:


len(col_sel)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 线下

# In[10]:


now_phase = 9
train_path = './user_data/dataset/'  
test_path = './user_data/dataset/'
header = 'underexpose'


item_sim_list = []
item_cnt_list = []
user_item = []

whole_click = pd.DataFrame()  
for c in range(7,now_phase + 1):  
    print('phase:', c)  
    click_train = pd.read_csv(train_path + header + '_train_click_{}_time.csv'.format(c))  
    click_test = pd.read_csv(test_path + header + '_test_click_{}_time.csv'.format(c))  
    click_query = pd.read_csv(test_path + header + '_test_qtime_{}_time.csv'.format(c)) 



    all_click = click_train.append(click_test)  
    whole_click = whole_click.append(all_click)  


    whole_click =  whole_click.drop_duplicates(subset=['user_id','item_id','time'],keep='last')
    whole_click = whole_click.sort_values('time')
    whole_click = whole_click.reset_index(drop=True)

# find most popular items  
top50_click = whole_click['item_id'].value_counts().index[:500].values  
top50_click = ','.join([str(i) for i in top50_click])  


# In[ ]:





# In[11]:


model_train = pd.concat([model1_train,offline_train])
model_train = model_train.reset_index(drop=True)

model_train_p = model_train[model_train['future_click']==1]
model_train_p = model_train_p.reset_index(drop=True)

model_train_n = model_train[model_train['future_click']==0]
model_train_n = model_train_n.reset_index(drop=True)


# In[12]:


model_train_p.shape


# In[13]:


online_train = pd.read_csv('./user_data/dataset/new_recall/recall_0531_addsim_addAA_RA_additemtime_addcount_addnn_addtxt_interactive_countdetail_userfeature_partialsim_emergency.csv')


# In[14]:


online_train.shape


# In[15]:


online_train = online_train[online_train['phrase']>6]
online_train = online_train.reset_index(drop=True)


# In[ ]:





# In[ ]:





# In[16]:


import random

def generateDataset(df_n,df_p,random_seed):
    random.seed(random_seed)
    n_index = random.sample(list(range(len(df_n))), len(df_p)*5)
    df_ns = df_n.loc[n_index]
    df = pd.concat([df_ns,df_p])
    df = df.reset_index(drop=True)
    return df

model_train_s_1 = generateDataset(model_train_n,model_train_p,2020)
model_train_s_2 = generateDataset(model_train_n,model_train_p,0)
model_train_s_3 = generateDataset(model_train_n,model_train_p,2019)
model_train_s_4 = generateDataset(model_train_n,model_train_p,1000)
model_train_s_5 = generateDataset(model_train_n,model_train_p,3000)
model_train_s_6 = generateDataset(model_train_n,model_train_p,2021)


# In[ ]:





# In[17]:


def addWeightForDataSet(df,item_degree_median,weight):
    df['sample_weight'] = df['count']/item_degree_median
    df['sample_weight'] = df['sample_weight'].apply(lambda x: 5 if x<1 else 1)
    df.loc[(df['count']<item_degree_median)&(df['future_click']==1),'sample_weight'] = weight
    df.loc[(df['count']<item_degree_median)&(df['future_click']==1)&(df['phrase'].isin([7,8,9])), 'sample_weight'] = weight * 2
    return df


# In[18]:


# def addWeightForDataSet(df,item_degree_median,weight):
#     df['sample_weight'] = df['count']/item_degree_median
#     df['sample_weight'] = df['sample_weight'].apply(lambda x: 5 if x<1 else 1)
#     df.loc[(df['count']<item_degree_median)&(df['future_click']==1),'sample_weight'] = weight
#     return df


# In[19]:


model_train_s_1 = addWeightForDataSet(model_train_s_1,30,35)
model_train_s_2 = addWeightForDataSet(model_train_s_2,30,35)
model_train_s_3 = addWeightForDataSet(model_train_s_3,30,35)
model_train_s_4 = addWeightForDataSet(model_train_s_4,30,35)
model_train_s_5 = addWeightForDataSet(model_train_s_5,30,35)
model_train_s_6 = addWeightForDataSet(model_train_s_6,30,35)


# In[20]:


feature_list = [x for x in col_sel if x not in ['user_id','item_id','future_click','sample_weight'] 
                and 'result' not in x]


# In[21]:


len(feature_list)


# In[22]:


feature_list


# In[ ]:





# In[ ]:





# In[23]:


feature_list_noleak = [x for x in col_sel if x not in ['user_id','item_id','future_click','sample_weight','result',
                                                       'diff_from_next'] and 'result' not in x]


# In[24]:


len(feature_list_noleak)


# In[ ]:





# In[25]:


def cbt_model(m,df_train,df_test,feat):
    m.fit(df_train[feat],df_train[['future_click']],sample_weight=list(df_train['sample_weight']))
    print(sorted(dict(zip(m.feature_names_,m.feature_importances_)).items(), key=lambda x:x[1], reverse=True))
    result = m.predict_proba(df_test[feat])[:,1]
    return result


# In[ ]:





# In[26]:


df_res = pd.DataFrame()


# In[27]:


import catboost as cat
clf_cbt = cat.CatBoostClassifier(iterations=2500,learning_rate=0.01,depth=6,
                                   verbose=True,thread_count=12,colsample_bylevel=0.8
                                   ,l2_leaf_reg=1
                                   ,random_seed=1024)

df_res['result_1'] = cbt_model(clf_cbt,model_train_s_1,online_train,feature_list)

df_res['result_2'] = cbt_model(clf_cbt,model_train_s_2,online_train,feature_list)

df_res['result_3'] = cbt_model(clf_cbt,model_train_s_3,online_train,feature_list)

df_res['result_4'] = cbt_model(clf_cbt,model_train_s_4,online_train,feature_list)

df_res['result_5'] = cbt_model(clf_cbt,model_train_s_5,online_train,feature_list)

df_res['result_6'] = cbt_model(clf_cbt,model_train_s_6,online_train,feature_list)


# In[ ]:





# In[ ]:





# In[28]:


df_res['phrase'] = online_train['phrase']
df_res['user_id'] = online_train['user_id']
df_res['item_id'] = online_train['item_id']


# In[ ]:





# In[29]:


df_res['result'] = df_res['result_1'] + df_res['result_2'] + df_res['result_3'] + df_res['result_4'] + df_res['result_5'] + df_res['result_6'] 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


def lgb_model(df_train,df_test,feat,params,num_round):
    train_data = lgb.Dataset(df_train[feat], 
                         label=df_train[['future_click']],weight=df_train['sample_weight'])  
    print('lgb training')
    bst = lgb.train(params,
                train_data,
                num_round)    
    print('lgb predicting')
    result = bst.predict(df_test[feat])    
    return result


# In[ ]:





# In[ ]:





# In[31]:


import lightgbm as lgb
import time

num_round = 2500
params = {
        'learning_rate': 0.01,
        'boosting_type': 'dart',
        'objective': 'binary',
        #'metric': 'auc',
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 10,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': 8,
        'verbose': 1,
    }


# In[32]:


df_res['lgb_dart_1'] = lgb_model(model_train_s_1,online_train,feature_list,params,num_round)

df_res['lgb_dart_2'] = lgb_model(model_train_s_2,online_train,feature_list,params,num_round)

df_res['lgb_dart_3'] = lgb_model(model_train_s_3,online_train,feature_list,params,num_round)

df_res['lgb_dart_4'] = lgb_model(model_train_s_4,online_train,feature_list,params,num_round)

df_res['lgb_dart_5'] = lgb_model(model_train_s_5,online_train,feature_list,params,num_round)

df_res['lgb_dart_6'] = lgb_model(model_train_s_6,online_train,feature_list,params,num_round)


# In[ ]:





# In[ ]:





# In[33]:


df_res['result_lgb_dart'] = df_res['lgb_dart_1'] + df_res['lgb_dart_2'] + df_res['lgb_dart_3'] + df_res['lgb_dart_4'] + df_res['lgb_dart_5'] + df_res['lgb_dart_6'] 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[34]:


df_res['result'] = df_res['result']/6
df_res['result_lgb_dart'] = df_res['result_lgb_dart']/6


# In[ ]:





# In[35]:


df_res['count_na'] = online_train['count'].apply(lambda x: np.nan if x ==0 else x)
df_res['m'] = df_res['count_na'].apply(lambda x:max(0.61,1/math.log1p(x+1)))


# In[ ]:





# In[36]:


df_res['result_PostProcess'] = df_res['result'] * df_res['m']
df_res['result_lgb_dart_PostProcess'] = df_res['result_lgb_dart'] * df_res['m']


# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


df_res['ensemble1'] = 10 / ( 6/df_res['result_PostProcess'] + 4/df_res['result_lgb_dart_PostProcess']) 
df_res['ensemble2'] = np.power( df_res['result_PostProcess']**6 * df_res['result_lgb_dart_PostProcess']**4 , 1/10) 


# In[ ]:





# In[38]:


df_res['ensemble'] = df_res['ensemble1']  + df_res['ensemble2'] 


# In[ ]:





# In[39]:


#df_res['PostProcess'] = df_res['ensemble'] * df_res['m']

recom_df = df_res[['user_id','item_id','ensemble']]
result = get_predict(recom_df, 'ensemble', top50_click, 50) 
result['user_id'] = result['user_id'].astype(int)
result.to_csv('./prediction_result/prediction_result.csv', index=False, header=None)

