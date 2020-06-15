#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from gensim.models import KeyedVectors
import gc


# In[3]:


train_path = './user_data/model_1/'
test_path = './user_data/model_1/'
header = 'model_1'
now_phase = 9


# In[4]:


file_name = 'recall_0531_addsim_addAA_RA_additemtime_addcount_addnn'
df = pd.read_csv(train_path + 'new_recall/' + file_name + '.csv')


# In[5]:


user_item_list = []
for phase in range(now_phase + 1):
    file = open(train_path + 'new_similarity/' + 'user2item_new%d.pkl'%phase, 'rb')
    user_item_list.append(pickle.load(file))


# In[6]:


txt_model = KeyedVectors.load_word2vec_format('./user_data/w2v_txt_vec.txt')
img_model = KeyedVectors.load_word2vec_format('./user_data/w2v_img_vec.txt')


# In[7]:


txt_similarity = {}
img_similarity = {}
txt_feature = []
img_feature = []


# In[8]:


for phase in range(0, now_phase + 1):
    current_recall = df[df['phrase'] == phase]
    current_data = user_item_list[phase]
    for eachrow in tqdm(current_recall[['user_id','item_id']].values):
        history_click = current_data[eachrow[0]][-15:]
        item = eachrow[1]
        txt_sim_list = []
        img_sim_list = []
        for related_item in history_click:
            index = '_'.join(sorted([str(item), str(related_item)]))
            
            # calculate txt similarity
            if index in txt_similarity:
                txt_sim = txt_similarity[index]
            else:
                try:
                    txt_sim = int(txt_model.similarity(str(item), str(related_item)) * 1e4) / 1e4
                except:
                    txt_sim = np.nan
            txt_similarity[index] = txt_sim
            txt_sim_list.append(txt_sim)
                
            # calculate img similarity
            
            if index in img_similarity:
                img_sim = img_similarity[index]
            else:
                try:
                    img_sim = int(img_model.similarity(str(item), str(related_item)) * 1e4) / 1e4
                except:
                    img_sim = np.nan
            img_similarity[index] = img_sim
            img_sim_list.append(img_sim)
            
        txt_feature.append([eachrow[0], item,
                            np.nanmax(txt_sim_list),
                            np.nanmean(txt_sim_list),
                            np.nanstd(txt_sim_list),
                            np.nansum(txt_sim_list),
                            np.sum(np.isnan(txt_sim_list))])
        img_feature.append([eachrow[0], item,
                            np.nanmax(img_sim_list),
                            np.nanmean(img_sim_list),
                            np.nanstd(img_sim_list),
                            np.nansum(img_sim_list),
                            np.sum(np.isnan(img_sim_list))])
    gc.collect()


# In[9]:


txt_feature = pd.DataFrame(txt_feature, columns=['user_id','item_id'] + ['txt_feature_' + str(x) for x in range(5)])
img_feature = pd.DataFrame(img_feature, columns=['user_id','item_id'] + ['img_feature_' + str(x) for x in range(5)])


# In[10]:


df = pd.merge(left = df,
              right = txt_feature,
              how = 'left',
              on = ['user_id','item_id'])


# In[11]:


df = pd.merge(left = df,
              right = img_feature,
              how = 'left',
              on = ['user_id','item_id'])


# In[12]:


df.to_csv(train_path + 'new_recall/' + file_name + '_addtxt.csv', index=False)


# In[ ]:


txt_df = pd.DataFrame(txt_similarity.items())
txt_df.columns = ['item_pair','txt_sim']
txt_df = txt_df[~pd.isna(txt_df['txt_sim'])]
txt_df['txt_sim'] = txt_df['txt_sim'].astype(np.float16)
txt_df.to_csv('txt_similarity.csv', index=False)
txt_df = []
txt_similarity = []
gc.collect()

img_df = pd.DataFrame(img_similarity.items())
img_df.columns = ['item_pair','img_sim']
img_df = img_df[~pd.isna(img_df['img_sim'])]
img_df['img_sim'] = img_df['img_sim'].astype(np.float16)
img_df.to_csv('img_similarity.csv', index=False)

