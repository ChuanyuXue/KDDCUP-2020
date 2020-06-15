#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from gensim.models import KeyedVectors
import gc


# In[2]:


train_path = './user_data/dataset/'
test_path = './user_data/dataset/'
header = 'underexpose'
now_phase = 9


# In[3]:


file_name = 'recall_0531_addsim_addAA_RA_additemtime_addcount_addnn'
df = pd.read_csv(train_path + 'new_recall/' + file_name + '.csv')


# # 线上只需提交对7-9阶段的预测

# In[4]:


user_item_list = []
for phase in range(now_phase + 1):
    file = open(train_path + 'new_similarity/' + 'user2item_new%d.pkl'%phase, 'rb')
    user_item_list.append(pickle.load(file))


# In[5]:


txt_model = KeyedVectors.load_word2vec_format('./user_data/w2v_txt_vec.txt')
img_model = KeyedVectors.load_word2vec_format('./user_data/w2v_img_vec.txt')


# In[6]:



txt_similarity = {}
img_similarity = {}


# In[ ]:



txt_feature = []
img_feature = []
for phase in range(7,now_phase + 1):
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
                    txt_sim = txt_model.similarity(str(item), str(related_item))
                except:
                    txt_sim = np.nan
            txt_similarity[index] = txt_sim
            txt_sim_list.append(txt_sim)
                
            # calculate img similarity
            
            if index in img_similarity:
                img_sim = img_similarity[index]
            else:
                try:
                    img_sim = img_model.similarity(str(item), str(related_item))
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


# In[ ]:


txt_feature = pd.DataFrame(txt_feature, columns=['user_id','item_id'] + ['txt_feature_' + str(x) for x in range(5)])
img_feature = pd.DataFrame(img_feature, columns=['user_id','item_id'] + ['img_feature_' + str(x) for x in range(5)])


# In[ ]:


df = pd.merge(left = df,
              right = txt_feature,
              how = 'left',
              on = ['user_id','item_id'])


# In[ ]:


df = pd.merge(left = df,
              right = img_feature,
              how = 'left',
              on = ['user_id','item_id'])


# In[ ]:


df.to_csv(train_path + 'new_recall/' + file_name + '_addtxt.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




