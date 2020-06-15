#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from gensim.models import KeyedVectors
from tqdm import tqdm
import numpy as np
import gc


# In[2]:


train_path = './user_data/dataset/'
test_path = './user_data/dataset/'
header = 'underexpose'
now_phase = 9


# In[3]:


file_name = 'recall_0531_addsim_addAA_RA_additemtime_addcount_addnn_addtxt_interactive_countdetail_userfeature_partialsim'
df = pd.read_csv(train_path + 'new_recall/' + file_name + '.csv')


# In[4]:


user_item_list = []
for phase in tqdm(range(now_phase + 1)):

    click_train = pd.read_csv(train_path + header + '_train_click-{}.csv'.format(phase), header=None,
                              names=['user_id', 'item_id', 'time'])
    click_test = pd.read_csv(train_path + header + '_test_click-{}.csv'.format(phase), header=None,
                             names=['user_id', 'item_id', 'time'])
    all_click = click_train.append(click_test)
    
    all_click = all_click.sort_values('time')
    all_click = all_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    all_click = all_click.sort_values('time')
    all_click = all_click.reset_index(drop=True)

    user_item_ = all_click.groupby('user_id')['item_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    user_item_list.append(user_item_dict)


# In[5]:


# user_item_list = []
# for phase in range(now_phase + 1):
#     file = open(train_path + 'new_similarity/' + 'user2item_new%d.pkl'%phase, 'rb')
#     user_item_list.append(pickle.load(file))


# In[6]:


txt_model = KeyedVectors.load_word2vec_format('./user_data/dataset/w2v_txt_vec.txt')
img_model = KeyedVectors.load_word2vec_format('./user_data/dataset/w2v_img_vec.txt')


# In[7]:


nodewalk_model = KeyedVectors.load_word2vec_format('./user_data/2_New_Similarity/node2vec_' + header + '.bin',binary=True)
deepwalk_model = KeyedVectors.load_word2vec_format('./user_data/2_New_Similarity/deepwalk_' + header + '.bin',binary=True)


# In[8]:


txt_similarity = {}
img_similarity = {}
deep_similarity = {}
node_similarity = {}

emergency_feature = []


# In[9]:


for phase in range(0, now_phase + 1):
    current_recall = df[df['phrase'] == phase]
    current_data = user_item_list[phase]
    for eachrow in tqdm(current_recall[['user_id','item_id']].values):
        related_item = current_data[eachrow[0]][-1]
        item = eachrow[1]

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

        # calculate img similarity
        if index in img_similarity:
            img_sim = img_similarity[index]
        else:
            try:
                img_sim = int(img_model.similarity(str(item), str(related_item)) * 1e4) / 1e4
            except:
                img_sim = np.nan
        img_similarity[index] = img_sim
            
        # calculate node similarity
        if index in node_similarity:
            node_sim = node_similarity[index]
        else:
            try:
                node_sim = int(nodewalk_model.similarity(str(item), str(related_item)) * 1e4) / 1e4
            except:
                node_sim = np.nan
        node_similarity[index] = node_sim
        
        # calculate deep similarity
        if index in deep_similarity:
            deep_sim = deep_similarity[index]
        else:
            try:
                deep_sim = int(deepwalk_model.similarity(str(item), str(related_item)) * 1e4) / 1e4
            except:
                deep_sim = np.nan
        deep_similarity[index] = deep_sim
        
        emergency_feature.append([eachrow[0], eachrow[1], txt_sim, img_sim, node_sim, deep_sim])
        
    gc.collect()


# In[10]:


emergency_feature = pd.DataFrame(emergency_feature, columns=['user_id','item_id'] + ['emergency_feature_' + str(x) for x in range(4)])


# In[11]:


df = pd.merge(left = df,
              right = emergency_feature,
              how = 'left',
              on = ['user_id','item_id'])


# In[12]:


df.to_csv(train_path + 'new_recall/' + file_name + '_emergency.csv', index=False)

