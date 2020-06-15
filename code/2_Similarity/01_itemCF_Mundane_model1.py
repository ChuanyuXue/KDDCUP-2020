#!/usr/bin/env python
# coding: utf-8

# In[13]:


from __future__ import division
from __future__ import print_function
from gensim.models import KeyedVectors
import gc
import os
import math
import time
import random
import joblib
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pickle
from multiprocessing import Pool as ProcessPool
import json


# In[14]:


random.seed(2020)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_colwidth", 100)
pd.set_option('display.width', 1000)


# In[15]:


def process(each_item):
    dict_tmp = item_sim_list[each_item]
    for j in dict_tmp:
        dict_tmp[j] = round(dict_tmp[j],4)
        dict_tmp[j] = round(dict_tmp[j],4)
    
    return (each_item,dict_tmp)

def myround(x, thres):
    temp = 10**thres
    return int(x * temp) / temp


# In[16]:


myround = lambda x,thres : int(x * 10**thres) / 10**thres


# In[17]:


def phase_predict(df, pred_col, top_fill, topk=50):
    """recom_df, 'sim', top50_click, "click_valid"
    """
    top_fill = [int(t) for t in top_fill.split(',')]
    top_fill = top_fill[:topk]
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
    df.sort_values("rank", inplace=True)
    df = df[df["rank"] <= topk]
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',',
                                                                                                   expand=True).reset_index()
    return df


def get_sim_item(df_, user_col, item_col):#, nodewalk_model,deepwalk_model,txt_vec_model):
    global txt_similarity
    global deepwalk_similarity
    global nodewalk_similarity

    df = df_.copy()
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    user_time_ = df.groupby(user_col)['time'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))

    item_user_ = df.groupby(item_col)[user_col].agg(set).reset_index()
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col]))

    item_dic = df[item_col].value_counts().to_dict()

    df.sort_values('time', inplace=True)
    df.drop_duplicates('item_id', keep='first', inplace=True)
    item_time_ = df.groupby(item_col)['time'].agg(list).reset_index()  # 引入时间因素
    item_time_dict = dict(zip(item_time_[item_col], item_time_['time']))


    sim_item = {}
    item_cnt = defaultdict(int)  # 商品被点击次数
    for user, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            users = item_user_dict[item]
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            user_item_len = len(items)
            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                t1 = user_time_dict[user][loc1]  # 点击时间提取
                t2 = user_time_dict[user][loc2]
                delta_t = abs(t1 - t2) * 650000
                delta_loc = abs(loc1 - loc2)
                '''
                The meaning of each columns:
                {'sim': 0,------------------------0
                  'item_cf': 0,-------------------1
                  'item_cf_weighted': 0,----------2
                  'time_diff': np.inf,------------3
                  'loc_diff': np.inf,-------------4
                  'node_sim_max': -1e8,-----------5
                  'node_sim_sum':0,---------------6
                  'deep_sim_max': -1e8,-----------7
                  'deep_sim_sum':0----------------8
                                          }
                '''
                
                sim_item[item].setdefault(relate_item,
                                          [0,0,0,np.inf,np.inf,-1e8,0,-1e8,0]
                                         )
                
                
                key = [str(int(item)), str(int(relate_item))]
                key_tmp = "_".join(key)
                
                ##nodewalk
                if key_tmp in nodewalk_similarity:
                    node_sim = nodewalk_similarity[key_tmp]
                else:
                    try:
                        node_sim = 0.5 * nodewalk_model.similarity(str(item), str(relate_item))+ 0.5
                    except:
                        node_sim = 0.5
                    nodewalk_similarity[key_tmp] = node_sim
                    
                ##deepwalk
                if key_tmp in deepwalk_similarity:
                    deep_sim = deepwalk_similarity[key_tmp]
                else:
                    try:
                        deep_sim = 0.5 * deepwalk_model.similarity(str(item), str(relate_item))+ 0.5
                    except:
                        deep_sim = 0.5
                    deepwalk_similarity[key_tmp] = deep_sim

                #txt
                if key_tmp in txt_similarity:
                    txt_sim = txt_similarity[key_tmp]
                else:
                    try:
                        txt_sim = 0.5 * txt_model.similarity(str(item), str(relate_item))+ 0.5
                    except:
                        txt_sim = 0.5
                    txt_similarity[key_tmp] = txt_sim

                '''
                WIJ
                The meaning of each columns:
                {'sim': 0,------------------------0
                  'item_cf': 0,-------------------1
                  'item_cf_weighted': 0,----------2
                  'time_diff': np.inf,------------3
                  'loc_diff': np.inf,-------------4
                  'node_sim_max': -1e8,-----------5
                  'node_sim_sum':0,---------------6
                  'deep_sim_max': -1e8,-----------7
                  'deep_sim_sum':0----------------8
                                          }
                '''
                
                if loc1 - loc2 > 0:
                    sim_item[item][relate_item][0] += (node_sim**2)*deep_sim*txt_sim * 0.8 * max(0.5, (0.9 ** (loc1 - loc2 - 1))) * (
                        max(0.5, 1 / (1 + delta_t))) / (math.log(len(users) + 1) * math.log(
                        1 + user_item_len))
                else:                 
                    sim_item[item][relate_item][0] += (node_sim**2)*deep_sim*txt_sim * 1.0 * max(0.5, (0.9 ** (loc2 - loc1 - 1))) * (
                        max(0.5, 1 / (1 + delta_t))) / (math.log(len(users) + 1) * math.log(
                        1 + user_item_len))
                
                if delta_t < sim_item[item][relate_item][3]:
                    sim_item[item][relate_item][3] = delta_t
                if delta_loc < sim_item[item][relate_item][4]:
                    sim_item[item][relate_item][4] = delta_loc
                sim_item[item][relate_item][1] += 1
                sim_item[item][relate_item][2] += (0.8**(loc2-loc1-1)) * (1 - (t2 - t1) * 2000) / math.log(1 + len(items))
                
                if node_sim > sim_item[item][relate_item][5]:
                    sim_item[item][relate_item][5] = node_sim
                sim_item[item][relate_item][6] += node_sim
                
                if deep_sim > sim_item[item][relate_item][7]:
                    sim_item[item][relate_item][7] = deep_sim
                sim_item[item][relate_item][8] += deep_sim
                
                

    sim_item_corr = sim_item.copy()
    for i, related_items in tqdm(sim_item.items()):
        for j, cij in related_items.items():
            cosine_sim = cij[0] / ((item_cnt[i] * item_cnt[j]) ** 0.2)
            sim_item_corr[i][j][0] = cosine_sim
            sim_item_corr[i][j] = [myround(x, 4) for x in sim_item_corr[i][j]]


    return sim_item_corr, user_item_dict, user_time_dict, item_dic, item_time_dict


def recommend(sim_item_corr, user_item_dict, user_id, times, item_dict, item_time_dict, top_k, item_num):
    '''
    input:item_sim_list, user_item, uid, 500, 50
    # 用户历史序列中的所有商品均有关联商品,整合这些关联商品,进行相似性排序
    '''
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]
    times = times[::-1]
    t0 = times[0]
    for loc, i in enumerate(interacted_items):
        for j, wij in sorted(sim_item_corr[i].items(), key=lambda d: d[1][0], reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, [0,0,0,np.inf,np.inf,np.inf,np.inf,np.inf,-1e8,0,-1e8,0])
                '''
                RANK
                {'sim': 0,---------------------------------0
                'item_cf': 0,------------------------------1
                'item_cf_weighted': 0,---------------------2
                'time_diff': np.inf,-----------------------3
                'loc_diff': np.inf,------------------------4
                # Some feature generated by recall
                'time_diff_recall': np.inf,----------------5
                'time_diff_recall_1': np.inf,--------------6
                'loc_diff_recall': np.inf,-----------------7
                # Nodesim and Deepsim
                  'node_sim_max': -1e8,--------------------8
                  'node_sim_sum':0,------------------------9
                  'deep_sim_max': -1e8,--------------------10
                  'deep_sim_sum':0,------------------------11
                                          }
                '''
                t1 = times[loc]
                t2 = item_time_dict[j][0]
                delta_t1 = abs(t0 - t1) * 650000
                delta_t2 = abs(t0 - t2) * 650000
                alpha = max(0.2, 1 / (1 + item_dict[j]))
                beta = max(0.5, (0.9 ** loc))
                theta = max(0.5, 1 / (1 + delta_t1))
                gamma = max(0.5, 1 / (1 + delta_t2))
                
                '''
                RANK
                {'sim': 0,---------------------------------0
                'item_cf': 0,------------------------------1
                'item_cf_weighted': 0,---------------------2
                'time_diff': np.inf,-----------------------3
                'loc_diff': np.inf,------------------------4
                # Some feature generated by recall
                'time_diff_recall': np.inf,----------------5
                'time_diff_recall_1': np.inf,--------------6
                'loc_diff_recall': np.inf,-----------------7
                # Nodesim and Deepsim
                  'node_sim_max': -1e8,--------------------8
                  'node_sim_sum':0,------------------------9
                  'deep_sim_max': -1e8,--------------------10
                  'deep_sim_sum':0,------------------------11
                                          }
                '''
                
                '''
                WIJ
                The meaning of each columns:
                {'sim': 0,------------------------0
                  'item_cf': 0,-------------------1
                  'item_cf_weighted': 0,----------2
                  'time_diff': np.inf,------------3
                  'loc_diff': np.inf,-------------4
                  'node_sim_max': -1e8,-----------5
                  'node_sim_sum':0,---------------6
                  'deep_sim_max': -1e8,-----------7
                  'deep_sim_sum':0----------------8
                                          }
                '''
                

                rank[j][0] += myround(wij[0] * (alpha ** 2) * (beta) * (theta ** 2) * gamma, 4)
                rank[j][1] += wij[1]
                rank[j][2] += wij[2]
                
                if wij[3] < rank[j][3]:
                    rank[j][3] = wij[3]
                if wij[4] < rank[j][4]:
                    rank[j][4] = wij[4]
                if delta_t1 < rank[j][5]:
                    rank[j][5] = myround(delta_t1, 4)
                if delta_t2 < rank[j][6]:
                    rank[j][6] = myround(delta_t2, 4)
                if loc < rank[j][7]:
                    rank[j][7] = loc
                    
                if wij[5] > rank[j][8]:
                    rank[j][8] = wij[5]
                rank[j][9] += wij[6] / wij[1]
                
                if wij[7] > rank[j][10]:
                    rank[j][10] = wij[7]
                rank[j][11] += wij[8] / wij[1]
                
    return sorted(rank.items(), key=lambda d: d[1][0], reverse=True)[:item_num]


# In[18]:


now_phase = 9
header = 'model_1'
txt_similarity = {}
deepwalk_similarity = {}
nodewalk_similarity = {}
offline = "./user_data/model_1/"
out_path = './user_data/model_1/new_similarity/'

print("start")
print("read sim")

nodewalk_model = KeyedVectors.load_word2vec_format(offline + 'node2vec_' + header + '.bin',binary=True)

deepwalk_model = KeyedVectors.load_word2vec_format(offline + 'deepwalk_' + header + '.bin',binary=True)

txt_model = KeyedVectors.load_word2vec_format('./user_data/w2v_txt_vec.txt')


# In[19]:


recom_item = []
for phase in range(now_phase + 1):
    a = time.time()
    history_list = []
    for i in range(now_phase + 1):
        click_train = pd.read_csv(offline + header + '_train_click-{}.csv'.format(i), header=None,
                                  names=['user_id', 'item_id', 'time'])
        click_test = pd.read_csv(offline + header + '_test_click-{}.csv'.format(i), header=None,
                                 names=['user_id', 'item_id', 'time'])
        all_click = click_train.append(click_test)
        history_list.append(all_click)

    qtime_test = pd.read_csv(offline + header + '_test_qtime-{}.csv'.format(phase), header=None,
                              names=['user_id', 'item_id', 'time'])

    print('phase:', phase)
    time_diff = max(history_list[now_phase]['time']) - min(history_list[0]['time'])
    for i in range(phase + 1, now_phase + 1):
        history_list[i]['time'] = history_list[i]['time'] - time_diff

    whole_click = pd.DataFrame()
    for i in range(now_phase + 1):
        whole_click = whole_click.append(history_list[i])


    whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    whole_click = whole_click.sort_values('time')
    whole_click = whole_click.reset_index(drop=True)


    item_sim_list, user_item, user_time_dict, item_dic, item_time_dict = get_sim_item(whole_click,
                                                                                      'user_id',
                                                                                      'item_id'
                                                                                      )       


    print("phase finish time:{:6.4f} mins".format((time.time() - a) / 60))
    
    for user in tqdm(qtime_test['user_id'].unique()):
        if user in user_time_dict:
            times = user_time_dict[user]
            rank_item = recommend(item_sim_list, user_item, user, times, item_dic, item_time_dict, 500, 1000)
            for j in rank_item:
                recom_item.append([user, int(j[0])] + j[1])    
                
    for i, related_items in tqdm(item_sim_list.items()):
        for j, cij in related_items.items():
            item_sim_list[i][j] = cij[0]
    
    write_file = open(out_path+'itemCF_new'+str(phase)+'.pkl', 'wb')
    pickle.dump(item_sim_list, write_file)
    write_file.close() 

    write_file = open(out_path+'user2item_new'+str(phase)+'.pkl', 'wb')
    pickle.dump(user_item, write_file)
    write_file.close()     

    write_file = open(out_path+'item2cnt_new'+str(phase)+'.pkl', 'wb')
    pickle.dump(item_dic, write_file)
    write_file.close() 

    write_file = open(out_path+'userTime'+str(phase)+'.pkl', 'wb')
    pickle.dump(user_time_dict, write_file)
    write_file.close()         

    write_file = open(out_path+'itemTime'+str(phase)+'.pkl', 'wb')
    pickle.dump(item_time_dict, write_file)
    write_file.close()  
    
    write_file = open(out_path+'recom_item'+'.pkl', 'wb')
    pickle.dump(recom_item, write_file)
    write_file.close() 

    
    del item_sim_list
    del user_item
    del user_time_dict
    del item_dic
    del item_time_dict
    gc.collect()

