#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict  
import math  
import json
from sys import stdout
import pickle


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # RA、AA一起运行的

# In[2]:


now_phase = 9

input_path = './user_data/offline/new_similarity/'
out_path = './user_data/offline/new_similarity/'


for num in range(now_phase+1):
    
    # 获取itemCF相似度
    with open(input_path+'itemCF_new'+str(num)+'.pkl','rb') as f:
        item_sim_list_tmp = pickle.load(f)  
    
    item_sim = {}
    for item in item_sim_list_tmp:
        item_sim.setdefault(item, {})
        for related_item in item_sim_list_tmp[item]:
            if item_sim_list_tmp[item][related_item] > 0.005:
                item_sim[item][related_item] = item_sim_list_tmp[item][related_item]
    
    item_sim_list_tmp = []
    
    strengh_dict = dict()
    print('Counting degree')
    for item in tqdm(item_sim):
        strengh_dict[item] = sum(item_sim[item].values())       
        
    strengh_AA_dict = dict()
    print('Counting degree')
    for item in tqdm(item_sim):
        strengh_AA_dict[item] = math.log(1+sum(item_sim[item].values()) )
        
        
    #RA
    RA_sim = dict()
    for item in tqdm(item_sim):
        neighbors = list(set(item_sim[item].keys()))
        for item1 in neighbors:
            if item in item_sim[item1]:
                RA_sim.setdefault(item1, {})
                for item2 in neighbors:
                    if item1 != item2:
                        RA_sim[item1].setdefault(item2, 0)
                        RA_sim[item1][item2] += item_sim[item1][item] * item_sim[item][item2]/strengh_dict[item]
    
    
    new_RA = dict()
    for item1 in tqdm(RA_sim):
        new_RA[item1] = {i: int(x * 1e3) / 1e3 for i, x in RA_sim[item1].items() if x > 1e-3}
    
    RA_sim = []
    print('Saving')
    write_file = open(out_path+'RA_P'+str(num)+'_new.pkl', 'wb')
    pickle.dump(new_RA, write_file)
    write_file.close() 
    
        
    new_RA = []
    
    
    #RA
    AA_sim = dict()
    for item in tqdm(item_sim):
        neighbors = list(set(item_sim[item].keys()))
        for item1 in neighbors:
            if item in item_sim[item1]:
                AA_sim.setdefault(item1, {})
                for item2 in neighbors:
                    if item1 != item2:
                        AA_sim[item1].setdefault(item2, 0)
                        AA_sim[item1][item2] += item_sim[item1][item] * item_sim[item][item2]/strengh_AA_dict[item]
    
    
    new_AA = dict()
    for item1 in tqdm(AA_sim):
        new_AA[item1] = {i: int(x * 1e3) / 1e3 for i, x in AA_sim[item1].items() if x > 1e-3}
    
    AA_sim = []
    print('Saving')
    write_file = open(out_path+'AA_P'+str(num)+'_new.pkl', 'wb')
    pickle.dump(new_AA, write_file)
    write_file.close() 
    
        
    new_AA = []    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # CN、HPI、HDI、LHN1是一起运行的

# In[3]:


now_phase = 9

input_path = './user_data/offline/new_similarity/'
out_path = './user_data/offline/new_similarity/'



for num in range(now_phase+1):
    
    # 获取itemCF相似度
    with open(input_path+'itemCF_new'+str(num)+'.pkl','rb') as f:
        item_sim_list_tmp = pickle.load(f)  
    
    item_sim = {}
    for item in item_sim_list_tmp:
        item_sim.setdefault(item, {})
        for related_item in item_sim_list_tmp[item]:
            if item_sim_list_tmp[item][related_item] > 0.005:
                item_sim[item][related_item] = item_sim_list_tmp[item][related_item]
    
    item_sim_list_tmp = []
    
    #CN
    CN_sim = dict()
    for item in tqdm(item_sim):
        neighbors = list(set(item_sim[item].keys()))
        for item1 in neighbors:
            if item in item_sim[item1]:
                CN_sim.setdefault(item1, {})
                for item2 in neighbors:
                    if item1 != item2:
                        CN_sim[item1].setdefault(item2, 0)
                        CN_sim[item1][item2] += item_sim[item1][item] * item_sim[item][item2]
    
    
    new_CN = dict()
    for item1 in tqdm(CN_sim):
        new_CN[item1] = {i: int(x * 1e3) / 1e3 for i, x in CN_sim[item1].items() if x > 1e-3}
    
    CN_sim = []
    print('Saving')
    write_file = open(out_path+'CN_P'+str(num)+'_new.pkl', 'wb')
    pickle.dump(new_CN, write_file)
    write_file.close() 
    
    strengh_dict = dict()
    print('Counting degree')
    for item in tqdm(item_sim):
        strengh_dict[item] = sum(item_sim[item].values())     
    
    #HPI
    HPI_sim = dict()
    for item in tqdm(new_CN):
        HPI_sim.setdefault(item,{})
        for related_item in new_CN[item]:
            HPI_sim[item][related_item] = new_CN[item][related_item]/min(strengh_dict[item],strengh_dict[related_item])       
            
    print('Saving')
    write_file = open(out_path+'HPI_P'+str(num)+'_new.pkl', 'wb')
    pickle.dump(HPI_sim, write_file)
    write_file.close()
    
    HPI_sim = []
    
    
    #HDI
    HDI_sim = dict()
    for item in tqdm(new_CN):
        HDI_sim.setdefault(item,{})
        for related_item in new_CN[item]:
            HDI_sim[item][related_item] = new_CN[item][related_item]/max(strengh_dict[item],strengh_dict[related_item])       
            
    print('Saving')
    write_file = open(out_path+'HDI_P'+str(num)+'_new.pkl', 'wb')
    pickle.dump(HDI_sim, write_file)
    write_file.close()    
    HDI_sim = []
    
    
    
    #LHN1
    LHN1_sim = dict()
    for item in tqdm(new_CN):
        LHN1_sim.setdefault(item,{})
        for related_item in new_CN[item]:
            LHN1_sim[item][related_item] = new_CN[item][related_item]/(strengh_dict[item]*strengh_dict[related_item])       
            
    print('Saving')
    write_file = open(out_path+'LHN1_P'+str(num)+'_new.pkl', 'wb')
    pickle.dump(LHN1_sim, write_file)
    write_file.close()    
    LHN1_sim = []
    
        
    new_CN = []


# In[ ]:





# In[ ]:





# In[ ]:




