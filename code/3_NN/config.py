#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:36:15 2020

@author: hcb
"""

class config:
    train_path = './user_data/dataset'
    test_path = './user_data/dataset'
    offline_path = './user_data/offline'
    model1_path = './user_data/model_1'
    
    save_path_offline = './user_data/offline/nn/nn_offline.csv'
    save_path_online = './user_data/dataset/nn/nn_underexpose.csv'
    save_path_model1 = './user_data/model_1/nn/nn_model_1.csv'
    
    online_item_file = './user_data/dataset/new_recall/user_item_index.csv'
    offline_item_file = './user_data/offline/new_recall/user_item_index.csv'
    model1_item_file = './user_data/model_1/new_recall/user_item_index.csv'
    # online_path = ''