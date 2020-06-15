import sys
import copy
import random
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


def evaluate6(model, dataset,user2idmap2, args, sess, id2item, id2user, 
              save_path='pred_valid.csv', read_path='all/offline.csv'):
    [train, usernum, itemnum] = copy.deepcopy(dataset)
    pred = []
    item_idx = list(range(1, itemnum + 1))
    id2itme_list = [id2item[i] for i in item_idx]
    df2 = pd.read_csv(read_path)
    item_map = {v:k for (k,v) in id2item.items()}
     
    for u in tqdm(train.keys()):

        if len(train[u]) < 1:
            print(u)
            continue
        score = []
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        u2 = id2user[u]
        # u2 = user2idmap2[int(u2.split('_')[-1])]
        u2 = user2idmap2[u2[2:]]
        predictions = model.predict(sess, [u2], [seq], item_idx)
        predictions = predictions[0]
        idx = np.argsort(predictions)[::-1][:500]
        # tmp_list = [id2itme_list[idx[i]] for i in range(500)]
        # score = [predictions[idx[i]] for i in range(500)]
        tmp_list = []
        score = []
        
        tmp_df = df2[df2['user_id'] == int(id2user[u].split('_')[-1])]['item_id']
        if len(tmp_df)>0:
            items = set(tmp_df.values[0][1:-1].split(','))
            tmp_list_set = set(tmp_list)
            for tmp_item in items:
                tmp_ = int(tmp_item)
                if tmp_ not in tmp_list_set:
                    tmp_idx = item_map[tmp_]
                    tmp_list.append(tmp_)
                    score.append(predictions[tmp_idx-1])
                       
        pred.append([id2user[u]] + [tmp_list] + [score])
        
    df = pd.DataFrame(pred)
    df[0] = df[0].apply(lambda x: x.split('_')[-1])
    df.columns = ['user', 'item', 'score']
    df.to_csv(save_path, index=None)
    return df

def evaluate5(model, dataset,user2idmap2, args, sess, id2item, id2user, 
              save_path='pred_valid.csv', read_path='all/offline.csv'):
    [train, usernum, itemnum] = copy.deepcopy(dataset)
    pred = []
    item_idx = list(range(1, itemnum + 1))
    id2itme_list = [id2item[i] for i in item_idx]
    df2 = pd.read_csv(read_path)
    df2 = df2.groupby('user_id')['item_id'].apply(list).reset_index()
    
    item_map = {v:k for (k,v) in id2item.items()}
     
    for u in tqdm(train.keys()):

        if len(train[u]) < 1:
            print(u)
            continue
        score = []
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        u2 = id2user[u]
        # u2 = user2idmap2[int(u2.split('_')[-1])]
        u2 = user2idmap2[u2[2:]]
        predictions = model.predict(sess, [u2], [seq], item_idx)
        predictions = predictions[0]
        idx = np.argsort(predictions)[::-1][:500]
        tmp_list = [id2itme_list[idx[i]] for i in range(500)]
        score = [predictions[idx[i]] for i in range(500)]
        
        tmp_df = df2[df2['user_id'] == int(id2user[u].split('_')[-1])]['item_id']
        if len(tmp_df)>0:
            items = set(tmp_df.values[0]) # [1:-1].split(',')
            tmp_list_set = set(tmp_list)
            for tmp_item in items:
                tmp_ = int(tmp_item)
                if tmp_ not in tmp_list_set:
                    tmp_idx = item_map[tmp_]
                    tmp_list.append(tmp_)
                    score.append(predictions[tmp_idx-1])
                       
        pred.append([id2user[u]] + [tmp_list] + [score])
        
    df = pd.DataFrame(pred)
    df[0] = df[0].apply(lambda x: x.split('_')[-1])
    df.columns = ['user', 'item', 'score']
    df.to_csv(save_path, index=None)
    return df


def evaluate4(model, dataset,user2idmap2, args, sess, id2item, id2user, user2idmap3):
    [train, usernum, itemnum] = copy.deepcopy(dataset)
    pred = []
    item_idx = list(range(1, itemnum + 1))
    id2itme_list = [id2item[i] for i in item_idx]
    
    for u in tqdm(train.keys()):

        if len(train[u]) < 1:
            print(u)
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        u2 = id2user[u]
        u3 = user2idmap3[int(u2.split('_')[-1])]
        
        u2 = user2idmap2[u2[2:]]
        # 
        predictions = model.predict(sess, [u2], [u3], [seq], item_idx)
        predictions = predictions[0]
        idx = np.argsort(predictions)[::-1][:50]
        tmp_list = [id2itme_list[idx[i]] for i in range(50)]
        pred.append([id2user[u]] + tmp_list)
        
    df = pd.DataFrame(pred)
    df[0] = df[0].apply(lambda x: x.split('_')[-1])
    df.to_csv('pred_valid.csv', index=None, header=None)
    return df


def evaluate3(model, dataset, args, sess, id2item, id2user, time_array):
    [train, usernum, itemnum] = copy.deepcopy(dataset)
    pred = []
    item_idx = list(range(1, itemnum + 1))
    id2itme_list = [id2item[i] for i in item_idx]
    
    for u in tqdm(train.keys()):

        if len(train[u]) < 1:
            print(u)
            continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        t = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t_ in zip(reversed(train[u]), reversed(time_array[u])):
            seq[idx] = i
            t[idx] = t_
            idx -= 1
            if idx == -1: break
        
        predictions = model.predict(sess, [u], [seq], item_idx, [t])
        predictions = predictions[0]
        idx = np.argsort(predictions)[::-1][:50]
        tmp_list = [id2itme_list[idx[i]] for i in range(50)]
        pred.append([id2user[u]] + tmp_list)
        
    df = pd.DataFrame(pred)
    df[0] = df[0].apply(lambda x: x.split('_')[-1])
    df.to_csv('pred_valid.csv', index=None, header=None)
    return df


def evaluate2(model, dataset,user2idmap2, args, sess, id2item, id2user, 
              save_path='pred_valid.csv'):
    [train, usernum, itemnum] = copy.deepcopy(dataset)
    pred = []
    item_idx = list(range(1, itemnum + 1))
    id2itme_list = [id2item[i] for i in item_idx]
    
    for u in tqdm(train.keys()):

        if len(train[u]) < 1:
            print(u)
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        u2 = id2user[u]
        # u2 = user2idmap2[int(u2.split('_')[-1])]
        u2 = user2idmap2[u2[2:]]
        predictions = model.predict(sess, [u2], [seq], item_idx)
        predictions = predictions[0]
        idx = np.argsort(predictions)[::-1][:50]
        tmp_list = [id2itme_list[idx[i]] for i in range(50)]
        pred.append([id2user[u]] + tmp_list)
        
    df = pd.DataFrame(pred)
    df[0] = df[0].apply(lambda x: x.split('_')[-1])
    df.to_csv(save_path, index=None, header=None)
    return df

def evaluate(model, dataset, args, sess, id2item, id2user):
    [train, usernum, itemnum] = copy.deepcopy(dataset)
    pred = []
    item_idx = list(range(1, itemnum + 1))
    id2itme_list = [id2item[i] for i in item_idx]
    
    for u in tqdm(train.keys()):

        if len(train[u]) < 1:
            print(u)
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        predictions = model.predict(sess, [u], [seq], item_idx)
        predictions = predictions[0]
        idx = np.argsort(predictions)[::-1][:50]
        tmp_list = [id2itme_list[idx[i]] for i in range(50)]
        pred.append([id2user[u]] + tmp_list)
        
    df = pd.DataFrame(pred)
    df[0] = df[0].apply(lambda x: x.split('_')[-1])
    df.to_csv('pred_valid.csv', index=None, header=None)
    return df