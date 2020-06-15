#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:juzphy
# datetime:2020/4/26 3:46 下午
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sampler2 import WarpSampler
from model2 import Model
import os
from util import *
import numpy as np
import argparse
from config import config

def get_data(now_phase, train_path, test_path, kind=1):
    click_train = pd.DataFrame()
    click_test = pd.DataFrame()
    for c in range(now_phase + 1):
        if kind == 1:
            click_tmp = pd.read_csv(os.path.join(train_path, f'underexpose_train_click-{c}.csv'), header=None,
                        names=['user_id', 'item_id', 'time'],converters={'time':np.float64})
            click_test_tmp = pd.read_csv(os.path.join(test_path, f'underexpose_test_click-{c}.csv'), header=None,
                             names=['user_id', 'item_id', 'time'])
            
        elif kind == 2:
            click_tmp = pd.read_csv(os.path.join(train_path, 'offline' + f'_train_click-{c}.csv'), header=None,
                        names=['user_id', 'item_id', 'time'],converters={'time':np.float64})
            click_test_tmp = pd.read_csv(os.path.join(test_path, 'offline' + f'_test_click-{c}.csv'), header=None,
                                         names=['user_id', 'item_id', 'time'])
            
        elif kind == 3:
            click_tmp = pd.read_csv(os.path.join(train_path, 'model_1' + f'_train_click-{c}.csv'), header=None,
                        names=['user_id', 'item_id', 'time'],converters={'time':np.float64})            
            click_test_tmp = pd.read_csv(os.path.join(test_path, 'model_1' + f'_test_click-{c}.csv'), header=None,
                                         names=['user_id', 'item_id', 'time'])
        
        # click_tmp['user_id2'] = click_tmp['user_id']
        click_tmp['user_id2'] ='{}_'.format(c) + click_tmp['user_id'].astype(str)
        click_tmp['user_id'] = '1_{}_'.format(c) + click_tmp['user_id'].astype(str)
        
            
        # click_test_tmp['user_id2'] = click_test_tmp['user_id']
        click_test_tmp['user_id2'] = '{}_'.format(c) + click_test_tmp['user_id'].astype(str)
        click_test_tmp['user_id'] = '0_{}_'.format(c) + click_test_tmp['user_id'].astype(str)
        click_train = click_train.append(click_tmp)
        click_test = click_test.append(click_test_tmp)
    
    # click_train.drop_duplicates(['item_id','time', 'user_id2'], inplace=True)
    
    all_click = click_train.append(click_test)
    num_items = all_click['item_id'].nunique()
    num_users = all_click['user_id'].nunique()
    num_users2 = all_click['user_id2'].nunique()
    item2idmap = dict(zip(all_click['item_id'].unique(), range(1, 1 + num_items)))
    user2idmap = dict(zip(all_click['user_id'].unique(), range(1, 1 + num_users)))
    user2idmap2 = dict(zip(all_click['user_id2'].unique(), range(1, 1 + num_users2)))
    all_click['map_user'] = all_click['user_id'].map(user2idmap)
    all_click['map_item'] = all_click['item_id'].map(item2idmap)
    item_deg = all_click['map_item'].value_counts().to_dict()
    
    use_train, use_valid, use_test = {}, {}, {}
    
    all_click = all_click.sort_values('time').groupby('user_id')['map_item'].apply(list).to_dict()

    for reviewerID, hist in tqdm(all_click.items()):
        is_train = reviewerID.split('_')[0]
        phase = reviewerID.split('_')[1]
        user = user2idmap[reviewerID]
            
        if is_train == '1':
            # if phase == str(now_phase):
            if phase in ['7', '8', '9']:
                use_train[user] = hist[:-1]
                use_valid[user] = [hist[-1]]
            else:
                use_train[user] = hist
                use_valid[user] = []                
        else:
            use_train[user] = hist
            use_valid[user] = []
            #if phase in ['7', '8', '9']:
            use_test[user] = hist

    id2item = dict()
    for tmp_key in item2idmap.keys():
        id2item[item2idmap[tmp_key]] = tmp_key
    id2user = dict()
    for tmp_key in user2idmap.keys():
        id2user[user2idmap[tmp_key]] = tmp_key
    
    emb = pd.read_csv('item_feat.csv')
    emb['item_id'] = emb['item_id'].map(item2idmap)
    emb = emb.sort_values('item_id', ascending=True).reset_index(drop=True)
    emb = emb[emb.columns[2:]].values
    return use_train, use_valid, num_items, num_users, id2item, id2user, \
        item_deg, emb, use_test, user2idmap2, num_users2
        
def eval_model(model, sess, train_data, eval_date, item_set, item_deg, idx2user, args, valid_array_):
    res = {}
    answers = {}
    [user, user_array, seqs_array, label_array] = valid_array_
    # eval_date = generate_vail_date(train_data, eval_date, 256)
    
    for u, seq,label in tqdm(gen(user, user_array, seqs_array, label_array, 32)):
        preds = model.predict(sess, u, seq, item_set)
        arg_sort =np.argsort(preds, -1)[:, ::-1]
        for i in range(len(u)):
            user_idx = u[i][0]
            label_item = label[i][0]
            # user = idx2user[user_idx]
            phase = '4'
            res.setdefault(phase, {})
            answers.setdefault(phase, {})
            _pred_top_50 = item_set[arg_sort[i][:50]]
            res[phase][user_idx] = _pred_top_50.tolist()
            answers[phase][user_idx] = (label_item, item_deg[label_item])
    finally_score, phase_score = evalation(res, answers, None)
    return finally_score, phase_score
 
def evaluate_each_phase(predictions, answers, recall_num=50):
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        rank = 0
        while rank < recall_num and predictions[user_id][rank] != item_id:
            rank += 1
        num_cases_full += 1.0
        if rank < recall_num:
            ndcg_50_full += 1.0 / np.log2(rank + 2.0)
            hitrate_50_full += 1.0
        if item_degree <= median_item_degree:
            num_cases_half += 1.0
            if rank < recall_num:
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([ndcg_50_full, ndcg_50_half,
                     hitrate_50_full, hitrate_50_half], dtype=np.float32)


def evalation(res, answers, item_deg=None, recall_num=50):
    if item_deg is not None:
        _  = {}
        for phase in answers.keys():
            _.setdefault(phase, {})
            for k,v in answers[phase].items():
                _[phase][k] = (v, item_deg[v])
        answers = _
    finally_score = np.zeros(4, dtype=np.float32)
    phase_score = {}
    for phase in res.keys():
    # We sum the scores from all the phases, instead of averaging them.
        score = evaluate_each_phase(res[phase], answers[phase], recall_num)
        print(f"phase: {phase},  hitrate_full:{score[2]}, ndcg_full:{score[0]}, hitrate_half:{score[3]}, ndcg_half:{score[1]}")
        finally_score += score
        phase_score[phase] = str(score.tolist())
    print(f"phase: all,  hitrate_full:{finally_score[2]}, ndcg_full:{finally_score[0]}, hitrate_half:{finally_score[3]}, ndcg_half:{finally_score[1]}")
    return finally_score, phase_score


def generate_vail_date(train, valid, id2user, user2idmap2):
    user = []
    seqs = []
    labels = []
    for user_idx, label_item in tqdm(valid.items(), leave=False, total=len(valid), desc="[EVAL] >> "):
        if len(label_item) < 1:
            continue
        seq = train[user_idx]
        seq_len = len(seq)
        if seq_len == 0:
            continue
        if seq_len <= args.maxlen:
            seq_ = [0] * (args.maxlen - seq_len) + seq
        else:
            seq_ = seq[-50:]
        seqs.append(seq_)
        
        u = id2user[user_idx]
        # u = user2idmap2[u.split('_')[-1]]
        u = user2idmap2[u[2:]]
        
        user.append([u])
        
        labels.append(label_item)
    user_array = np.array(user)
    seqs_array = np.array(seqs)
    label_array = np.array(labels)
    return user, user_array, seqs_array, label_array


def gen(user, user_array, seqs_array, label_array, batch_size):
    for i in range(len(user)//batch_size):
        yield (user_array[i*batch_size:(i+1)*batch_size], seqs_array[i*batch_size:(i+1)*batch_size], label_array[i*batch_size:(i+1)*batch_size])
    yield (user_array[(i+1)*batch_size:], seqs_array[(i+1)*batch_size:], label_array[(i+1)*batch_size:])


class Args:
    lr = 0.002
    maxlen = 50
    hidden_units = 256
    num_blocks = 1
    dropout_rate = 0.5
    num_heads = 2
    l2_emb = 0.0


if __name__ == "__main__":
    now_phase = 9

    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", type=int, default=0)
    parser.add_argument("--train", type=int, default=0)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--valid", type=int, default=0)
    
    args = parser.parse_args()
    
    kind = int(args.kind)
    if kind == 1:
        read_path = config.online_item_file
        save_path = config.save_path_online
        model_base_path = 'ckpt'
        train_path = config.train_path
        test_path = config.test_path
    elif kind == 2:
        read_path = config.offline_item_file
        save_path = config.save_path_offline
        model_base_path = 'ckpt2'   
        train_path = config.offline_path
        test_path = config.offline_path
    elif kind == 3:
        read_path = config.model1_item_file
        save_path = config.save_path_model1
        model_base_path = 'ckpt3'
        train_path = config.model1_path
        test_path = config.model1_path
    
    train, valid, n_items, n_users, id2item, id2user, \
        item_deg, emb, use_test, user2idmap2, num_users2 = get_data(now_phase, train_path,
                                                                    test_path, kind) # , base_path='F:\data_kdd',
    emb = np.concatenate((np.zeros((1,256)), emb), axis=0) / 25    
    usr_emb = 0
    
    print('Reading data done.')
    train_flag = args.train 
    valid_flag = args.valid
    test_flag = args.test
    test_flag2 = 0
    
    num_neg = 20
    batch_size = 256
    args = Args()
    num_batch = len(train) // batch_size
    num_epochs = 75
    item_set = np.arange(1, n_items+1)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    print(n_items)
    sampler = WarpSampler(train, n_users, n_items, id2user, user2idmap2, num_neg=num_neg, 
                          batch_size=batch_size, maxlen=args.maxlen, n_workers=3)
    
    model = Model(num_users2, n_items, args, emb, num_neg, dec_step=num_batch*25,
                  emb_usr=usr_emb)
    
    sess.run(tf.initialize_all_variables())
    sess.run(tf.assign(model.item_emb_table, model.emb_item))
    # sess.run(tf.assign(model.user_emb_table, model.usr_emb))
    
    user, user_array, seqs_array, label_array = generate_vail_date(train, valid, id2user, user2idmap2)
    valid_array = [user, user_array, seqs_array, label_array]
    idx = np.random.choice(len(user), 5000, replace=False)
    user2, user_array2, seqs_array2, label_array2= [], [], [], []
    
    for i in range(len(idx)):
        user2.append(user[idx[i]])
        user_array2.append(user_array[idx[i]])
        seqs_array2.append(seqs_array[idx[i]])
        label_array2.append(label_array[idx[i]])
    valid_array2 = [user2, user_array2, seqs_array2, label_array2]
    
    
    saver = tf.train.Saver()
    ckpt_path = os.path.join(model_base_path, 'model.ckpt')
    
    if not os.path.exists(model_base_path):
        os.mkdir(model_base_path)
#    saver.restore(sess, ckpt_path)
    
    if train_flag:
        finally_score = [0]
        best_score = 0
        for epoch in range(1, num_epochs + 1):
#            auc_ = []
            loss_ = []
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()
                loss, _ = sess.run([model.loss, model.train_op],
                                          {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                          model.is_training: True})
#                auc_.append(auc)
                loss_.append(loss)
            print('epoch:%d, loss:%.3f' %(epoch, np.mean(loss_)))
            if epoch % 25 == 0:
                print("[EVAL] valid...")
                finally_score, phase_score = eval_model(model, sess, train, valid, 
                                                        item_set, item_deg, id2user, args, valid_array2)
                
                if finally_score[0] > best_score:
                    best_score = finally_score[0]
                save_path = saver.save(sess, ckpt_path)
                
        ckpt_path = os.path.join(model_base_path, 'model_last.ckpt')
        save_path = saver.save(sess, ckpt_path)

    sampler.close()
    print("Done!")
    
    
    # sess.run(tf.initialize_all_variables())
    
    if valid_flag:
        ckpt_path = os.path.join(model_base_path, 'model.ckpt')
        saver.restore(sess, ckpt_path)
        finally_score, phase_score = eval_model(model, sess, train, valid, 
                                                item_set, item_deg, id2user, args, valid_array)
    if test_flag2:
        ckpt_path = os.path.join(model_base_path, 'model_last.ckpt')
        saver.restore(sess, ckpt_path)
        evaluate2(model, [use_test, n_users, n_items], user2idmap2, 
                  args, sess, id2item, id2user)
        from evaulation import evaluate_
        evaluate_('pred_valid.csv', answer_fname='model_1/model_1_debias_track_answer.csv')
												
    if test_flag:
        # resotre model
        ckpt_path = os.path.join(model_base_path, 'model_last.ckpt')
        saver.restore(sess, ckpt_path)
        evaluate5(model, [use_test, n_users, n_items], user2idmap2, 
                  args, sess, id2item, id2user, save_path=save_path, read_path=read_path)
        
