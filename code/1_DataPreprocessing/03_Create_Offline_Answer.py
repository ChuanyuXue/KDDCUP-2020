#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict

current_phases = 9

def _create_answer_file_for_evaluation(answer_fname='debias_track_answer.csv'):
    train = './user_data/offline/offline_train_click-%d.csv'
    test = './user_data/offline/offline_test_click-%d.csv'

    
#     train = 'model'+str(number)+'/model_'+str(number)+'_train_click-%d.csv'
#     test = 'model'+str(number)+'/model_'+str(number)+'_test_click-%d.csv'

    
    # underexpose_test_qtime-T.csv contains only <user_id, item_id>
    # underexpose_test_qtime_with_answer-T.csv contains <user_id, item_id, time>
    #answer = 'model/model_test_qtime-%d.csv'  # not released
    
    answer = './user_data/offline/offline_test_qtime-%d.csv'

#     answer = 'model'+str(number)+'/model_'+str(number)+'_test_qtime-%d.csv'

    item_deg = defaultdict(lambda: 0)
    with open(answer_fname, 'w') as fout:
        for phase_id in range(current_phases+1):
            with open(train % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(test % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1
            with open(answer % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    assert user_id % 11 == phase_id
                    print(phase_id, user_id, item_id, item_deg[item_id],
                          sep=',', file=fout)


# In[2]:


_create_answer_file_for_evaluation('./user_data/offline/offline_debias_track_answer.csv')


# In[ ]:





# In[3]:


# _create_answer_file_for_evaluation('model'+str(number)+'/model_'+str(number)+'_debias_track_answer.csv')

