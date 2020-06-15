import numpy as np
from multiprocessing import Process, Queue
import random

def random_neq(l, r, s, num_neg):
    negs = []
    for i in range(num_neg):
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        negs.append(t)
            
    return negs


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, num_neg, 
                    id2user, user2idmap2, result_queue, SEED):
    def sample():
        
#        num_neg = 2
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen, num_neg], dtype=np.int32)
#        nxt = user_train[user][-1]
        idx = maxlen - 1
        
        seq_ = user_train[user]
        st = 0
        if len(seq_) > (maxlen+1) :
            st = np.random.randint(0, len(seq_)-maxlen-1)
        seq_ = seq_[st:st+(maxlen+1)]
        nxt = seq_[-1]
        # nexts = [nxt]
        ts = set(seq_)
        
        for i in reversed(seq_[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx, :] = random_neq(1, itemnum + 1, ts, num_neg)
            nxt = i
            # nexts.append(i)
            # nxt = random.choice(nexts)
            
            idx -= 1
            if idx == -1: break
        
        user = id2user[user]
        # user = user2idmap2[int(user.split('_')[-1])]
        user = user2idmap2[user[2:]]
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, id2user, user2idmap2,
                 num_neg=20, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      num_neg,
                                                      id2user, user2idmap2,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
