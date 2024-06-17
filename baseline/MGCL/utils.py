import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, user_train2, time1, time2, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        seq2 = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        t1 = np.zeros([maxlen], dtype=np.int32)  #
        t2 = np.zeros([maxlen], dtype=np.int32)  #
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i, t in reversed(list(zip(user_train[user][:-1], time1[user][:-1]))):
            seq[idx] = i
            t1[idx] = t
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i, t in reversed(list(zip(user_train2[user][:-1], time2[user][:-1]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask = np.zeros([maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask[idx] = idx2

        return (user, seq, pos, neg, seq2, mask)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, User2, time1, time2, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      User2,
                                                      time1,
                                                      time2,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
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


def data_partition(fname, fname2):
    usernum = 0
    itemnum1 = 0
    User = defaultdict(list)
    User1 = defaultdict(list)
    User2 = defaultdict(list)
    user_train1 = {}
    user_valid1 = {}
    user_test1 = {}
    neglist1 = defaultdict(list)
    user_neg1 = {}

    itemnum2 = 0
    user_train2 = {}
    user_valid2 = {}
    user_test2 = {}
    neglist2 = defaultdict(list)
    user_neg2 = {}

    user_map = dict()
    item_map = dict()

    user_ids = list()
    item_ids1 = list()
    item_ids2 = list()

    Time = defaultdict(list)
    Time1 = {}
    Time2 = {}

    f = open('cross_data/processed_data_all/%s_train.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids1.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_valid.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids1.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_test.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids1.append(i)
        User[u].append(i)
        Time[u].append(t)

    for u in user_ids:
        if u not in user_map:
            user_map[u] = usernum + 1
            usernum += 1
    for i in item_ids1:
        if i not in item_map:
            item_map[i] = itemnum1 + 1
            itemnum1 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User1[u].append(i)
        Time1[u] = Time[user]

    User = defaultdict(list)
    Time = defaultdict(list)

    f = open('cross_data/processed_data_all/%s_train.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids2.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_valid.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids2.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('cross_data/processed_data_all/%s_test.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids2.append(i)
        User[u].append(i)
        Time[u].append(t)

    for i in item_ids2:
        if i not in item_map:
            item_map[i] = itemnum1 + itemnum2 + 1
            itemnum2 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User2[u].append(i)
        Time2[u] = Time[user]

    f = open('cross_data/processed_data_all/%s_negative.csv' % fname, 'r')
    for line in f:
        l = line.rstrip().split(',')
        u = user_map[int(l[0])]
        for j in range(1, 101):
            i = item_map[int(l[j])]
            neglist1[u].append(i)

    for user in User1:
        nfeedback = len(User1[user])
        if nfeedback < 3:
            user_train1[user] = User1[user]
            user_valid1[user] = []
            user_test1[user] = []
        else:
            user_train1[user] = User1[user][:-2]
            user_valid1[user] = []
            user_valid1[user].append(User1[user][-2])
            user_test1[user] = []
            user_test1[user].append(User1[user][-1])
        user_neg1[user] = neglist1[user]

    for user in User2:
        nfeedback = len(User2[user])
        if nfeedback < 3:
            user_train2[user] = User2[user]
            user_valid2[user] = []
            user_test2[user] = []
        else:
            user_train2[user] = User2[user][:-2]
            user_valid2[user] = []
            user_valid2[user].append(User2[user][-2])
            user_test2[user] = []
            user_test2[user].append(User2[user][-1])

    return [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1, user_train2, user_valid2, user_test2,
            itemnum2, Time1, Time2]


def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2, itemnum2, time1,
     time2] = copy.deepcopy(dataset)

    NDCG10 = 0.0
    HT10 = 0.0
    valid_user = 0.0

    NDCG5 = 0.0
    HT5 = 0.0

    users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)  #
        t2 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i, t in reversed(list(zip(train[u], time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(user_train2[u], time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask = np.zeros([args.maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < args.maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask[idx] = idx2

        for i in neg[u]:
            item_idx.append(i)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], item_idx, [mask]]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG10 += 1 / np.log2(rank + 2)
            HT10 += 1
        if rank < 5:
            NDCG5 += 1 / np.log2(rank + 2)
            HT5 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG10 / valid_user, HT10 / valid_user, NDCG5 / valid_user, HT5 / valid_user


def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2, itemnum2, time1,
     time2] = copy.deepcopy(dataset)

    NDCG10 = 0.0
    HT10 = 0.0
    valid_user = 0.0
    NDCG5 = 0.0
    HT5 = 0.0


    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)  #
        t2 = np.zeros([args.maxlen], dtype=np.int32)  #
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(train[u], time1[u]))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        for i, t in reversed(list(zip(user_train2[u], time2[u]))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1: break

        mask = np.zeros([args.maxlen], dtype=np.int32)  #
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < args.maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask[idx] = idx2

        for i in neg[u]:
            item_idx.append(i)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq2], item_idx, [mask]]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG10 += 1 / np.log2(rank + 2)
            HT10 += 1
        if rank < 5:
            NDCG5 += 1 / np.log2(rank + 2)
            HT5 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG10 / valid_user, HT10 / valid_user, NDCG5 / valid_user, HT5 / valid_user

def get_slice(inputs):
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        #max_n_node = np.max(n_node)
        max_n_node = 100
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))   
            for i in np.arange(len(u_input) - 1):
                if u_input[i+1] == 0:
                    continue
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items

