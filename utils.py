import numpy as np
from collections import defaultdict
import pickle
from multiprocessing import Queue, Process
import os
import sys
import pandas as pd


def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x = pickle.load(f)
    except:
        x = []
    return x


def data_partition(target_domain_fname, source_domain_fname):
    usernum = 0
    itemnum_target_domain = 0
    itemnum_source_domain = 0

    User_target_domain = defaultdict(list)
    User_source_domain = defaultdict(list)

    user_train_target_domain = defaultdict(list)
    user_valid_target_domain = defaultdict(list)
    user_test_target_domain = defaultdict(list)

    user_train_source_domain = defaultdict(list)
    user_valid_source_domain = defaultdict(list)
    user_test_source_domain = defaultdict(list)

    user_map = dict()
    item_map = dict()

    user_ids = list()
    item_ids_target_domain = list()
    item_ids_source_domain = list()

    id2asin = dict()
    idmap2asin = dict()

    # ========================================================
    # target domain
    # train data
    User_target = defaultdict(list)
    Time_target = defaultdict(list)
    with open('./review_datasets/%s/%s_train.txt' % (target_domain_fname, target_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids_target_domain.append(i)
            User_target[u].append(i)
            Time_target[u].append(t)

    # valid data
    with open('./review_datasets/%s/%s_valid.txt' % (target_domain_fname, target_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids_target_domain.append(i)
            User_target[u].append(i)
            Time_target[u].append(t)

    # test data
    with open('./review_datasets/%s/%s_test.txt' % (target_domain_fname, target_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids_target_domain.append(i)
            User_target[u].append(i)
            Time_target[u].append(t)

    # load item review and meta embedding
    ItemMeanFeatures = load_data('./review_datasets/%s/%s_review_emb_mean.dat' % (target_domain_fname, target_domain_fname))
    MetaMeanFeatures = load_data('./review_datasets/%s/%s_meta_emb.dat' % (target_domain_fname, target_domain_fname))

    ItemMeanFeatures_emb = {}
    MetaMeanFeatures_emb = {}

    print(f"Index i: {i}, Length of ItemMeanFeatures: {len(ItemMeanFeatures)}")

    for i in item_ids_target_domain:
        if i not in item_map:
            item_map[i] = itemnum_target_domain + 1
            idmap2asin[item_map[i]] = id2asin[i]
            ItemMeanFeatures_emb[item_map[i]] = ItemMeanFeatures[i]
            MetaMeanFeatures_emb[item_map[i]] = MetaMeanFeatures[i]
            itemnum_target_domain += 1

    # leave one out
    for user in user_ids:
        if user not in user_map:
            user_map[user] = usernum + 1
            usernum += 1

            u = user_map[user]
            for item in User_target[user]:
                i = item_map[item]
                User_target_domain[u].append(i)

            nfeedback = len(User_target_domain[u])
            if nfeedback < 3:
                user_train_target_domain[u] = User_target_domain[u]
                user_valid_target_domain[u] = []
                user_test_target_domain[u] = []
            else:
                user_train_target_domain[u] = User_target_domain[u][:-2]
                user_valid_target_domain[u] = []
                user_valid_target_domain[u].append(User_target_domain[u][-2])
                user_test_target_domain[u] = []
                user_test_target_domain[u].append(User_target_domain[u][-1])

    # ========================================================
    # source domain
    User_source = defaultdict(list)
    Time_source = defaultdict(list)
    with open('./review_datasets/%s/%s_train.txt' % (source_domain_fname, source_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            item_ids_source_domain.append(i)
            User_source[u].append(i)
            Time_source[u].append(t)

    with open('./review_datasets/%s/%s_valid.txt' % (source_domain_fname, source_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            item_ids_source_domain.append(i)
            User_source[u].append(i)
            Time_source[u].append(t)

    with open('./review_datasets/%s/%s_test.txt' % (source_domain_fname, source_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            item_ids_source_domain.append(i)
            User_source[u].append(i)
            Time_source[u].append(t)

    ItemMeanFeatures = load_data('./review_datasets/%s/%s_review_emb_mean.dat' % (source_domain_fname, source_domain_fname))
    MetaMeanFeatures = load_data('./review_datasets/%s/%s_meta_emb.dat' % (source_domain_fname, source_domain_fname))

    for i in item_ids_source_domain:
        if i not in item_map:
            item_map[i] = itemnum_target_domain + itemnum_source_domain + 1
            idmap2asin[item_map[i]] = id2asin[i]
            ItemMeanFeatures_emb[item_map[i]] = ItemMeanFeatures[i]
            MetaMeanFeatures_emb[item_map[i]] = MetaMeanFeatures[i]

            itemnum_source_domain += 1

    for user in User_source:
        u = user_map[user]
        for item in User_source[user]:
            i = item_map[item]
            User_source_domain[u].append(i)

        index1 = max([idx for idx, t in enumerate(Time_source[user]) if t <= Time_target[user][-2]], default=0)
        user_train_source_domain[u] = User_source_domain[u][:index1+1]
        index2 = max([idx for idx, t in enumerate(Time_source[user]) if t <= Time_target[user][-1]], default=0)
        user_valid_source_domain[u] = User_source_domain[u][:index2+1]

    # text information
    ItemFeatures = []
    MetaFeatures = []
    ItemFeatures.append(np.zeros(np.array(ItemMeanFeatures_emb[1].shape[0])))
    MetaFeatures.append(np.zeros(np.array(ItemMeanFeatures_emb[1].shape[0])))
    for item in range(1, itemnum_target_domain + itemnum_source_domain + 1):
        ItemFeatures.append(np.array(ItemMeanFeatures_emb[item]))
        MetaFeatures.append(np.array(MetaMeanFeatures_emb[item]))

    ItemFeatures = np.array(ItemFeatures)
    MetaFeatures = np.array(MetaFeatures)

    # ========================================================
    # neg
    user_neg_target_domain = defaultdict(list)
    with open("./review_datasets/%s/%s_negative.csv" % (target_domain_fname, target_domain_fname), 'r') as f:
        for line in f:
            l = line.rstrip().split(',')
            u = user_map[int(l[0])]
            for j in range(1, 101):
                i = item_map[int(l[j])]
                user_neg_target_domain[u].append(i)

    is_saved = False
    if is_saved:
        data_directory = './review_datasets'
        df = pd.DataFrame(list(idmap2asin.items()), columns=['id', 'asin'])
        df = df.loc[:, ['id', 'asin']]
        df.to_csv(os.path.join(data_directory, '%s2%s_id2asin.txt' % (source_domain_fname, target_domain_fname)), sep=' ', index=False,
                  header=False)
        print("success write")

    return [user_train_target_domain, user_valid_target_domain, user_test_target_domain,
            user_train_source_domain, user_valid_source_domain, user_test_source_domain,
            usernum, itemnum_target_domain, itemnum_source_domain, user_neg_target_domain,
            ItemFeatures, MetaFeatures]


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train_target_domain, user_train_source_domain, usernum, itemnum,
                    batch_size, maxlen, result_queue, SEED):

    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train_target_domain[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        # seq
        seq_target = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train_target_domain[user][-1]
        idx = maxlen - 1

        ts = set(user_train_target_domain[user])
        for i in reversed(user_train_target_domain[user][:-1]):
            seq_target[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg_i = random_neq(1, itemnum + 1, ts)
                neg[idx] = neg_i
            nxt = i
            idx -= 1
            if idx == -1:
                break

        seq_source = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train_source_domain[user][:-1]):
            seq_source[idx] = i
            idx -= 1
            if idx == -1:
                break

        return (np.ones(maxlen) * user, seq_target, seq_source, pos,  neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, user_train_target_domain, user_train_source_domain,
                 usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):

        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function,
                        args=(user_train_target_domain, user_train_source_domain, usernum, itemnum,
                              batch_size, maxlen, self.result_queue, np.random.randint(2e9))))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def evaluate_valid(model, dataset, args):
    [user_train_target_domain, user_valid_target_domain, user_test_target_domain,
     user_train_source_domain, user_valid_source_domain, user_test_source_domain,
     usernum, itemnum_target_domain, itemnum_source_domain, user_neg_target_domain,
     ItemMeanFeatures, MetaFeatures] = dataset

    NDCG10 = 0.0
    HT10 = 0.0
    NDCG5 = 0.0
    HT5 = 0.0
    valid_user = 0.0

    users = range(1, usernum + 1)
    for u in users:
        if len(user_train_target_domain[u]) < 1 or len(user_valid_target_domain[u]) < 1:
            continue  # remove the seq only on the left

        seq_t = np.zeros([args.maxlen], dtype=np.int32)
        seq_s = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1
        for i in reversed(user_train_target_domain[u]):
            seq_t[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        if u in user_train_source_domain:
            for i in reversed(user_train_source_domain[u]):
                seq_s[idx] = i
                idx -= 1
                if idx == -1: break

        item_idx = [user_valid_target_domain[u][0]]
        for t in user_neg_target_domain[u]:
            item_idx.append(t)

        predictions = model.predict(*[np.array(l) for l in [[seq_t], [seq_s], item_idx]])
        predictions = -predictions[0]
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


def evaluate_test(model, dataset, args):
    [user_train_target_domain, user_valid_target_domain, user_test_target_domain,
     user_train_source_domain, user_valid_source_domain, user_test_source_domain,
     usernum, itemnum_target_domain, itemnum_source_domain, user_neg_target_domain,
     ItemMeanFeatures, MetaFeatures] = dataset

    NDCG10 = 0.0
    HT10 = 0.0

    NDCG5 = 0.0
    HT5 = 0.0

    test_user = 0.0

    users = range(1, usernum + 1)
    for u in users:
        if len(user_train_target_domain[u]) < 1 or len(user_test_target_domain[u]) < 1:
            continue  # remove the seq only on the left

        seq_t = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        valid_item = user_valid_target_domain[u][0]
        seq_t[idx] = valid_item
        idx -= 1
        for i in reversed(user_train_target_domain[u]):
            seq_t[idx] = i
            idx -= 1
            if idx == -1: break

        seq_s = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        if u in user_valid_source_domain:
            for i in reversed(user_valid_source_domain[u]):
                seq_s[idx] = i
                idx -= 1
                if idx == -1: break

        item_idx = [user_test_target_domain[u][0]]
        for t in user_neg_target_domain[u]:
            item_idx.append(t)

        predictions = model.predict(*[np.array(l) for l in [[seq_t], [seq_s], item_idx]])
        predictions = -predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        test_user += 1
        if rank < 10:
            NDCG10 += 1 / np.log2(rank + 2)
            HT10 += 1

        if rank < 5:
            NDCG5 += 1 / np.log2(rank + 2)
            HT5 += 1

        if test_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # print(test_user)
    return NDCG10 / test_user, HT10 / test_user, NDCG5 / test_user, HT5 / test_user
