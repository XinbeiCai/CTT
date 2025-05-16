from collections import defaultdict
import pickle
import numpy as np


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
    # User = defaultdict(list)
    Time_target = defaultdict(list)

    train_data = []
    valid_data = []
    test_data = []

    # source domain
    User_source = defaultdict(list)
    Time_source = defaultdict(list)


    user_seq = defaultdict(list)

    with open('../../review_datasets/%s/%s_train.txt' % (source_domain_fname, source_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids_source_domain.append(i)
            User_source[u].append(i)
            Time_source[u].append(t)
            user_seq[u].append((i, t))

    with open('../../review_datasets/%s/%s_valid.txt' % (source_domain_fname, source_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids_source_domain.append(i)
            User_source[u].append(i)
            Time_source[u].append(t)
            user_seq[u].append((i, t))

    with open('../../review_datasets/%s/%s_test.txt' % (source_domain_fname, source_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids_source_domain.append(i)
            User_source[u].append(i)
            Time_source[u].append(t)
            user_seq[u].append((i, t))

    ItemMeanFeatures = load_data('../../review_datasets/%s/%s_review_emb_mean.dat' % (source_domain_fname, source_domain_fname))
    MetaMeanFeatures = load_data('../../review_datasets/%s/%s_meta_emb.dat' % (source_domain_fname, source_domain_fname))
    ItemMeanFeatures_emb = {}
    MetaMeanFeatures_emb = {}

    print(len(ItemMeanFeatures))

    for i in item_ids_source_domain:
        if i not in item_map:
            item_map[i] = itemnum_source_domain
            idmap2asin[item_map[i]] = id2asin[i]

            ItemMeanFeatures_emb[item_map[i]] = ItemMeanFeatures[i]
            # try:
            #     ItemMeanFeatures_emb[item_map[i]] = ItemMeanFeatures[i]
            # except Exception as e:
            #     print(f"[ERROR] i={i}, len(ItemMeanFeatures)={len(ItemMeanFeatures)}, len(item_map)={len(item_map)}")
            #     print(f"item_map[i] error -> item_map[{i}]: {item_map[i] if i < len(item_map) else 'N/A'}")
            #     raise e
            MetaMeanFeatures_emb[item_map[i]] = MetaMeanFeatures[i]

            itemnum_source_domain += 1



    User_target = defaultdict(list)
    with open('../../review_datasets/%s/%s_train.txt' % (target_domain_fname, target_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids_target_domain.append(i)
            User_target[u].append(i)
            Time_target[u].append(t)
            user_seq[u].append((i, t))

    # valid data
    with open('../../review_datasets/%s/%s_valid.txt' % (target_domain_fname, target_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids_target_domain.append(i)
            User_target[u].append(i)
            Time_target[u].append(t)
            user_seq[u].append((i, t))

    # test data
    with open('../../review_datasets/%s/%s_test.txt' % (target_domain_fname, target_domain_fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids_target_domain.append(i)
            User_target[u].append(i)
            Time_target[u].append(t)
            user_seq[u].append((i, t))


    ItemMeanFeatures = load_data('../../review_datasets/%s/%s_review_emb_mean.dat' % (target_domain_fname, target_domain_fname))
    MetaMeanFeatures = load_data('../../review_datasets/%s/%s_meta_emb.dat' % (target_domain_fname, target_domain_fname))

    for i in item_ids_target_domain:
        if i not in item_map:
            item_map[i] = itemnum_target_domain + itemnum_source_domain
            idmap2asin[item_map[i]] = id2asin[i]
            ItemMeanFeatures_emb[item_map[i]] = ItemMeanFeatures[i]
            MetaMeanFeatures_emb[item_map[i]] = MetaMeanFeatures[i]
            itemnum_target_domain += 1

    def takeSecond(elem):
        return elem[1]

    # leave one out
    data_seq = defaultdict(list)
    for user in user_ids:
        if user not in user_map:
            user_map[user] = usernum + 1
            usernum += 1
            u = user_map[user]
            user_seq[user].sort(key=takeSecond)
            for item, time in user_seq[user]:
                i = item_map[item]
                data_seq[u].append(i)

            nfeedback = len(data_seq[u])
            if nfeedback > 3:
                train_data.append(data_seq[u][:-2])
                if data_seq[u][-2] < itemnum_source_domain:
                    valid_data.append([data_seq[u][1:-2], 0, data_seq[u][-2]])
                else:
                    valid_data.append([data_seq[u][1:-2], 1, data_seq[u][-2]])
                if data_seq[u][-1] < itemnum_source_domain:
                    test_data.append([data_seq[u][2:-1], 0, data_seq[u][-1]])
                else:
                    test_data.append([data_seq[u][2:-1], 1, data_seq[u][-1]])

    # text information
    ItemFeatures = []
    MetaFeatures = []
    for item in range(0, itemnum_target_domain + itemnum_source_domain):
        ItemFeatures.append(np.array(ItemMeanFeatures_emb[item]))
        MetaFeatures.append(np.array(MetaMeanFeatures_emb[item]))
    ItemFeatures.append(np.zeros(np.array(ItemMeanFeatures_emb[1].shape[0])))
    MetaFeatures.append(np.zeros(np.array(ItemMeanFeatures_emb[1].shape[0])))
    ItemFeatures = np.array(ItemFeatures)
    MetaFeatures = np.array(MetaFeatures)

    # target_neg_sample = []
    # with open("../../review_datasets/%s/%s_negative.csv" % (target_domain_fname, target_domain_fname), 'r') as f:
    #     for line in f:
    #         l = line.rstrip().split(',')
    #         for j in range(1, 101):
    #             i = item_map[int(l[j])] - itemnum_source_domain
    #             target_neg_sample.append(i)
    #         break
    #
    # source_neg_sample = []
    # with open("../../review_datasets/%s/%s_negative.csv" % (source_domain_fname, source_domain_fname), 'r') as f:
    #     for line in f:
    #         l = line.rstrip().split(',')
    #         for j in range(1, 101):
    #             i = item_map[int(l[j])]
    #             source_neg_sample.append(i)
    #         break




    return itemnum_target_domain, itemnum_source_domain, train_data, valid_data, test_data, ItemFeatures, MetaFeatures