import numpy as np
from collections import defaultdict
import pickle
import copy
import sys
import os
import torch

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-x', default='Books', help='category')
parser.add_argument('-y', default='Movies_and_TV', help='category')
parser.add_argument('-z', default='CDs_and_Vinyl', help='category')
args = parser.parse_args()

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x = pickle.load(f)
    except:
        x = []
    return x

def data_partition(fname):
    output_path = "../unirec_datasets"

    User = defaultdict(list)

    train_data = defaultdict(list)
    valid_data = defaultdict(list)
    test_data = defaultdict(list)

    user_ids = list()
    item_ids1 = list()

    # train data
    with open('../review_datasets/%s/%s_train.txt' % (fname, fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)-1
            i = int(i)
            user_ids.append(u)
            item_ids1.append(i)
            User[u].append(i)

    with open('../review_datasets/%s/%s_valid.txt' % (fname, fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)-1
            i = int(i)
            user_ids.append(u)
            item_ids1.append(i)
            User[u].append(i)

    with open('../review_datasets/%s/%s_test.txt' % (fname, fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)-1
            i = int(i)
            user_ids.append(u)
            item_ids1.append(i)
            User[u].append(i)

    item_map = dict()
    itemnum1 = 0
    ItemMeanFeatures = load_data('../review_datasets/%s/%s_review_emb_mean.dat' % (fname, fname))
    MetaMeanFeatures = load_data('../review_datasets/%s/%s_meta_emb.dat' % (fname, fname))

    ItemMeanFeatures1 = {}
    MetaMeanFeatures1 = {}
    for i in item_ids1:
        if i not in item_map:
            item_map[i] = itemnum1
            ItemMeanFeatures1[item_map[i]] = ItemMeanFeatures[i]
            MetaMeanFeatures1[item_map[i]] = MetaMeanFeatures[i]
            itemnum1 += 1

    uid_list = list(User.keys())
    uid_list.sort(key=lambda t: int(t))

    item_list = list(range(itemnum1))

    embeddings = []

    for i in item_list:
        embeddings.append(ItemMeanFeatures1[i]+MetaMeanFeatures1[i])
    #
    embeddings = torch.tensor(embeddings,dtype=torch.float32)

    # embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)
    suffix = '1'
    emb_type = 'Mean'
    file = os.path.join(output_path, fname,
                        fname + '.pth')
    # embeddings.tofile(file)
    torch.save(embeddings, file)
    print("text success")

    User1 = defaultdict(list)
    for user in uid_list:
        for item in User[user]:
            i = item_map[item]
            User1[user].append(i)

    for user in User1:
        nfeedback = len(User1[user])
        if nfeedback < 3:
            train_data[user] = User1[user]
            valid_data[user] = []
            test_data[user] = []
        else:
            train_data[user] = User1[user][:-2]
            valid_data[user] = []
            valid_data[user].append(User1[user][-2])
            test_data[user] = []
            test_data[user].append(User1[user][-1])

    with open(os.path.join(output_path, fname, f'{fname}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                seq = [str(item) for item in seq]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(os.path.join(output_path, fname, f'{fname}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            if uid in valid_data and valid_data[uid]:  # 检查 uid 是否存在于 valid_data 字典中，以及 valid_data[uid] 列表是否为空
                target_item = valid_data[uid][0]
                item_seq = [str(item) for item in item_seq]
                file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')
            else:
                print(f"Warning: No validation data for user {uid}")

    with open(os.path.join(output_path, fname, f'{fname}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            if uid in test_data and test_data[uid]:  # 检查 uid 是否存在于 test_data 字典中，以及 test_data[uid] 列表是否为空
                target_item = test_data[uid][0]
                item_seq = [str(item) for item in item_seq]
                file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')
            else:
                print(f"Warning: No test data for user {uid}")

    print('success')


if __name__ == '__main__':
    data_partition(args.x)
    data_partition(args.y)
    data_partition(args.z)