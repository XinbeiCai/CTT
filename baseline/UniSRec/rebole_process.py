import os
import pickle
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import argparse
import numpy as np
from collections import defaultdict

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x = pickle.load(f)
    except:
        x = []
    return x

parser = argparse.ArgumentParser()
parser.add_argument('-x', default='Books', help='category')
parser.add_argument('-y', default='Movies_and_TV', help='category')
parser.add_argument('-z', default='CDs_and_Vinyl', help='category')

args = parser.parse_args()

def get_emb(category_name):
    # read_data
    batch_size = 5
    data_directory = './review_datasets/%s/' % category_name
    data_path = os.path.join(data_directory, '%s_review.dat' % category_name)
    df = load_data(data_path)
    df.reset_index(inplace=True)

    # bert_train_embedding
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-cased', use_fast=True)
    model = AutoModel.from_pretrained('./bert-base-cased')
    model.to(device)

    # ========================================================
    # meta
    data_directory = './review_datasets/%s/' % category_name
    data_path = os.path.join(data_directory, '%s_meta.dat' % category_name)

    train = load_data('./review_datasets/%s/%s_train.dat' % (category_name, category_name))
    df = load_data(data_path)

    df_merge = pd.merge(train, df, how='inner', on=['item_id'])
    df_merge = df_merge.drop_duplicates(subset='item_id')

    output_path = "./unirec_datasets/%s/" % category_name
    file = os.path.join(output_path, f'{category_name}.text')
    with open(file, 'w') as fp:
        fp.write('item_id:token\ttext:token_seq\n')
        for index, row in df_merge.iterrows():
            item = row['asin']
            text = row['information']
            fp.write(item + '\t' + text + '\n')

    print('Writing text file done')

    df.reset_index(inplace=True)
    information_embedding = []
    batch_num = int(df.shape[0] / batch_size)
    for i in tqdm(range(batch_num), total=int(batch_num), ncols=70, leave=False, unit='b'):
        if i == batch_num - 1:
            batch = df.loc[batch_size * i:, 'information'].tolist()
        else:
            batch = df.loc[batch_size * i:batch_size * (i + 1) - 1, 'information'].tolist()

        input = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
        input = input.to(device)

        outputs = model(**input)  # shape (batch_size, max_length, 768)
        v = torch.mean(outputs.last_hidden_state, dim=1)  # shape (batch_size, 768)
        information_embedding.extend(v.tolist())

    # processing_data
    df['information_embedding'] = information_embedding
    df_information = df.drop(['information'], axis=1)

    InformationInfo = {}
    for index, row in df_information.iterrows():
        itemid = int(row['item_id'])
        information_embedding = row['information_embedding']
        InformationInfo[itemid] = information_embedding

    print('meta_%s_done!' % category_name)

    return InformationInfo


def data_partition(fname, MetaMeanFeatures):
    output_path = "./unirec_datasets"

    User = defaultdict(list)

    train_data = defaultdict(list)
    valid_data = defaultdict(list)
    test_data = defaultdict(list)

    user_ids = list()
    item_ids1 = list()

    id2asin = dict()

    # train data
    with open('./review_datasets/%s/%s_train.txt' % (fname, fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)-1
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids1.append(i)
            User[u].append(i)

    with open('./review_datasets/%s/%s_valid.txt' % (fname, fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)-1
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids1.append(i)
            User[u].append(i)

    with open('./review_datasets/%s/%s_test.txt' % (fname, fname), 'r') as f:
        for line in f:
            u, i, t, asin = line.rstrip().split(' ')
            u = int(u)-1
            i = int(i)
            id2asin[i] = asin
            user_ids.append(u)
            item_ids1.append(i)
            User[u].append(i)

    item_map = dict()
    itemnum1 = 0

    idmap2asin = dict()
    MetaMeanFeatures1 = {}
    for i in item_ids1:
        if i not in item_map:
            item_map[i] = itemnum1
            idmap2asin[item_map[i]] = id2asin[i]
            MetaMeanFeatures1[item_map[i]] = MetaMeanFeatures[i]
            itemnum1 += 1


    uid_list = list(User.keys())
    uid_list.sort(key=lambda t: int(t))
    item_list = list(range(itemnum1))

    meta_embeddings = []
    for i in item_list:
        meta_embeddings.append(MetaMeanFeatures1[i])

    meta_embeddings = torch.tensor(meta_embeddings, dtype=torch.float32)

    meta_file = os.path.join(output_path, fname, fname + '.pth')
    torch.save(meta_embeddings, meta_file)

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
            target_item = valid_data[uid][0]
            item_seq = [str(item) for item in item_seq]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(os.path.join(output_path, fname, f'{fname}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            item_seq = [str(item) for item in item_seq]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    print('success')


if __name__ == '__main__':
    InformationInfo = get_emb(args.x)
    data_partition(args.x, InformationInfo)

    InformationInfo = get_emb(args.y)
    data_partition(args.y, InformationInfo)

    InformationInfo = get_emb(args.z)
    data_partition(args.z, InformationInfo)


