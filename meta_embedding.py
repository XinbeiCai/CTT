import os
import pickle
import torch
import pandas as pd
import sys
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

def review_emb(category_name):
    # read_data
    batch_size = 5
    data_directory = './datasets/%s/' % category_name
    data_path = os.path.join(data_directory, '%s_meta.dat' % category_name)
    df = load_data(data_path)
    df.reset_index(inplace=True)

    # bert_train_embedding
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-cased', use_fast=True)
    model = AutoModel.from_pretrained('./bert-base-cased')
    model.to(device)

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

    # write_review_embedding_data
    with open(data_directory + '%s_meta_emb.dat' % category_name, 'wb') as f:
        pickle.dump(InformationInfo, f)

    print('meta_%s_write_done!' % category_name)

if __name__ == '__main__':
    review_emb(args.x)
    review_emb(args.y)
    review_emb(args.z)

