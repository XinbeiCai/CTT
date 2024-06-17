import os
import pickle
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import argparse
import numpy as np

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

def review_emb(category_name):
    # read_data
    batch_size = 5
    data_directory = './datasets/%s/' % category_name
    data_path = os.path.join(data_directory, '%s_review.dat' % category_name)
    df = load_data(data_path)
    df.reset_index(inplace=True)

    # bert_train_embedding
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    tokenizer = AutoTokenizer.from_pretrained('./bert-base-cased', use_fast=True)
    model = AutoModel.from_pretrained('./bert-base-cased')
    model.to(device)

    review_embedding = []
    batch_num = int(df.shape[0] / batch_size)
    for i in tqdm(range(batch_num), total=int(batch_num), ncols=70, leave=False, unit='b'):
        if i == batch_num - 1:
            batch = df.loc[batch_size * i:, 'reviews'].tolist()
        else:
            batch = df.loc[batch_size * i:batch_size * (i + 1) - 1, 'reviews'].tolist()

        input = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
        input = input.to(device)

        outputs = model(**input)  # shape (batch_size, max_length, 768)

        v = torch.mean(outputs.last_hidden_state, dim=1)  # shape (batch_size, 768)

        review_embedding.extend(v.tolist())

    # processing_data
    df['review_embedding'] = review_embedding

    print('review_%s_emb_done!' % category_name)

    return df
    # df_review = df.drop(['reviews'], axis=1)
    #
    # with open(os.path.join(data_directory, '%s_review_emb.dat' % category_name), 'wb') as f:
    #     pickle.dump(df_review, f)


def review_mean(category_name, df):
    data_directory = './datasets/%s/' % category_name
    train_review = df[df['flag'] == 'train']

    # only process the train data
    def embedding_mean(x):
        all_review_emb = list(x.review_embedding)
        return np.mean(all_review_emb, axis=0)

    # group by item id
    group_item = train_review.groupby('item_id').apply(embedding_mean)
    item_emb = pd.DataFrame({'item_id': group_item.index, 'reviews_emb': group_item.values})

    ReviewINFO = {}
    for index, row in item_emb.iterrows():
        itemid = int(row['item_id'])
        review_embedding = row['reviews_emb']
        ReviewINFO[itemid] = review_embedding

    with open(data_directory + '%s_review_emb_mean.dat' % category_name, 'wb') as f:
        pickle.dump(ReviewINFO, f)

    print('review_%s_write_done!' % category_name)


if __name__ == '__main__':
    df = review_emb(args.x)
    review_mean(args.x, df)

    df = review_emb(args.y)
    review_mean(args.y, df)

    df = review_emb(args.z)
    review_mean(args.z, df)
