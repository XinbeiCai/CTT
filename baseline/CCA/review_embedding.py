import os
import pickle
import torch
import pandas as pd
import sys
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

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
    df_review = df.drop(['reviews'], axis=1)
    ReviewInfo = {}
    for index, row in df_review.iterrows():
        userid = int(row['user_id'])
        itemid = int(row['item_id'])
        review_embedding = row['review_embedding']
        ReviewInfo[(userid, itemid)] = review_embedding

    # write_review_embedding_data
    with open(data_directory + '%s_review_emb_single.dat' % category_name, 'wb') as f:
        pickle.dump(ReviewInfo, f)

    with open(os.path.join(data_directory, '%s_review_emb.dat' % category_name), 'wb') as f:
        pickle.dump(df, f)

    print('review_%s_write_done!' % category_name)

if __name__ == '__main__':
    review_emb(args.x)
    review_emb(args.y)
    review_emb(args.z)

