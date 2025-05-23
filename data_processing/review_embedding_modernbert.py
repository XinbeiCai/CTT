import os
import pickle
import torch
import pandas as pd

import torch
from tqdm import tqdm
import argparse
import numpy as np
from transformers import AutoModel, AutoTokenizer

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
    batch_size = 1024
    data_directory = '../review_datasets/%s/' % category_name
    data_path = os.path.join(data_directory, '%s_review.csv' % category_name)
    df = pd.read_csv(data_path, names=['user_id', 'item_id', 'reviews', 'flag'])
    df.reset_index(inplace=True)

    # bert_train_embedding
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model_name = "./modernbert"  # 或 "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    review_embedding = []
    batch_num = int(df.shape[0] / batch_size)
    for i in tqdm(range(batch_num), total=int(batch_num), ncols=70, leave=False, unit='b'):
        if i == batch_num - 1:
            batch = df.loc[batch_size * i:, 'reviews'].tolist()
        else:
            batch = df.loc[batch_size * i:batch_size * (i + 1) - 1, 'reviews'].tolist()

        inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            review_embedding.extend(embeddings.tolist())

    # processing_data
    df['review_embedding'] = review_embedding
    assert len(review_embedding) == len(df), "error num"
    print('review_%s_emb_done!' % category_name)

    return df
    # df_review = df.drop(['reviews'], axis=1)
    #
    # with open(os.path.join(data_directory, '%s_review_emb.dat' % category_name), 'wb') as f:
    #     pickle.dump(df_review, f)


def review_mean(category_name, df):
    data_directory = '../review_datasets/%s/' % category_name
    train_review = df[df['flag'] == 'train']

    # only process the train data
    def embedding_mean(x):
        all_review_emb = list(x.review_embedding)
        return np.mean(all_review_emb, axis=0)

    # group by item id
    group_item = train_review.groupby('item_id').apply(embedding_mean)
    item_emb = pd.DataFrame({'item_id': group_item.index, 'reviews_emb': group_item.values})

    ReviewINFO = {}
    error_count = 0
    for index, row in item_emb.iterrows():
        try:
            itemid = int(row['item_id'])
            review_embedding = row['reviews_emb']
            ReviewINFO[itemid] = review_embedding
        except Exception as e:
            error_count += 1
            print(f"warn: error item_id '{row['item_id']}' (error: {str(e)})")
            continue
    if error_count > 0:
        print(f"num: {error_count}")

    with open(data_directory + '%s_review_emb_mean_modernbert.dat' % category_name, 'wb') as f:
        pickle.dump(ReviewINFO, f)

    print('review_%s_write_done!' % category_name)


if __name__ == '__main__':
    df = review_emb(args.x)
    review_mean(args.x, df)

    df = review_emb(args.y)
    review_mean(args.y, df)

    df = review_emb(args.z)
    review_mean(args.z, df)
