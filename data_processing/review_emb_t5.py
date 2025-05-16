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
        print('文件为空')
    return x

parser = argparse.ArgumentParser()
parser.add_argument('-x', default='Books', help='category')
parser.add_argument('-y', default='Movies_and_TV', help='category')
parser.add_argument('-z', default='CDs_and_Vinyl', help='category')

args = parser.parse_args()

def review_emb(category_name, type):
    # read_data
    batch_size = 5
    data_directory = '../review_datasets/%s/' % category_name
    data_path = os.path.join(data_directory, '%s_review.csv' % category_name)
    # df = load_data(data_path)
    df = pd.read_csv(data_path, header=None, names=['user_id', 'item_id', 'reviews', 'flag'])
    df.reset_index(inplace=True)

    # bert_train_embedding
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model_path = './'+type
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)

    review_embedding = []
    batch_num = int(df.shape[0] / batch_size)
    for i in tqdm(range(batch_num), total=int(batch_num), ncols=70, leave=False, unit='b'):
        if i == batch_num - 1:
            batch = df.loc[batch_size * i:, 'reviews'].tolist()
        else:
            batch = df.loc[batch_size * i:batch_size * (i + 1) - 1, 'reviews'].tolist()

        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
        inputs = inputs.to(device)

        with torch.no_grad():
            encoder_outputs = model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            last_hidden_state = encoder_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

            # Mean pooling (加权平均)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch_size, seq_len, 1)
            summed = torch.sum(last_hidden_state * attention_mask, dim=1)
            count = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / count  # (batch_size, hidden_size)

        review_embedding.extend(mean_pooled.cpu().tolist())

    # 保存
    df['review_embedding'] = review_embedding
    print('review_%s_%s_emb_done!' % (category_name, type))

    return df



def review_mean(category_name, df, type):
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
    for index, row in item_emb.iterrows():
        itemid = int(row['item_id'])
        review_embedding = row['reviews_emb']
        ReviewINFO[itemid] = review_embedding

    model_name = type.split("-")[1]
    with open(data_directory + '%s_review_emb_mean_%s.dat' % (category_name, model_name), 'wb') as f:
        pickle.dump(ReviewINFO, f)

    print('review_%s_%s_write_done!' % (category_name, model_name))


if __name__ == '__main__':
    # type = 'multilingual-e5-large'
    type = 'sentence-t5'
    df = review_emb(args.x, type)
    review_mean(args.x, df, type)

    df = review_emb(args.y, type)
    review_mean(args.y, df, type)

    df = review_emb(args.z, type)
    review_mean(args.z, df, type)
