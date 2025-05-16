import os
import pickle
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-x', default='Home_and_Kitchen', help='category')
parser.add_argument('-y', default='Grocery_and_Gourmet_Food', help='category')
parser.add_argument('-z', default='CDs_and_Vinyl', help='category')

args = parser.parse_args()
def review_emb(category_name, type):

    # read_data
    batch_size = 512
    data_directory = '../review_datasets/%s/' % category_name
    data_path = os.path.join(data_directory, '%s_review.csv' % category_name)
    df = pd.read_csv(data_path, names=['user_id', 'item_id', 'reviews', 'flag'])
    df.reset_index(inplace=True)

    # bert_train_embedding
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model_path = './' + type
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)

    review_embedding = []
    batch_num = int(df.shape[0] / batch_size) + 1
    for i in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b'):
        batch = df.loc[batch_size * i:batch_size * (i + 1) - 1, 'reviews'].tolist()

        # 强制转换为字符串
        batch = list(map(str, batch))

        try:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                v = torch.mean(outputs.last_hidden_state, dim=1)
                review_embedding.extend(v.tolist())
        except Exception as e:
            print(f"[Error] 第 {i} 个 batch 编码失败，错误信息：{e}")
            continue

    # 确保长度一致
    df = df.iloc[:len(review_embedding)]
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
    print("#items: %d" % len(train_review.item_id.unique()))
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
    type = 'bert-base-cased'
    # type = 'sentence-t5'
    df = review_emb(args.x, type)
    review_mean(args.x, df, type)

    df = review_emb(args.y, type)
    review_mean(args.y, df, type)

    df = review_emb(args.z, type)
    review_mean(args.z, df, type)
