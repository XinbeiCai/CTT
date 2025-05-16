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
parser.add_argument('-x', default='Home_and_Kitchen', help='category')
parser.add_argument('-y', default='Grocery_and_Gourmet_Food', help='category')
parser.add_argument('-z', default='CDs_and_Vinyl', help='category')

args = parser.parse_args()

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x = pickle.load(f)
    except:
        x = []
    return x

def review_emb(category_name, type):
    # read_data
    batch_size = 512
    data_directory = '../review_datasets/%s/' % category_name
    # data_path = os.path.join(data_directory, '%s_meta.dat' % category_name)
    data_path = os.path.join(data_directory, '%s_meta.csv' % category_name)
    # df = load_data(data_path)
    df = pd.read_csv(data_path, names=['item_id', 'information'])
    df.reset_index(inplace=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model_path = './' + type
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)

    information_embedding = []
    batch_num = int(df.shape[0] / batch_size)
    for i in tqdm(range(batch_num), total=int(batch_num), ncols=70, leave=False, unit='b'):
        if i == batch_num - 1:
            batch = df.loc[batch_size * i:, 'information'].tolist()
        else:
            batch = df.loc[batch_size * i:batch_size * (i + 1) - 1, 'information'].tolist()

        # 强制转换为字符串
        batch = list(map(str, batch))
        input = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
        input = input.to(device)

        with torch.no_grad():
            outputs = model(**input)  # shape (batch_size, max_length, 768)
            v = torch.mean(outputs.last_hidden_state, dim=1)  # shape (batch_size, 768)
            information_embedding.extend(v.tolist())

    # processing_data
    df['information_embedding'] = information_embedding
    df_information = df.drop(['information'], axis=1)

    InformationInfo = {}
    for index, row in df_information.iterrows():
        try:
            itemid = int(row['item_id'])  # Try to convert the 'item id' to an integer
            information_embedding = row['information_embedding']
            InformationInfo[itemid] = information_embedding
        except ValueError:  # Catch the error if it's not a valid integer
            print(f"Invalid item id found: {row['item_id']}")  # Print a warning
            continue  # Skip this row or handle it differently

    # write_review_embedding_data
    with open(data_directory + '%s_meta_emb_%s.dat' % (category_name, type), 'wb') as f:
        pickle.dump(InformationInfo, f)

    print('meta_%s_%s_write_done!' % (category_name, type))

if __name__ == '__main__':
    type = 'bert-base-cased'
    review_emb(args.x, type)
    review_emb(args.y, type)
    review_emb(args.z, type)

