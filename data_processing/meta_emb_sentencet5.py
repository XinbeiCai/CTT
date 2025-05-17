import os
import pickle
import torch
import pandas as pd
from transformers import T5EncoderModel,AutoModel, AutoTokenizer
from tqdm import tqdm
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-x', default='Books', help='category')
parser.add_argument('-y', default='Movies_and_TV', help='category')
parser.add_argument('-z', default='CDs_and_Vinyl', help='category')
args = parser.parse_args()



def review_emb(category_name):
    batch_size = 1024
    data_directory = f'../review_datasets/{category_name}/'
    meta_file_path = os.path.join(data_directory, f'{category_name}_meta.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_name = "./sentence-t5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    df = pd.read_csv(meta_file_path, header=None, names=['item_id', 'information'])
    df.reset_index(inplace=True)

    information_embedding = []
    batch_num = (df.shape[0] + batch_size - 1) // batch_size

    for i in tqdm(range(batch_num), total=int(batch_num), ncols=70, leave=False, unit='b'):
        if i == batch_num - 1:
            batch = df.loc[batch_size * i:, 'information'].tolist()
        else:
            batch = df.loc[batch_size * i:batch_size * (i + 1) - 1, 'information'].tolist()

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
            information_embedding.extend(embeddings.tolist())

    df['meta_embedding'] = information_embedding
    df_information = df.drop(['information'], axis=1)
    assert len(information_embedding) == len(df), "error！"

    InformationInfo = {}
    error_count = 0

    for _, row in df_information.iterrows():
        try:
            item_str = str(row['item_id']).strip()

            if not item_str.isdigit():
                raise ValueError(f"error char: {item_str}")

            itemid = int(item_str)
            InformationInfo[itemid] = row['meta_embedding']

        except Exception as e:
            error_count += 1
            print(f"warn: error item_id '{row['item_id']}' (error: {str(e)})")
            continue

    if error_count > 0:
        print(f"num: {error_count}")

    with open(data_directory + f'{category_name}_meta_emb_t5.dat', 'wb') as f:
        pickle.dump(InformationInfo, f)

    print(f'meta_{category_name}_write_done!')


if __name__ == '__main__':
    review_emb(args.x)
    review_emb(args.y)
    review_emb(args.z)