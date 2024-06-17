import pickle
import os
import numpy as np
import pandas as pd
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


def review_meaning(category_name):
    data_directory = './review_datasets/%s/' % category_name
    all_review_emb = load_data(os.path.join(data_directory, '%s_review_emb.dat' % category_name))

    train = load_data(data_directory + category_name + '_train.dat')
    valid = load_data(data_directory + category_name + '_valid.dat')
    test = load_data(data_directory + category_name + '_test.dat')
    df_concat = pd.concat([train, test, valid], axis=0)

    df_concat_review = pd.merge(all_review_emb, df_concat, how='inner', on=['user_id', 'item_id', 'reviews'])
    train_review = df_concat_review[df_concat_review['flag'] == 'train']

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

    print('mean_review_%s_write_done!' % category_name)


if __name__ == '__main__':
    review_meaning(args.x)
    review_meaning(args.y)
    review_meaning(args.z)
