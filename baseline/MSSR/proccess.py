import json

import pandas as pd
import numpy as np
import argparse
import os
from sklearn import preprocessing
import pickle

filter_min = 5
sample_num = 100
sample_pop = True

parser = argparse.ArgumentParser()
parser.add_argument('-x', default='Books', help='category')
parser.add_argument('-y', default='Movies_and_TV', help='category')
parser.add_argument('-z', default='CDs_and_Vinyl', help='category')
args = parser.parse_args()

def raw_process(category_name):
    data_directory = './dataset/%s/' % category_name
    data_path = os.path.join(data_directory, 'ratings_%s.csv' % category_name)

    raw_sep = ','
    df = pd.read_csv(data_path, sep=raw_sep, header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])

    item_id, title, salesRank, categories, price, brand = [], [], [], [], [], []
    meta_data_path = os.path.join(data_directory, 'meta_%s.json' % category_name)
    with open(meta_data_path, 'r') as f:
        for line in f:
            js = eval(line)
            item_id.append(str(js['asin']))
            title.append(js.get('title', ''))
            salesRank.append(js.get('salesRank', ''))
            categories.append(js.get('categories', ''))
            price.append(js.get('price', ''))
            brand.append(js.get('brand', ''))

    meta_df = pd.DataFrame({
        'item_id': pd.Series(item_id),
        'title': pd.Series(title),
        'salesRank': pd.Series(salesRank),
        'categories': pd.Series(categories),
        'price': pd.Series(price),
        'brand': pd.Series(brand),
    })[['item_id', 'title', 'salesRank', 'categories', 'price', 'brand']]

    meta_df.drop_duplicates(subset=['item_id'], keep='first', inplace=True)
    df = pd.merge(df, meta_df, how='left', on='item_id')

    # raw_info
    print("============ %s ============" % data_path)
    print("================= raw info =============================")
    print("#users: %d" % len(df.user_id.unique()))
    print("#items: %d" % len(df.item_id.unique()))
    print("#actions: %d" % len(df))

    # ========================================================
    # sort by time
    df.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)

    # ========================================================
    # drop duplicated user-item pairs
    df.drop_duplicates(subset=['user_id', 'item_id'], keep='first', inplace=True)

    # ========================================================
    # discard cold-start items
    count_i = df.groupby('item_id').user_id.count()
    item_keep = count_i[count_i >= filter_min].index
    df = df[df['item_id'].isin(item_keep)]

    # discard cold-start users
    count_u = df.groupby('user_id').item_id.count()
    user_keep = count_u[filter_min <= count_u].index
    df = df[df['user_id'].isin(user_keep)]

    return df


def whole_process(df1, df2, df3):
    user = pd.Series(list(set(df1['user_id']).intersection(set(df2['user_id'])).intersection(set(df3['user_id']))))
    print("same user: ", len(user))
    items = pd.Series(list(set(df1['item_id']).intersection(set(df2['item_id'])).intersection(set(df3['item_id']))))
    print("same items: ", len(items))
    print(items)

    df4 = pd.concat([df1, df2, df3], keys=['x', 'y', 'z'])
    df4 = df4[df4['user_id'].isin(user)]

    # renumber user ids and item ids

    # output statistical information
    print("==== statistic of processed data (whole) ====")
    n = len(df4.user_id.unique())
    m = len(df4.item_id.unique())
    p = len(df4)
    print("#users: %d" % n)
    print("#items: %d" % m)
    print("#actions: %d" % p)
    print("density: %.4f" % (p / n / m))

    count_u = df4.groupby(['user_id']).item_id.count()
    print("min #actions per user: %.2f" % count_u.min())
    print("max #actions per user: %.2f" % count_u.max())
    print("ave #actions per user: %.2f" % count_u.mean())

    return df4


def domain_process(df4, category_name):
    df = None
    if category_name == args.x:
        df = df4.loc['x']
    elif category_name == args.y:
        df = df4.loc['y']
    elif category_name == args.z:
        df = df4.loc['z']
    # output statistical information
    print("==== statistic of %s processed data (whole) ====" % category_name)
    n = len(df.user_id.unique())
    m = len(df.item_id.unique())
    p = len(df)
    print("#users: %d" % n)
    print("#items: %d" % m)
    print("#actions: %d" % p)
    print("density: %.4f" % (p / n / m))

    count_u = df.groupby(['user_id']).item_id.count()
    print("min #actions per user: %.2f" % count_u.min())
    print("max #actions per user: %.2f" % count_u.max())
    print("ave #actions per user: %.2f" % count_u.mean())

    # split data into test set, valid set and train set,
    # adopting the leave-one-out evaluation for next-item recommendation task

    # ========================================
    # obtain possible records in test set
    df_test = df.groupby(['user_id']).tail(1)
    df.drop(df_test.index, axis='index', inplace=True)

    # ========================================
    # obtain possible records in valid set
    df_valid = df.groupby(['user_id']).tail(1)
    df.drop(df_valid.index, axis='index', inplace=True)

    # ========================================
    # drop cold-start items in valid set and test set
    df_valid = df_valid[df_valid.item_id.isin(df.item_id)]
    df_test = df_test[df_test.user_id.isin(df_valid.user_id) & (
            df_test.item_id.isin(df.item_id) | df_test.item_id.isin(df_valid.item_id))]

    # ========================================
    # df_concat
    data_path = './dataset/%s/%s.inter' % (category_name, category_name)
    df_concat = pd.concat([df, df_valid, df_test], axis='index')
    print(df_concat.head(5))
    df_inter = df_concat.loc[:, ['user_id', 'item_id', 'rating','timestamp']]
    print(df_inter.head(5))
    with open(data_path, 'w') as file:
        file.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        for index, row in df_inter.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = float(row['rating'])
            timestamp = float(row['timestamp'])
            file.write(f'{user_id}\t{item_id}\t{rating}\t{timestamp}\n')
    # df_inter.to_csv(data_path, sep='\t', header=False, index=False)
    file.close()
    print('success_inter')

    data_path = './dataset/%s/%s.item' % (category_name, category_name)
    meta_df = df_concat.loc[:, ['item_id', 'title', 'salesRank', 'categories','price','brand']]
    meta_df.drop_duplicates(subset=['item_id'], keep='first', inplace=True)

    with open(data_path, 'w') as file:
        file.write('item_id:token\ttitle:token\tsales_rank:float\tcategories:token_seq\tprice:float\tbrand:token\n')
        for index, row in meta_df.iterrows():
            item_id = row['item_id']
            title = row['title']
            # sales_rank = float(row['salesRank'])
            categories = row['categories']
            # price = float(row['price'])
            brand = row['brand']
            file.write(f'{item_id}\t{title}\t{categories}\t{brand}\n')
    print('success_item')


if __name__ == '__main__':
    df1 = raw_process(args.x)
    df2 = raw_process(args.y)
    df3 = raw_process(args.z)
    df4 = whole_process(df1, df2, df3)
    domain_process(df4, args.x)
    domain_process(df4, args.y)
    domain_process(df4, args.z)
