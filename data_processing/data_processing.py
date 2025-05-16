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
    data_directory = '../review_datasets/%s/' % category_name
    data_path = os.path.join(data_directory, 'reviews_%s.json' % category_name)

    meta_data_path = os.path.join(data_directory, 'meta_%s.json' % category_name)

    asin, title, brand, description = [], [], [], []
    with open(meta_data_path, 'r') as f:
        for line in f:
            js = eval(line)
            asin.append(str(js['asin']))
            title.append(js.get('title', ''))
            brand.append(js.get('brand', ''))
            description.append(js.get('description', ''))

    meta_df = pd.DataFrame({
        'asin': pd.Series(asin),
        'title': pd.Series(title),
        'brand': pd.Series(brand),
        'description': pd.Series(description),
    })[['asin', 'title', 'brand', 'description']]

    users_id, items_id, ratings, reviews, times = [], [], [], [], []
    with open(data_path, 'r') as f:
        for line in f:
            js = json.loads(line)
            reviews.append(js['reviewText'])
            users_id.append(str(js['reviewerID']))
            items_id.append(str(js['asin']))
            ratings.append(float(js['overall']))
            times.append(int(js['unixReviewTime']))

    f.close()

    df = pd.DataFrame({
        'user_id': pd.Series(users_id),
        'asin': pd.Series(items_id),
        'ratings': pd.Series(ratings),
        'reviews': pd.Series(reviews),
        'timestamp': pd.Series(times)
    })[['user_id', 'asin', 'ratings', 'reviews', 'timestamp']]

    # raw_info
    print("============ %s ============" % data_path)
    print("================= raw info =============================")
    print("#users: %d" % len(df.user_id.unique()))
    print("#items: %d" % len(df.asin.unique()))
    print("#actions: %d" % len(df))

    # ========================================================
    # sort by time
    df.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)

    # ========================================================
    # drop duplicated user-item pairs
    df.drop_duplicates(subset=['user_id', 'asin'], keep='first', inplace=True)
    meta_df.drop_duplicates(subset=['asin'], keep='first', inplace=True)

    df = pd.merge(df, meta_df, how='left', on='asin')

    columns_to_fill = ['title', 'brand', 'description']
    df[columns_to_fill] = df[columns_to_fill].fillna('')

    # ========================================================
    # discard cold-start items
    count_i = df.groupby('asin').user_id.count()
    item_keep = count_i[count_i >= filter_min].index
    df = df[df['asin'].isin(item_keep)]

    # discard cold-start users
    count_u = df.groupby('user_id').asin.count()
    user_keep = count_u[filter_min <= count_u].index
    df = df[df['user_id'].isin(user_keep)]

    # output statistical information
    # print("==== statistic of processed data (whole) ====")
    # n = len(df.user_id.unique())
    # m = len(df.asin.unique())
    # p = len(df)
    # print("#users: %d" % n)
    # print("#items: %d" % m)
    # print("#actions: %d" % p)
    # print("density: %.4f" % (p / n / m))
    #
    # count_u = df.groupby(['user_id']).asin.count()
    # print("min #actions per user: %.2f" % count_u.min())
    # print("max #actions per user: %.2f" % count_u.max())
    # print("ave #actions per user: %.2f" % count_u.mean())

    return df


def whole_process(df1, df2, df3):
    user = pd.Series(list(set(df1['user_id']).intersection(set(df2['user_id'])).intersection(set(df3['user_id']))))
    print("same user: ", len(user))
    items = pd.Series(list(set(df1['asin']).intersection(set(df2['asin'])).intersection(set(df3['asin']))))
    print("same items: ", len(items))
    print(items)

    df4 = pd.concat([df1, df2, df3], keys=['x', 'y', 'z'])
    df4 = df4[df4['user_id'].isin(user)]

    # renumber user ids and item ids
    le = preprocessing.LabelEncoder()
    df4['user_id'] = le.fit_transform(df4['user_id']) + 1
    df4['item_id'] = le.fit_transform(df4['asin']) + 1

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
    data_directory = '../review_datasets/%s/' % category_name
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

    df['flag'] = list(['train'] * df.shape[0])
    df_valid['flag'] = list(['valid'] * df_valid.shape[0])
    df_test['flag'] = list(['test'] * df_test.shape[0])

    # output statistical information
    print("==== statistic of %s processed data (split) ====" % category_name)
    print("#train_users: %d" % len(df.user_id.unique()))
    print("#train_items: %d" % len(df.item_id.unique()))
    print("#valid_users: %d" % len(df_valid.user_id.unique()))
    print("#test_users: %d" % len(df_test.user_id.unique()))

    # =====================================================================================
    # save the item_seq to dat
    # with open(data_directory + category_name + '_train.dat', 'wb') as f:
    #     pickle.dump(df, f)
    # with open(data_directory + category_name + '_valid.dat', 'wb') as f:
    #     pickle.dump(df_valid, f)
    # with open(data_directory + category_name + '_test.dat', 'wb') as f:
    #     pickle.dump(df_test, f)

    # =====================================================================================
    # # save the item_seq to txt
    df_train_data = df.loc[:, ['user_id', 'item_id', 'timestamp','asin']]
    df_train_data.to_csv(data_directory + category_name + '_train.txt', sep=' ', index=False, header=False)

    df_test_data = df_test.loc[:, ['user_id', 'item_id', 'timestamp','asin']]
    df_test_data.to_csv(data_directory + category_name + '_test.txt', sep=' ', index=False, header=False)

    df_valid_data = df_valid.loc[:, ['user_id', 'item_id', 'timestamp','asin']]
    df_valid_data.to_csv(data_directory + category_name + '_valid.txt', sep=' ', index=False, header=False)

    print('success write %s train, valid, test item' % category_name)

    # ========================================
    # df_concat
    df_concat = pd.concat([df, df_valid, df_test], axis='index')

    # ========================================
    # save the review document
    # df_review = df_concat.loc[:, ['user_id', 'item_id', 'reviews', 'flag']]

    # with open(data_directory + category_name + '_review.dat', 'wb') as f:
    #     pickle.dump(df_review, f)
    #
    # print('success write %s review.dat' % category_name)

    # meta
    # df_meta = df_concat.loc[:, ['item_id', 'title', 'brand', 'description']]
    # deduplicated_df = df_meta.drop_duplicates(subset='item_id')
    # deduplicated_df.loc[:, 'information'] = deduplicated_df['title'] + " " +deduplicated_df['brand'] + " " + deduplicated_df[
    #     'description']
    # df_meta = deduplicated_df.loc[:, ['item_id', 'information']]
    #
    # with open(data_directory + category_name + '_meta.dat', 'wb') as f:
    #     pickle.dump(df_meta, f)
    #
    # print('success write %s meta.dat' % category_name)


    # save the review document
    df_review = df_concat.loc[:, ['user_id', 'item_id', 'reviews', 'flag']]
    df_review.to_csv(os.path.join(data_directory, f'{category_name}_review.csv'), index=False)
    print(f'success write {category_name} review.csv')

    # meta
    df_meta = df_concat.loc[:, ['item_id', 'title', 'brand', 'description']]
    deduplicated_df = df_meta.drop_duplicates(subset='item_id')
    deduplicated_df.loc[:, 'information'] = deduplicated_df['title'] + " " + deduplicated_df['brand'] + " " + \
                                            deduplicated_df['description']
    df_meta = deduplicated_df.loc[:, ['item_id', 'information']]
    df_meta.to_csv(os.path.join(data_directory, f'{category_name}_meta.csv'), index=False)
    print(f'success write {category_name} meta.csv')

    # sr_user2items = df_concat.groupby(['user_id']).item_id.unique()
    # df_negative = pd.DataFrame({'user_id': df_concat.user_id.unique()})

    sr_user2items = df_concat.groupby(['user_id']).item_id.unique()
    df_negative = pd.DataFrame({'user_id': df_concat.user_id.unique()})

    # ========================================================
    # For each user, randomly sample some negative items,
    # and rank these items with the ground-truth item when testing or validation
    # sample according to popularity
    if sample_pop == True:
        sr_item2pop = df_concat.item_id.value_counts(sort=True, ascending=False)
        arr_item = sr_item2pop.index.values
        arr_pop = sr_item2pop.values

        def get_negative_sample(pos):
            neg_idx = ~np.in1d(arr_item, pos)
            neg_item = arr_item[neg_idx]
            neg_pop = arr_pop[neg_idx]
            neg_pop = neg_pop / neg_pop.sum()

            return np.random.choice(neg_item, size=sample_num, replace=False, p=neg_pop)

        arr_sample = df_negative.user_id.apply(
            lambda x: get_negative_sample(sr_user2items[x])).values

    # ========================================
    # sample uniformly
    else:
        arr_item = df_concat.item_id.unique()
        arr_sample = df_negative.user_id.apply(
            lambda x: np.random.choice(
                arr_item[~np.in1d(arr_item, sr_user2items[x])], size=sample_num, replace=False)).values

    # output negative data
    df_negative = pd.concat([df_negative, pd.DataFrame(list(arr_sample))], axis='columns')
    df_negative.to_csv(data_directory + "%s_negative.csv" % category_name, header=False, index=False)
    print('success write %s negetive item' % category_name)


if __name__ == '__main__':
    df1 = raw_process(args.x)
    df2 = raw_process(args.y)
    df3 = raw_process(args.z)
    df4 = whole_process(df1, df2, df3)
    domain_process(df4, args.x)
    domain_process(df4, args.y)
    domain_process(df4, args.z)
