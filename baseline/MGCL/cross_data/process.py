from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from sklearn import preprocessing
import argparse


raw_sep = ','
filter_min = 5
sample_num = 100
sample_pop = True

parser = argparse.ArgumentParser()
parser.add_argument('-x', default='Books', help='category')
parser.add_argument('-y', default='Movies_and_TV', help='category')
parser.add_argument('-z', default='CDs_and_Vinyl', help='category')

args = parser.parse_args()
raw_file = "amazon/ratings_" + args.x + ".csv"
processed_file_prefix_x = "processed_data_all/" + args.x + "_"
processed_file_prefix_y = "processed_data_all/" + args.y + "_"
processed_file_prefix_z = "processed_data_all/" + args.z + "_"
processed_file_prefix = "processed_data_all/" + args.x + args.y + args.z + "_"
# ================================================================================
# obtain implicit feedbacks
df = pd.read_csv(raw_file, sep=raw_sep, header=None, names=['user_id','item_id','rating','timestamp'])
df = df[['user_id','item_id','timestamp']]

print("==== statistic of raw data ====")
print("#users: %d" % len(df.user_id.unique()))
print("#items: %d" % len(df.item_id.unique()))
print("#actions: %d" % len(df))

# ========================================================
# sort by time
df.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)

# ========================================================
# drop duplicated user-item pairs
df.drop_duplicates(subset=['user_id','item_id'], keep='first', inplace=True)

# ========================================================
# discard cold-start items
count_i = df.groupby('item_id').user_id.count()
item_keep = count_i[count_i >= filter_min].index
df = df[df['item_id'].isin(item_keep)]

# discard cold-start users
count_u = df.groupby('user_id').item_id.count()
user_keep = count_u[count_u >= filter_min].index
df = df[df['user_id'].isin(user_keep)]

# output statistical information
print("==== statistic of processed data (whole) ====")
n = len(df.user_id.unique())
m = len(df.item_id.unique())
p = len(df)
print("#users: %d" % n)
print("#items: %d" % m)
print("#actions: %d" % p)
print("density: %.4f" % (p/n/m))

count_u = df.groupby(['user_id']).item_id.count()
print("min #actions per user: %.2f" % count_u.min())
print("max #actions per user: %.2f" % count_u.max())
print("ave #actions per user: %.2f" % count_u.mean())

# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================

raw_file = "amazon/ratings_" + args.y + ".csv"
# ================================================================================
# obtain implicit feedbacks
df2 = pd.read_csv(raw_file, sep=raw_sep, header=None, names=['user_id','item_id','rating','timestamp'])
df2 = df2[['user_id','item_id','timestamp']]

# df = pd.concat([df, df2], keys=['x', 'y'], ignore_index=True)
print("\n\n\n==== statistic of raw data ====")
print("#users: %d" % len(df2.user_id.unique()))
print("#items: %d" % len(df2.item_id.unique()))
print("#actions: %d" % len(df2))

# ========================================================
# sort by time
df2.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)

# ========================================================
# drop duplicated user-item pairs
df2.drop_duplicates(subset=['user_id','item_id'], keep='first', inplace=True)

# ========================================================
# discard cold-start items
count_i = df2.groupby('item_id').user_id.count()
item_keep = count_i[count_i >= filter_min].index
df2 = df2[df2['item_id'].isin(item_keep)]

# discard cold-start users
count_u = df2.groupby('user_id').item_id.count()
user_keep = count_u[count_u >= filter_min].index
df2 = df2[df2['user_id'].isin(user_keep)]

# output statistical information
print("==== statistic of processed data (whole) ====")
n = len(df2.user_id.unique())
m = len(df2.item_id.unique())
p = len(df2)
print("#users: %d" % n)
print("#items: %d" % m)
print("#actions: %d" % p)
print("density: %.4f" % (p/n/m))

count_u = df2.groupby(['user_id']).item_id.count()
print("min #actions per user: %.2f" % count_u.min())
print("max #actions per user: %.2f" % count_u.max())
print("ave #actions per user: %.2f" % count_u.mean())


# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================

raw_file = "amazon/ratings_" + args.z + ".csv"
# ================================================================================
# obtain implicit feedbacks
df3 = pd.read_csv(raw_file, sep=raw_sep, header=None, names=['user_id','item_id','rating','timestamp'])
df3 = df3[['user_id','item_id','timestamp']]

# df = pd.concat([df, df2], keys=['x', 'y'], ignore_index=True)
print("\n\n\n==== statistic of raw data ====")
print("#users: %d" % len(df3.user_id.unique()))
print("#items: %d" % len(df3.item_id.unique()))
print("#actions: %d" % len(df3))

# ========================================================
# sort by time
df3.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)

# ========================================================
# drop duplicated user-item pairs
df3.drop_duplicates(subset=['user_id','item_id'], keep='first', inplace=True)

# ========================================================
# discard cold-start items
count_i = df3.groupby('item_id').user_id.count()
item_keep = count_i[count_i >= filter_min].index
df3 = df3[df3['item_id'].isin(item_keep)]

# discard cold-start users
count_u = df3.groupby('user_id').item_id.count()
user_keep = count_u[count_u >= filter_min].index
df3 = df3[df3['user_id'].isin(user_keep)]

# output statistical information
print("==== statistic of processed data (whole) ====")
n = len(df3.user_id.unique())
m = len(df3.item_id.unique())
p = len(df3)
print("#users: %d" % n)
print("#items: %d" % m)
print("#actions: %d" % p)
print("density: %.4f" % (p/n/m))

count_u = df3.groupby(['user_id']).item_id.count()
print("min #actions per user: %.2f" % count_u.min())
print("max #actions per user: %.2f" % count_u.max())
print("ave #actions per user: %.2f" % count_u.mean())


# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================
# ================================================================================

user = pd.Series(list(set(df['user_id']).intersection(set(df2['user_id'])).intersection(set(df3['user_id']))))
print("same user: ", len(user))
items = pd.Series(list(set(df['item_id']).intersection(set(df2['item_id'])).intersection(set(df3['item_id']))))
print("same items: ", len(items))
print(items)

df4 = pd.concat([df, df2, df3], keys=['x', 'y', 'z'])
df4 = df4[df4['user_id'].isin(user)]

# renumber user ids and item ids
le = preprocessing.LabelEncoder()
df4['user_id'] = le.fit_transform(df4['user_id'])+1
df4['item_id'] = le.fit_transform(df4['item_id'])+1

# output statistical information
print("==== statistic of processed data (whole) ====")
n = len(df4.user_id.unique())
m = len(df4.item_id.unique())
p = len(df4)
print("#users: %d" % n)
print("#items: %d" % m)
print("#actions: %d" % p)
print("density: %.4f" % (p/n/m))

count_u = df4.groupby(['user_id']).item_id.count()
print("min #actions per user: %.2f" % count_u.min())
print("max #actions per user: %.2f" % count_u.max())
print("ave #actions per user: %.2f" % count_u.mean())

# ========================================================
dfx = df4.loc['x']
# output statistical information
print("==== statistic of processed data (whole) ====")
n = len(dfx.user_id.unique())
m = len(dfx.item_id.unique())
p = len(dfx)
print("#users: %d" % n)
print("#items: %d" % m)
print("#actions: %d" % p)
print("density: %.4f" % (p/n/m))

count_u = dfx.groupby(['user_id']).item_id.count()
print("min #actions per user: %.2f" % count_u.min())
print("max #actions per user: %.2f" % count_u.max())
print("ave #actions per user: %.2f" % count_u.mean())

# split data into test set, valid set and train set,
# adopting the leave-one-out evaluation for next-item recommendation task

# ========================================
# obtain possible records in test set
df_test = dfx.groupby(['user_id']).tail(1)

dfx.drop(df_test.index, axis='index', inplace=True)

# ========================================
# obtain possible records in valid set
df_valid = dfx.groupby(['user_id']).tail(1)

dfx.drop(df_valid.index, axis='index', inplace=True)

# ========================================
# drop cold-start items in valid set and test set
df_valid = df_valid[df_valid.item_id.isin(dfx.item_id)]
df_test = df_test[df_test.user_id.isin(df_valid.user_id) & (
    df_test.item_id.isin(dfx.item_id) | df_test.item_id.isin(df_valid.item_id))]

# output data file
df_valid.to_csv(processed_file_prefix_x + "valid.csv", header=False, index=False)
df_test.to_csv(processed_file_prefix_x + "test.csv", header=False, index=False)
dfx.to_csv(processed_file_prefix_x + "train.csv", header=False, index=False)

# output statistical information
print("==== statistic of processed data (split) ====")
print("#train_users: %d" % len(dfx.user_id.unique()))
print("#train_items: %d" % len(dfx.item_id.unique()))
print("#valid_users: %d" % len(df_valid.user_id.unique()))
print("#test_users: %d" % len(df_test.user_id.unique()))

# ========================================================
# For each user, randomly sample some negative items,
# and rank these items with the ground-truth item when testing or validation
df_concat = pd.concat([dfx, df_valid, df_test], axis='index')
sr_user2items = df_concat.groupby(['user_id']).item_id.unique()
df_negative = pd.DataFrame({'user_id': df_concat.user_id.unique()})

# ========================================
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
df_negative.to_csv(processed_file_prefix_x + "negative.csv", header=False, index=False)


# ========================================
# ========================================
# ========================================
# ========================================

dfy = df4.loc['y']
# output statistical information
print("==== statistic of processed data (whole) ====")
n = len(dfy.user_id.unique())
m = len(dfy.item_id.unique())
p = len(dfy)
print("#users: %d" % n)
print("#items: %d" % m)
print("#actions: %d" % p)
print("density: %.4f" % (p/n/m))

count_u = dfy.groupby(['user_id']).item_id.count()
print("min #actions per user: %.2f" % count_u.min())
print("max #actions per user: %.2f" % count_u.max())
print("ave #actions per user: %.2f" % count_u.mean())

# split data into test set, valid set and train set,
# adopting the leave-one-out evaluation for next-item recommendation task

# ========================================
# obtain possible records in test set
df_test = dfy.groupby(['user_id']).tail(1)

dfy.drop(df_test.index, axis='index', inplace=True)

# ========================================
# obtain possible records in valid set
df_valid = dfy.groupby(['user_id']).tail(1)

dfy.drop(df_valid.index, axis='index', inplace=True)

# ========================================
# drop cold-start items in valid set and test set
df_valid = df_valid[df_valid.item_id.isin(dfy.item_id)]
df_test = df_test[df_test.user_id.isin(df_valid.user_id) & (
    df_test.item_id.isin(dfy.item_id) | df_test.item_id.isin(df_valid.item_id))]

# output data file
df_valid.to_csv(processed_file_prefix_y + "valid.csv", header=False, index=False)
df_test.to_csv(processed_file_prefix_y + "test.csv", header=False, index=False)
dfy.to_csv(processed_file_prefix_y + "train.csv", header=False, index=False)

# output statistical information
print("==== statistic of processed data (split) ====")
print("#train_users: %d" % len(dfy.user_id.unique()))
print("#train_items: %d" % len(dfy.item_id.unique()))
print("#valid_users: %d" % len(df_valid.user_id.unique()))
print("#test_users: %d" % len(df_test.user_id.unique()))

# ========================================================
# For each user, randomly sample some negative items,
# and rank these items with the ground-truth item when testing or validation
df_concat = pd.concat([dfy, df_valid, df_test], axis='index')
sr_user2items = df_concat.groupby(['user_id']).item_id.unique()
df_negative = pd.DataFrame({'user_id': df_concat.user_id.unique()})

# ========================================
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
df_negative.to_csv(processed_file_prefix_y + "negative.csv", header=False, index=False)


# ========================================
# ========================================
# ========================================
# ========================================

# ========================================================
dfz = df4.loc['z']
# output statistical information
print("==== statistic of processed data (whole) ====")
n = len(dfz.user_id.unique())
m = len(dfz.item_id.unique())
p = len(dfz)
print("#users: %d" % n)
print("#items: %d" % m)
print("#actions: %d" % p)
print("density: %.4f" % (p/n/m))

count_u = dfz.groupby(['user_id']).item_id.count()
print("min #actions per user: %.2f" % count_u.min())
print("max #actions per user: %.2f" % count_u.max())
print("ave #actions per user: %.2f" % count_u.mean())

# split data into test set, valid set and train set,
# adopting the leave-one-out evaluation for next-item recommendation task

# ========================================
# obtain possible records in test set
df_test = dfz.groupby(['user_id']).tail(1)

dfz.drop(df_test.index, axis='index', inplace=True)

# ========================================
# obtain possible records in valid set
df_valid = dfz.groupby(['user_id']).tail(1)

dfz.drop(df_valid.index, axis='index', inplace=True)

# ========================================
# drop cold-start items in valid set and test set
df_valid = df_valid[df_valid.item_id.isin(dfz.item_id)]
df_test = df_test[df_test.user_id.isin(df_valid.user_id) & (
    df_test.item_id.isin(dfz.item_id) | df_test.item_id.isin(df_valid.item_id))]

# output data file
df_valid.to_csv(processed_file_prefix_z + "valid.csv", header=False, index=False)
df_test.to_csv(processed_file_prefix_z + "test.csv", header=False, index=False)
dfz.to_csv(processed_file_prefix_z + "train.csv", header=False, index=False)

# output statistical information
print("==== statistic of processed data (split) ====")
print("#train_users: %d" % len(dfz.user_id.unique()))
print("#train_items: %d" % len(dfz.item_id.unique()))
print("#valid_users: %d" % len(df_valid.user_id.unique()))
print("#test_users: %d" % len(df_test.user_id.unique()))

# ========================================================
# For each user, randomly sample some negative items,
# and rank these items with the ground-truth item when testing or validation
df_concat = pd.concat([dfz, df_valid, df_test], axis='index')
sr_user2items = df_concat.groupby(['user_id']).item_id.unique()
df_negative = pd.DataFrame({'user_id': df_concat.user_id.unique()})

# ========================================
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
df_negative.to_csv(processed_file_prefix_z + "negative.csv", header=False, index=False)


# ========================================
# ========================================
# ========================================
# ========================================

# split data into test set, valid set and train set,
# adopting the leave-one-out evaluation for next-item recommendation task

# ========================================
# obtain possible records in test set
df_test = df4.groupby(['user_id']).tail(1)

df4.drop(df_test.index, axis='index', inplace=True)

# ========================================
# obtain possible records in valid set
df_valid = df4.groupby(['user_id']).tail(1)

df4.drop(df_valid.index, axis='index', inplace=True)

# ========================================
# drop cold-start items in valid set and test set
df_valid = df_valid[df_valid.item_id.isin(df4.item_id)]
df_test = df_test[df_test.user_id.isin(df_valid.user_id) & (
    df_test.item_id.isin(df4.item_id) | df_test.item_id.isin(df_valid.item_id))]

# output data file  TODO
# df_valid.to_csv(processed_file_prefix + args.y + "valid.csv", header=False, index=False)
# df_test.to_csv(processed_file_prefix + args.y + "test.csv", header=False, index=False)
# df4.to_csv(processed_file_prefix + args.y + "train.csv", header=False, index=False)

# output statistical information
print("==== statistic of processed data (split) ====")
print("#train_users: %d" % len(df4.user_id.unique()))
print("#train_items: %d" % len(df4.item_id.unique()))
print("#valid_users: %d" % len(df_valid.user_id.unique()))
print("#test_users: %d" % len(df_test.user_id.unique()))


# ========================================================
# For each user, randomly sample some negative items,
# and rank these items with the ground-truth item when testing or validation
df_concat = pd.concat([df4, df_valid, df_test], axis='index')
sr_user2items = df_concat.groupby(['user_id']).item_id.unique()
df_negative = pd.DataFrame({'user_id': df_concat.user_id.unique()})

# ========================================
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

# output negative data  TODO
df_negative = pd.concat([df_negative, pd.DataFrame(list(arr_sample))], axis='columns')
# df_negative.to_csv(processed_file_prefix + args.y + "negative.csv", header=False, index=False)



