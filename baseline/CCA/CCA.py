# -*- coding: utf-8 -*-

# v10_3
# 在v8的基础上进一步完善
# 把两个attention共享，其他不动

"""#UTILS"""

import sys
import copy
import random
import numpy as np
from collections import defaultdict
import pandas as pd
import pickle
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x = pickle.load(f)
    except:
        x = []
    return x


def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def data_partition(fname):
    usernum = 0
    itemnum1 = 0

    User = defaultdict(list)
    User1 = defaultdict(list)

    user_train1 = defaultdict(list)
    user_valid1 = defaultdict(list)
    user_test1 = defaultdict(list)


    user_map = dict()
    item_map = dict()

    user_ids = list()
    item_ids1 = list()


    # train data
    with open('./review_datasets/%s/%s_train.txt' % (fname, fname), 'r') as f:
        for line in f:
            u, i, t = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            t = int(t)
            user_ids.append(u)
            item_ids1.append(i)
            User[u].append(i)


    with open('./review_datasets/%s/%s_valid.txt' % (fname, fname), 'r') as f:
        for line in f:
            u, i, t= line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            t = int(t)
            user_ids.append(u)
            item_ids1.append(i)
            User[u].append(i)


    with open('./review_datasets/%s/%s_test.txt' % (fname, fname), 'r') as f:
        for line in f:
            u, i, t = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            t = int(t)
            user_ids.append(u)
            item_ids1.append(i)
            User[u].append(i)

    ItemFeatures = load_data('./review_datasets/%s/%s_review_emb_single.dat' % (fname, fname))
    ItemMeanFeatures = load_data('./review_datasets/%s/%s_review_emb_mean.dat' % (fname, fname))
    CXTdict = load_data('./review_datasets/%s/CXTDictSasRec_%s.dat' % (fname, fname))

    ItemFeatures1 = {}
    ItemMeanFeatures1 = {}
    CXTdict1 = {}

    for u in user_ids:
        if u not in user_map:
            user_map[u] = usernum + 1
            usernum += 1

    for i in item_ids1:
        if i not in item_map:
            item_map[i] = itemnum1 + 1
            ItemMeanFeatures1[item_map[i]] = ItemMeanFeatures[i]
            itemnum1 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User1[u].append(i)
            ItemFeatures1[(u, i)] = ItemFeatures[(user, item)]
            CXTdict1[(u, i)] = CXTdict[(user, item)]

    # neg
    neglist1 = defaultdict(list)
    user_neg1 = defaultdict(list)
    with open("./review_datasets/%s/%s_negative.csv" % (fname, fname), 'r') as f:
        for line in f:
            l = line.rstrip().split(',')
            u = user_map[int(l[0])]
            for j in range(1, 101):
                i = item_map[int(l[j])]
                neglist1[u].append(i)

    for user in User1:
        nfeedback = len(User1[user])
        if nfeedback < 3:
            user_train1[user] = User1[user]
            user_valid1[user] = []
            user_test1[user] = []
        else:
            user_train1[user] = User1[user][:-2]
            user_valid1[user] = []
            user_valid1[user].append(User1[user][-2])
            user_test1[user] = []
            user_test1[user].append(User1[user][-1])
        user_neg1[user] = neglist1[user]

    return [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1, ItemFeatures1, ItemMeanFeatures1, CXTdict1]


# valid
def evaluate_valid(model, dataset, args, sess, cxtdict, cxtsize, embedding_size, negnum=100):
    [train, valid, test, usernum, itemnum, user_neg, ItemFeatures, itemFeat, CXTdict] = dataset

    NDCG10 = 0.0
    valid_user = 0.0
    HT10 = 0.0

    NDCG5 = 0.0
    HT5 = 0.0

    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue  # 这句可排除在时间线左边的序列

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seqcxt = np.zeros([args.maxlen, cxtsize], dtype=np.float32)
        seqfeat_single = np.zeros([args.maxlen, embedding_size], dtype=np.float32)
        seqfeat_mean = np.zeros([args.maxlen, embedding_size], dtype=np.float32)
        # seqfeat = np.zeros([args.maxlen,embedding_size],dtype=np.float32)
        testitemscxt = []
        testitemsfeat = []

        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            # cxt
            seqcxt[idx] = CXTdict[(u, i)]
            seqfeat_mean[idx] = np.array(itemFeat[i]).tolist()
            seqfeat_single[idx] = np.array(ItemFeatures[(u, i)]).tolist()
            # seqfeat[idx] = (emb_mean*args.weight + emb_single*(1-args.weight)).tolist()

            idx -= 1
            if idx == -1: break

        item_idx = [valid[u][0]]
        testitemscxt.append(CXTdict[(u, valid[u][0])])
        testitemsfeat.append(itemFeat[valid[u][0]])
        # testitemsfeat.append(feat_valid[(u,valid[u][0])])

        for t in user_neg[u]:
            item_idx.append(t)
            testitemscxt.append(CXTdict[(u, valid[u][0])])
            testitemsfeat.append(itemFeat[t])

        predictions = model.predict(sess, np.ones(args.maxlen) * u, [seq], item_idx, [seqcxt], [seqfeat_single],
                                    [seqfeat_mean], testitemscxt, testitemsfeat)

        predictions = -predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG10 += 1 / np.log2(rank + 2)
            HT10 += 1

        if rank < 5:
            NDCG5 += 1 / np.log2(rank + 2)
            HT5 += 1

    return NDCG10 / valid_user, HT10 / valid_user, NDCG5 / valid_user, HT5 / valid_user



# test
def evaluate_test(model, dataset, args, sess, cxtdict, cxtsize, embedding_size, negnum=100):
    [train, valid, test, usernum, itemnum, user_neg, ItemFeatures, itemFeat, CXTdict] = dataset

    NDCG10 = 0.0
    valid_user = 0.0
    HT10 = 0.0

    NDCG5 = 0.0
    HT5 = 0.0

    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue  # 这句可排除在时间线左边的序列

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seqcxt = np.zeros([args.maxlen, cxtsize], dtype=np.float32)
        seqfeat_single = np.zeros([args.maxlen, embedding_size], dtype=np.float32)
        seqfeat_mean = np.zeros([args.maxlen, embedding_size], dtype=np.float32)
        # seqfeat = np.zeros([args.maxlen,embedding_size],dtype=np.float32)
        testitemscxt = []
        testitemsfeat = []

        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            # cxt
            seqcxt[idx] = CXTdict[(u, i)]
            seqfeat_mean[idx] = np.array(itemFeat[i]).tolist()
            seqfeat_single[idx] = np.array(ItemFeatures[(u, i)]).tolist()
            # seqfeat[idx] = (emb_mean*args.weight + emb_single*(1-args.weight)).tolist()

            idx -= 1
            if idx == -1: break

        item_idx = [test[u][0]]
        testitemscxt.append(CXTdict[(u, test[u][0])])
        testitemsfeat.append(itemFeat[test[u][0]])
        # testitemsfeat.append(feat_valid[(u,valid[u][0])])

        for t in user_neg[u]:
            item_idx.append(t)
            testitemscxt.append(CXTdict[(u, test[u][0])])
            testitemsfeat.append(itemFeat[t])

        predictions = model.predict(sess, np.ones(args.maxlen) * u, [seq], item_idx, [seqcxt], [seqfeat_single],
                                    [seqfeat_mean], testitemscxt, testitemsfeat)

        predictions = -predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG10 += 1 / np.log2(rank + 2)
            HT10 += 1

        if rank < 5:
            NDCG5 += 1 / np.log2(rank + 2)
            HT5 += 1

    return NDCG10 / valid_user, HT10 / valid_user, NDCG5 / valid_user, HT5 / valid_user


"""#Sampler"""

import numpy as np
import random
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, cxtdict, cxtsize, ItemFeatures, itemFeat, embedding_size, batch_size,
                    maxlen, result_queue, SEED):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        ###CXT
        seqcxt = np.zeros([maxlen, cxtsize], dtype=np.float32)
        poscxt = np.zeros([maxlen, cxtsize], dtype=np.float32)
        negcxt = np.zeros([maxlen, cxtsize], dtype=np.float32)
        ###feat_train
        seqFeat_single = np.zeros([maxlen, embedding_size], dtype=np.float32)
        seqFeat_mean = np.zeros([maxlen, embedding_size], dtype=np.float32)
        # seqFeat = np.zeros([maxlen,embedding_size], dtype=np.float32)
        posFeat = np.zeros([maxlen, embedding_size], dtype=np.float32)
        negFeat = np.zeros([maxlen, embedding_size], dtype=np.float32)

        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = list(set(user_train[user]))
        # keys = np.array(list(feat_train.keys()))  很耗时
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg_i = 0
            if nxt != 0:
                neg_i = random_neq(1, itemnum + 1, ts)
                negFeat[idx] = itemFeat[neg_i]
                neg[idx] = neg_i

            ###CXT
            seqcxt[idx] = cxtdict[(user, i)]
            poscxt[idx] = cxtdict[(user, nxt)]
            negcxt[idx] = cxtdict[(user, nxt)]
            ###feat_train
            seqFeat_mean[idx] = np.array(itemFeat[i])
            seqFeat_single[idx] = np.array(ItemFeatures[(user, i)])

            posFeat[idx] = itemFeat[nxt]
            # negFeat[idx] = np.zeros([embedding_size], dtype=np.float32)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (
            np.ones(maxlen) * user, seq, pos, neg, seqcxt, poscxt, negcxt, seqFeat_single, seqFeat_mean, posFeat,
            negFeat)

    # np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, cxtdict, cxtsize, ItemFeatures, itemFeat, embedding_size, batch_size=64,
                 maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        # keys = np.array(list(cxtdict.keys()))
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      cxtdict,
                                                      cxtsize,
                                                      ItemFeatures,
                                                      itemFeat,
                                                      embedding_size,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


"""#Modules"""

'''
Modified version of the original code by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

import tensorflow as tf
import numpy as np


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       # initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        res=True,
                        with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.leaky_relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.leaky_relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.leaky_relu)  # (N, T_k, C)
        # Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        if res:
            outputs *= queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def multihead_attention2(queries,
                         keys,
                         num_units=None,
                         num_heads=8,
                         dropout_rate=0,
                         is_training=True,
                         causality=False,
                         scope="multihead_attention",
                         reuse=None,
                         res=True,
                         with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.leaky_relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.leaky_relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.leaky_relu)  # (N, T_k, C)
        # Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        if res:
            outputs *= queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def multihead_attention_sasrec(queries,
                               keys,
                               num_units=None,
                               num_heads=8,
                               dropout_rate=0,
                               is_training=True,
                               causality=False,
                               scope="multihead_attention",
                               reuse=None,
                               res=True,
                               with_qk=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.leaky_relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.leaky_relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.leaky_relu)  # (N, T_k, C)
        # Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale   scaled_dot
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        if res:
            outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.leaky_relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = normalize(outputs)

    return outputs


"""#Model"""


class Model():
    def __init__(self, usernum, itemnum, args, embedding_size, UserFeatures=None, cxt_size=None, reuse=None,
                 use_res=False):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.seq_cxt = tf.placeholder(tf.float32, shape=(None, args.maxlen, cxt_size))
        self.pos_cxt = tf.placeholder(tf.float32, shape=(None, args.maxlen, cxt_size))
        self.neg_cxt = tf.placeholder(tf.float32, shape=(None, args.maxlen, cxt_size))
        self.seq_feat_single = tf.placeholder(tf.float32, shape=(None, args.maxlen, embedding_size))
        self.seq_feat_mean = tf.placeholder(tf.float32, shape=(None, args.maxlen, embedding_size))
        # self.seq_feat = tf.placeholder(tf.float32, shape=(None, args.maxlen, embedding_size))
        self.pos_feat_in = tf.placeholder(tf.float32, shape=(None, args.maxlen, embedding_size))
        self.neg_feat_in = tf.placeholder(tf.float32, shape=(None, args.maxlen, embedding_size))

        # self.ItemFeats = tf.constant(ItemFeatures,name="ItemFeats", shape=[itemnum + 1, ItemFeatures.shape[1]],dtype=tf.float32)
        # self.UserFeats = tf.constant(UserFeatures,name="UserFeats", shape=[usernum + 1, UserFeatures.shape[1]],dtype=tf.float32)

        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        # sequence embedding, item embedding table
        self.seq_in, item_emb_table = embedding(self.input_seq,
                                                vocab_size=itemnum + 1,
                                                num_units=args.hidden_units,
                                                zero_pad=True,
                                                scale=True,
                                                l2_reg=args.l2_emb,
                                                scope="input_embeddings",
                                                with_t=True,
                                                reuse=reuse
                                                )
        # gate
        W_s = tf.Variable(
            tf.random.normal(shape=(embedding_size, embedding_size), stddev=0.01, mean=0, dtype=tf.float32))  # 维度问题
        W_m = tf.Variable(
            tf.random.normal(shape=(embedding_size, embedding_size), stddev=0.01, mean=0, dtype=tf.float32))
        b_g = tf.Variable(tf.random.normal(shape=(args.maxlen, embedding_size), stddev=0.01, mean=0, dtype=tf.float32))
        one = tf.constant(1.0)

        g = tf.sigmoid(tf.matmul(self.seq_feat_single, W_s) + tf.matmul(self.seq_feat_mean, W_m) + b_g)
        self.seq_feat = tf.multiply(g, self.seq_feat_single) + tf.multiply((one - g), self.seq_feat_mean)

        # Cxt
        # seq_feat = tf.reshape(self.u, [tf.shape(self.input_seq)[0] * args.maxlen]) #(128 x 200) x 1
        self.seq_feat_in = tf.concat([self.seq_feat, self.seq_cxt], -1)
        # cxt
        self.seq_feat_emb = tf.layers.dense(inputs=self.seq_feat_in, units=args.hidden_units * 5, activation=None,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            name="feat_emb")

        # Positional Encoding
        t, pos_emb_table = embedding(
            tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
            vocab_size=args.maxlen,
            num_units=args.hidden_units,
            zero_pad=False,
            scale=False,
            l2_reg=args.l2_emb,
            scope="dec_pos",
            reuse=reuse,
            with_t=True
        )

        #### Features Part
        self.seq_concat = tf.concat([self.seq_in, self.seq_feat_emb], 2)
        self.seq = tf.layers.dense(inputs=self.seq_concat, units=args.hidden_units, activation=None,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='embComp')
        #### Features Part
        #### Cxt part

        ####  add position embedding
        # self.seq += t

        # Dropout
        self.seq = tf.layers.dropout(self.seq,
                                     rate=args.dropout_rate,
                                     training=tf.convert_to_tensor(self.is_training))
        self.seq *= mask

        # SASRec block
        # self.seq_sasrec = tf.tile(self.seq,[1,1,1])

        for i in range(args.num_blocks):
            with tf.variable_scope("num_blocks_shared_%d" % i):
                # Self-attention
                self.seq = multihead_attention_sasrec(queries=normalize(self.seq),
                                                      keys=self.seq,
                                                      num_units=args.hidden_units,
                                                      num_heads=args.num_heads,
                                                      dropout_rate=args.dropout_rate,
                                                      is_training=self.is_training,
                                                      causality=True,
                                                      scope="self_attention")

                # Feed forward
                self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                       dropout_rate=args.dropout_rate, is_training=self.is_training)
                self.seq *= mask

        self.seq = normalize(self.seq)

        self.seq_sasrec = tf.tile(self.seq, [1, 1, 1])


        pos_emb_in = tf.nn.embedding_lookup(item_emb_table, pos)  # (128 x 200) x h
        neg_emb_in = tf.nn.embedding_lookup(item_emb_table, neg)  # (128 x 200) x h

        seq_emb_train = self.seq  # 128 x 200 x h
        seq_emb_test = self.seq  # 128 x 200 x h


        pos_feat = tf.concat([self.pos_feat_in, self.pos_cxt], -1)  # (128 x 200) x h
        ##
        pos_feat_emb = tf.layers.dense(inputs=pos_feat, reuse=True, units=args.hidden_units * 5, activation=None,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="feat_emb")
        pos_emb_con = tf.concat([pos_emb_in, pos_feat_emb], -1)
        pos_emb = tf.layers.dense(inputs=pos_emb_con, reuse=True, units=args.hidden_units, activation=None,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  name='embComp')  # 128 x 200 x h


        ##cxt
        neg_feat = tf.concat([self.neg_feat_in, self.neg_cxt], -1)
        ##
        neg_feat_emb = tf.layers.dense(inputs=neg_feat, reuse=True, units=args.hidden_units * 5, activation=None,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01), name="feat_emb")
        neg_emb_con = tf.concat([neg_emb_in, neg_feat_emb], -1)
        neg_emb = tf.layers.dense(inputs=neg_emb_con, reuse=True, units=args.hidden_units, activation=None,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  name='embComp')  # 128 x 200 x h

        # neg_emb = tf.multiply(neg_emb,user_emb)
        #### Features Part

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        self.test_item_cxt = tf.placeholder(tf.float32, shape=(101, cxt_size))
        self.test_feat_in = tf.placeholder(tf.float32, shape=(101, embedding_size))

        test_item_resh = tf.reshape(self.test_item, [1, 101])
        test_item_cxt_resh = tf.reshape(self.test_item_cxt, [1, 101, cxt_size])  # 1 x 101 x 6
        test_item_feat_resh = tf.reshape(self.test_feat_in, [1, 101, embedding_size])  # 1 x itemnum x 768

        self.test_item_emb_in = tf.nn.embedding_lookup(item_emb_table, test_item_resh)  # 1 x 101 x h

        ########### Test user
        self.test_user = tf.placeholder(tf.int32, shape=(args.maxlen))

        ##cxt
        test_feat_con = tf.concat([test_item_feat_resh, test_item_cxt_resh], -1)  # 1 x 101 x f + 6
        ##
        self.test_feat_emb = tf.layers.dense(inputs=test_feat_con, reuse=True, units=args.hidden_units * 5,
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                             name="feat_emb")  # 1 x 101 x 5h
        test_item_emb_con = tf.concat([self.test_item_emb_in, self.test_feat_emb], -1)  # 1 x 101 x 6h
        self.test_item_emb = tf.layers.dense(inputs=test_item_emb_con, reuse=True, units=args.hidden_units,
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                             name='embComp')  # 1 x 101 x h


        #### Features Part
        mask_pos = tf.expand_dims(tf.to_float(tf.not_equal(self.pos, 0)), -1)
        mask_neg = tf.expand_dims(tf.to_float(tf.not_equal(self.neg, 0)), -1)

        self.test_logits_carca = None
        for i in range(1):
            with tf.variable_scope("num_blocks_p_%d" % i):
                # Self-attentions, # 1 x 200 x h
                # Self-attentions, # 1 x 101 x h
                self.test_logits_carca = multihead_attention2(queries=self.test_item_emb,
                                                              keys=seq_emb_test,
                                                              num_units=args.hidden_units,
                                                              num_heads=args.num_heads,
                                                              dropout_rate=args.dropout_rate,
                                                              is_training=self.is_training,
                                                              causality=False,
                                                              res=use_res,
                                                              scope="self_attention")

                # Feed forward , # 1 x 101 x h
                # self.test_logits_carca = feedforward(self.test_logits_carca, num_units=[args.hidden_units, args.hidden_units], dropout_rate=args.dropout_rate, is_training=self.is_training)

        # self.test_logits_carca = normalize(self.test_logits_carca)
        ##Without User
        self.test_logits_carca = tf.layers.dense(inputs=self.test_logits_carca, units=1, activation=None,
                                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                 name='logit')  # 1 x 101 x 1
        self.test_logits_carca = tf.reshape(self.test_logits_carca, [1, 101], name="Reshape_pos")  # itemnum x 1

        # carca   seq_sasrec   (128, 75, h )
        outputs_te = tf.expand_dims(self.seq_sasrec[:, -1, :], 1)  # (1, 1, h )
        self.test_logits_sasrec = tf.reduce_sum(outputs_te * self.test_item_emb, -1)  # (1, 101)

        # final output
        self.test_logits = self.test_logits_carca + self.test_logits_sasrec

        ## prediction layer
        ############################################################################
        self.pos_logits = None
        self.neg_logits = None
        for i in range(1):
            with tf.variable_scope("num_blocks_p_%d" % i):
                # Self-attentions, # 128 x 200 x 1
                self.pos_logits = multihead_attention2(queries=pos_emb,
                                                       keys=seq_emb_train,
                                                       num_units=args.hidden_units,
                                                       num_heads=args.num_heads,
                                                       dropout_rate=args.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=False,
                                                       reuse=True,
                                                       res=use_res,
                                                       scope="self_attention")

                # Feed forward , # 128 x 200 x 1
                # self.pos_logits = feedforward(normalize(self.pos_logits), num_units=[args.hidden_units, args.hidden_units], dropout_rate=args.dropout_rate, is_training=self.is_training,reuse=True)
                self.pos_logits *= mask_pos

        # self.pos_logits = normalize(self.pos_logits)
        for i in range(1):
            with tf.variable_scope("num_blocks_p_%d" % i):
                # Self-attentions
                self.neg_logits = multihead_attention2(queries=neg_emb,
                                                       keys=seq_emb_train,
                                                       num_units=args.hidden_units,
                                                       num_heads=args.num_heads,
                                                       dropout_rate=args.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=False,
                                                       reuse=True,
                                                       res=use_res,
                                                       scope="self_attention")

                # Feed forward  # 128 x 200 x 1
                # self.neg_logits = feedforward(normalize(self.neg_logits), num_units=[args.hidden_units, args.hidden_units], dropout_rate=args.dropout_rate, is_training=self.is_training,reuse=True)
                self.neg_logits *= mask_neg

                # sasrec prediction
        # self.neg_logits = normalize(self.neg_logits)
        outputs_pos = self.seq_sasrec  # (?,75,90)
        outputs_neg = self.seq_sasrec
        positive_rating = tf.reduce_sum(outputs_pos * pos_emb, -1)  # (128,75)
        negative_rating = tf.reduce_sum(outputs_neg * neg_emb, -1)


        self.pos_logits_sasrec = tf.reshape(positive_rating, [tf.shape(self.input_seq)[0] * args.maxlen],
                                            name="Reshape_pos")  # 128 x 200 x 1=> (128 x 200) x 1
        self.neg_logits_sasrec = tf.reshape(negative_rating, [tf.shape(self.input_seq)[0] * args.maxlen],
                                            name="Reshape_neg")  # 128 x 200 x 1=> (128 x 200) x 1

        print('type', type(positive_rating))

        # carca
        self.pos_logits = tf.layers.dense(inputs=self.pos_logits, reuse=True, units=1, activation=None,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                          name='logit')  # (128, 75, 1)
        self.neg_logits = tf.layers.dense(inputs=self.neg_logits, reuse=True, units=1, activation=None,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01), name='logit')
        # tf.reduce_sum(pos_emb * seq_emb_train, -1)

        self.pos_logits = tf.reshape(self.pos_logits, [tf.shape(self.input_seq)[0] * args.maxlen],
                                     name="Reshape_pos")  # 128 x 200 x 1=> (128 x 200) x 1    (128*75)
        self.neg_logits = tf.reshape(self.neg_logits, [tf.shape(self.input_seq)[0] * args.maxlen],
                                     name="Reshape_neg")  # 128 x 200 x 1=> (128 x 200) x 1

        # print('pos_logits shape:',self.pos_logits.shape)

        ###########################################################################

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss_carca = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses_carca = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss_carca += sum(reg_losses_carca)

        # sasrec loss
        self.loss_sasrec = tf.reduce_sum(- tf.log(tf.sigmoid(self.pos_logits_sasrec) + 1e-24) * istarget
                                         - tf.log(
            1 - tf.sigmoid(self.neg_logits_sasrec) + 1e-24) * istarget) / tf.reduce_sum(istarget)
        reg_loss_sasrec = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss_sasrec += sum(reg_loss_sasrec)

        # self.loss = 1*self.loss_sasrec
        self.loss = self.loss_sasrec + self.loss_carca

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx, seqcxt, seqfeat_single, seqfeat_mean, testitemcxt, testitemsfeat):
        return sess.run(self.test_logits,
                        {self.test_user: u, self.input_seq: seq, self.test_item: item_idx, \
                         self.is_training: False, self.seq_cxt: seqcxt, self.seq_feat_single: seqfeat_single, \
                         self.seq_feat_mean: seqfeat_mean, self.test_item_cxt: testitemcxt,
                         self.test_feat_in: testitemsfeat})


"""#Main"""

import os
import time
import argparse
import sys
import datetime
import random
import tensorflow as tf
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Configurations.")
    parser.add_argument('--dataset_name', type=str, default='Movies_and_TV', help='Name of the dataset (e.g. Steam).')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--maxlen', type=int, default=50, help='Maximum length of sequences.')
    parser.add_argument('--hidden_units', type=int, default=64, help='i.e. latent vector dimensionality.')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of self-attention blocks.')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads for attention.')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--l2_emb', type=float, default=0.01)
    parser.add_argument('--cxt_size', type=int, default=6)
    parser.add_argument('--use_res', type=bool, default=True)
    parser.add_argument('--random_seed', type=int, default=2024)
    parser.add_argument('--embedding_size', type=int, default=768, help='the embedding size of review')
    parser.add_argument('--early_stop_epoch', type=int, default=50)
    parser.add_argument('--save_model', type=int, default=1, help='Whether to save the tensorflow model.')
    parser.add_argument('--eva_interval', type=int, default=10, help='Number of epoch interval for evaluation.')
    parser.add_argument('--is_test', type=int, default=1)
    parser.add_argument('--model_num', type=int, default=1, help='')
    return parser.parse_args()


class Args:
    def __init__(self):

        self.dataset_name = 'Video_Games'
        self.train_dir = 'default'
        self.batch_size = 128
        self.lr = 0.001
        self.maxlen = 75
        self.hidden_units = 90
        self.num_blocks = 2
        self.num_epochs = 1000
        self.num_heads = 3  #
        self.dropout_rate = 0.5
        self.l2_emb = 0.0
        self.cxt_size = 6
        self.use_res = True
        self.random_seed = 0
        self.embedding_size = 768  # review embedding size
        self.early_stop_epoch = 100
        self.log_file = 'log_review_%s_v10_3.txt' % self.dataset_name
        self.save_model = 0
        self.is_test = 1
        self.eva_interval = 10


class Trainer(object):
    def __init__(self) -> None:
        super(Trainer, self).__init__()

        # self.args = Args()
        self.args = parse_args()
        # self.log_file = 'log_review_%s_v10_3.txt'%self.args.dataset_name
        self.log_file = 'log_review_%s_v10_3_%s.txt' % (self.args.dataset_name, self.args.model_num)

        with open(self.log_file, 'a') as f:
            print('\n--------------------------------------', file=f)
            print('args', file=f)
            print('\n'.join([str(k) + ': ' + str(v) for k, v in sorted(vars(self.args).items(), key=lambda x: x[0])]),
                  file=f)
            print('\n'.join([str(k) + ': ' + str(v) for k, v in sorted(vars(self.args).items(), key=lambda x: x[0])]))
            print('--------------------------------------\n', file=f)

    def train(self):
        dataset = data_partition(self.args.dataset_name)
        # [user_train, user_valid, user_test, itemFeat, ItemFeatures, candidate, usernum, itemnum] = dataset
        [user_train, user_valid, user_test, usernum, itemnum, user_neg, ItemFeatures, itemFeat, CXTdict] = dataset

        num_batch = len(user_train) / self.args.batch_size
        print(usernum, '--', itemnum)
        UserFeatures = None

        # CXTDict = load_data('./review_datasets/%s/CXTDictSasRec_%s.dat' % (self.args.dataset_name, self.args.dataset_name))

        random.seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        tf.set_random_seed(self.args.random_seed)

        sampler = WarpSampler(user_train, usernum, itemnum, CXTdict, self.args.cxt_size, ItemFeatures, itemFeat,
                              self.args.embedding_size, batch_size=self.args.batch_size, maxlen=self.args.maxlen, n_workers=3)

        # tf.compat.v1.reset_default_graph()
        model = Model(usernum, itemnum, self.args, self.args.embedding_size, UserFeatures, self.args.cxt_size,
                      use_res=self.args.use_res)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        T = 0.0
        t0 = time.time()

        best_hr10 = -1
        count = 0

        for epoch in range(1, self.args.num_epochs + 1):
            print("epoch {}:".format(epoch))
            for step in tqdm(range(int(num_batch)), total=int(num_batch), ncols=70, leave=False, unit='b'):
                u, seq, pos, neg, seqcxt, poscxt, negcxt, seqFeat_single, seqFeat_mean, posFeat, negFeat = sampler.next_batch()
                # print('seq shape:',np.array(seq).shape)
                # print('seqcxt shape:',np.array(seqcxt).shape)

                auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                        {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                         model.is_training: True, model.seq_cxt: seqcxt, model.pos_cxt: poscxt,
                                         model.neg_cxt: negcxt, \
                                         model.seq_feat_single: seqFeat_single, model.seq_feat_mean: seqFeat_mean,
                                         model.pos_feat_in: posFeat, model.neg_feat_in: negFeat})

            if epoch % self.args.eva_interval == 0:
                t1 = time.time() - t0
                T += t1

                t_valid = evaluate_valid(model, dataset, self.args, sess, CXTdict, self.args.cxt_size,
                                         self.args.embedding_size)
                print(
                    'epoch:%d, valid ( HR@10: %.4f, NDCG@10: %.4f, HR@5: %.4f, NDCG@5: %.4f)), time: %f(s)' % (epoch, t_valid[1], t_valid[0],t_valid[3], t_valid[2], T))

                t0 = time.time()

                with open(self.log_file, 'a') as f:
                    print(
                        'epoch:%d, valid ( HR@10: %.4f, NDCG@10: %.4f, HR@5: %.4f, NDCG@5: %.4f)), time: %f(s)' % (
                        epoch, t_valid[1], t_valid[0], t_valid[3], t_valid[2], T), file=f)

                t_test = evaluate_test(model, dataset, self.args, sess, CXTdict, self.args.cxt_size,
                                       self.args.embedding_size)

                print('test (HR@10: %.4f, NDCG@10: %.4f, HR@5: %.4f, NDCG@5: %.4f)' % (t_test[1], t_test[0], t_test[3], t_test[2]))

                with open(self.log_file, 'a') as f:
                    print('test (HR@10: %.4f, NDCG@10: %.4f, HR@5: %.4f, NDCG@5: %.4f)' % (t_test[1], t_test[0], t_test[3], t_test[2]),file=f)

                if t_valid[1] > best_hr10:
                    best_hr10 = t_valid[1]
                    count = 0

                    # if self.args.save_model == 1:
                    #     nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    #     model_saver = tf.compat.v1.train.Saver()
                    #     self.save_fname = (
                    #         'CARCA_v10_3_{}_lr{}_l2{}_units{}_heads{}_blocks{}_dropout{}_seed{}_maxlen{}_model{}'.format(
                    #             self.args.dataset_name, self.args.lr, self.args.l2_emb, self.args.hidden_units,
                    #             self.args.num_heads, self.args.num_blocks,
                    #             self.args.dropout_rate, self.args.random_seed, self.args.maxlen, self.args.model_num))
                    #
                    #     # self.save_fname = ('CARCA_v10_3_{}_lr{}_l2{}_units{}_heads{}_blocks{}_dropout{}_seed{}_maxlen{}_epoch{}_hr10_{}_{}'.format(
                    #     #     self.args.dataset_name,self.args.lr,self.args.l2_emb,self.args.hidden_units,self.args.num_heads,self.args.num_blocks,
                    #     #     self.args.dropout_rate,self.args.random_seed,self.args.maxlen,epoch,best_hr10,nowTime))
                    #     model_saver.save(sess,
                    #                      'Model/%s_test/model_' % self.args.dataset_name + self.save_fname + '.ckpt')

                else:
                    count += 10
                if count == self.args.early_stop_epoch:
                    break

        if self.args.is_test == 1:
            print('start test!')
            # model_saver.restore(sess, 'Model/%s_test/model_' % self.args.dataset_name + self.save_fname + '.ckpt')
            t_test = evaluate_test(model, dataset, self.args, sess, CXTdict, self.args.cxt_size,
                                   self.args.embedding_size)

            print('test (HR@10: %.4f, NDCG@10: %.4f)' % (t_test[1], t_test[0]))

            with open(self.log_file, 'a') as f:
                print('test (HR@10: %.4f, NDCG@10: %.4f)' % (t_test[1], t_test[0]), file=f)
            sampler.close()


trainer = Trainer()
trainer.train()
