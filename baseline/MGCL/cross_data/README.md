# Data Preprocessing


We implement our preprocessing codes with an adaption from the published code of [FISSA](https://csse.szu.edu.cn/staff/panwk/publications/FISSA/). 
>[1]. Jing Lin, Weike Pan, and Zhong Ming. 2020. FISSA: Fusing Item Similarity Models with Self-Attention Networks for Sequential Recommendation. In Proceedings of the 14th ACM Conference on Recommender Systems (RecSys ’20). 130–139.

## Data
We download the original file of the datasets (i.e., ''Movies and TV'', ''CDs_and_Vinyl'', ''Books'') from: http://jmcauley.ucsd.edu/data/amazon/

## Process
The preprocessing procedure is as follows:
1) select positive implicit feedbacks, 
2) drop duplicated user-item pairs, 
3) discard cold-start (with interactions less than 5) items, then discard cold-start users,
4) discard users who do not interact in all the three domains,
5) renumber user ids and item ids, 
6) split out possible test and valid records, 
7) drop valid records with items that do not exist in train set, then drop test records with items that do not exist in train or valid set, 
8) sample 100 negative items according to their popularity for each user.

Notice that when evaluating model performance, we adopt the leave-one-out evaluation by splitting each sequence into three parts, i.e., the last interaction for test, the penultimate interaction for validation and the remaining interactions for training

## Statistical Details

| Dataset           | # Overlapped-Users  | # Items       | # Interactions    | Avg. Length     | Density   |
|:------------------|:--------            |:--------------|:------------------| :------         |:------    |
| Movie             | 10929               | 59513         | 460226            | 42.11           |0.07%      |
| CD                | 10929               | 91169         | 344221            | 31.50           |0.03%      |
| Book              | 10929               | 236049        | 607657            | 55.60           |0.02%      |


