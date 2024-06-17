# -*- coding: utf-8 -*-
# @Time   : 2020/8/17 19:38
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# recbole version UPDATE:
# @Time   : 2020/8/19, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com
# https://recbole.io/docs/user_guide/model/sequential/gru4rec.html

# pytorch version
# @Time   : 2024/6/17

r"""
GRU4Rec
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""


import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, itemnum, args):
        super().__init__()
        self.args = args
        self.dev = args.device
        self.hidden_units = args.hidden_units

        self.item_emb = nn.Embedding(itemnum + 1, self.hidden_units)
        self.num_layers = args.num_layers

        self.gru_layers = nn.GRU(
            input_size=self.hidden_units,
            hidden_size=self.hidden_units,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.emb_dropout = nn.Dropout(args.dropout_rate)
        self.dense = nn.Linear(self.hidden_units, self.hidden_units)

    def gru4rec(self, item_seq):
        item_seq_emb = self.item_emb(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        return gru_output

    def forward(self, targetSeq, pos, neg):
        logits = self.gru4rec(torch.LongTensor(targetSeq).to(self.dev))
        pos_emb = self.item_emb(torch.LongTensor(pos).to(self.dev))
        neg_emb = self.item_emb(torch.LongTensor(neg).to(self.dev))

        pos_logits = (logits * pos_emb).sum(dim=-1)
        neg_logits = (logits * neg_emb).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, targetSeq, test_item):
        logits = self.gru4rec(torch.LongTensor(targetSeq).to(self.dev))

        # embedding_weight = self.item_emb.weight
        test_item_in = self.item_emb(torch.LongTensor(test_item).to(self.dev))

        final_feat = logits[:, -1, :]  # 1 x h
        test_logits = test_item_in.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return test_logits
