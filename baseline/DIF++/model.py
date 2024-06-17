import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from dif_layer import DIFMultiHeadAttention


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class multihead_attention(torch.nn.Module):
    def __init__(self, item_num, args):
        super(multihead_attention, self).__init__()

        self.dev = args.device
        self.num_blocks = args.num_blocks
        self.first_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.query_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.key_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.key_attention_layernorms = torch.nn.ModuleList()
        self.value_attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()

        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.layer_norm_eps = 1e-8

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            key_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.key_attention_layernorms.append(key_attn_layernorm)

            value_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.value_attention_layernorms.append(value_attn_layernorm)

            new_attn_layer = DIFMultiHeadAttention(args.num_heads, args.hidden_units, args.dropout_rate,
                                                args.dropout_rate, self.layer_norm_eps, args.maxlen)

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def get_attn_mask(self, attention_mask, queries):
        new_attn_mask = torch.zeros_like(attention_mask, dtype=queries.dtype)
        new_attn_mask.masked_fill_(attention_mask, float("-inf"))
        return new_attn_mask

    def seq_log2feats(self, queries, timeline_mask, position_embedding, feat_embedding):
        queries = self.emb_dropout(queries)
        queries *= ~timeline_mask.unsqueeze(-1)

        tl = queries.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        attention_mask = self.get_attn_mask(attention_mask, queries)

        for i in range(self.num_blocks):
            Q = self.attention_layernorms[i](queries)
            queries, attention_map = self.attention_layers[i](Q, queries, queries, attention_mask, position_embedding, feat_embedding)
            queries += Q
            queries = self.forward_layernorms[i](queries)
            queries = self.forward_layers[i](queries)
            queries *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(queries)
        return log_feats

    def forward(self, queries, timeline_mask, position_embedding, feat_embedding):
        return self.seq_log2feats(queries, timeline_mask, position_embedding, feat_embedding)


class Model(nn.Module):
    def __init__(self, itemnum, ItemMeanFeatures, MetaFeatures, args):
        super().__init__()
        self.args = args
        self.dev = args.device
        self.itemnum = itemnum
        self.hidden_units = args.hidden_units
        self.embedding_size = args.embedding_size
        self.review_emb = torch.tensor(ItemMeanFeatures, dtype=torch.float32, device=self.dev)
        self.meta_emb = torch.tensor(MetaFeatures, dtype=torch.float32, device=self.dev)
        self.item_emb = nn.Embedding(itemnum + 1, self.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)

        self.featLinear = nn.Linear(self.embedding_size, self.hidden_units)
        self.review_layer_norm = nn.LayerNorm(self.hidden_units)
        self.review_embed_dropout = nn.Dropout(args.dropout_rate)

        self.multihead_attention_layers = multihead_attention(itemnum, args)
        self.predictor = nn.Linear(self.hidden_units,  self.itemnum+1)

        self.attribute_loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        self.last_linear = torch.nn.Linear(2 * args.hidden_units, args.hidden_units)
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)


    def feat_emb(self, input_seq):
        seq_feat = self.review_emb[input_seq]
        seq_meta = self.meta_emb[input_seq]
        meta_feat = seq_feat + seq_meta
        meta_feat_linear = self.featLinear(meta_feat)
        meta_feat_linear = self.review_layer_norm(self.review_embed_dropout(meta_feat_linear))
        return meta_feat_linear

    def seq_self_attention_layer(self, targetSeq):
        target_timeline_mask = torch.BoolTensor(targetSeq == 0).to(self.dev)
        targetEmb = self.item_emb(torch.LongTensor(targetSeq).to(self.dev))  # 1 x item_num x h
        positions = np.tile(np.array(range(targetEmb.shape[1])), [targetEmb.shape[0], 1])
        position_embedding = self.pos_emb(torch.LongTensor(positions).to(self.dev))
        feat_embedding = self.feat_emb(torch.LongTensor(targetSeq).to(self.dev))
        targetSeqs = self.multihead_attention_layers(targetEmb, target_timeline_mask, position_embedding, feat_embedding)

        return targetSeqs

    def forward(self, targetSeq, pos, neg):
        seq_sasrec = self.seq_self_attention_layer(targetSeq)

        pos_item_in = self.item_emb(torch.LongTensor(pos).to(self.dev))
        neg_item_in = self.item_emb(torch.LongTensor(neg).to(self.dev))

        pos_logits = (seq_sasrec * pos_item_in).sum(dim=-1)
        neg_logits = (seq_sasrec * neg_item_in).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, targetSeq, test_item):
        seq_sasrec = self.seq_self_attention_layer(targetSeq)

        # embedding_weight = self.item_emb.weight
        test_item_in = self.item_emb(torch.LongTensor(test_item).to(self.dev))

        final_feat = seq_sasrec[:, -1, :]  # 1 x h
        test_logits = test_item_in.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return test_logits
