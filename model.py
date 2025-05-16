import torch
import numpy as np
import torch.nn as nn

from layer import MultiHeadAttention


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


class Shared_Transformer(torch.nn.Module):
    def __init__(self, item_num, args):
        super(Shared_Transformer, self).__init__()
        self.dev = args.device
        self.num_blocks = args.num_blocks
        self.layer_norm_eps = args.eps

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.query_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.key_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.query_attention_layernorms = torch.nn.ModuleList()
        self.key_attention_layernorms = torch.nn.ModuleList()
        self.value_attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()

        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
            self.attention_layernorms.append(new_attn_layernorm)

            query_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
            self.query_attention_layernorms.append(query_attn_layernorm)

            key_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
            self.key_attention_layernorms.append(key_attn_layernorm)

            value_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
            self.value_attention_layernorms.append(value_attn_layernorm)

            new_attn_layer = MultiHeadAttention(args.num_heads, args.hidden_units, args.dropout_rate,
                                                args.dropout_rate, self.layer_norm_eps, args.maxlen)

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def get_attn_mask(self, attention_mask, queries):
        new_attn_mask = torch.zeros_like(attention_mask, dtype=queries.dtype)
        new_attn_mask.masked_fill_(attention_mask, float("-inf"))
        return new_attn_mask

    def forward(self, queries, timeline_mask, attention_map_list=None):
        queries = self.emb_dropout(queries)
        queries *= ~timeline_mask.unsqueeze(-1)

        tl = queries.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        attention_mask = self.get_attn_mask(attention_mask, queries)

        for i in range(self.num_blocks):
            Q = self.attention_layernorms[i](queries)
            attribute_attention_map = None
            if attention_map_list is not None and i == 0:
                attribute_attention_map = attention_map_list[i]
            queries, attention_map = self.attention_layers[i](Q, queries, queries, attention_mask,
                                                                  attribute_attention_map)
            queries += Q
            queries = self.forward_layernorms[i](queries)
            queries = self.forward_layers[i](queries)
            queries *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(queries)
        return log_feats

class Text_aware_Transformer(torch.nn.Module):
    def __init__(self, item_num, args):
        super(Text_aware_Transformer, self).__init__()

        self.dev = args.device
        self.num_blocks = args.num_blocks
        self.layer_norm_eps = args.eps
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.query_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.key_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.query_attention_layernorms = torch.nn.ModuleList()
        self.key_attention_layernorms = torch.nn.ModuleList()
        self.value_attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()

        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
            self.attention_layernorms.append(new_attn_layernorm)

            query_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
            self.query_attention_layernorms.append(query_attn_layernorm)

            key_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
            self.key_attention_layernorms.append(key_attn_layernorm)

            value_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
            self.value_attention_layernorms.append(value_attn_layernorm)

            new_attn_layer = MultiHeadAttention(args.num_heads, args.hidden_units, args.dropout_rate,
                                                args.dropout_rate, self.layer_norm_eps, args.maxlen)

            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=self.layer_norm_eps)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def get_attn_mask(self, attention_mask, queries):
        new_attn_mask = torch.zeros_like(attention_mask, dtype=queries.dtype)
        new_attn_mask.masked_fill_(attention_mask, float("-inf"))
        return new_attn_mask

    def forward(self, queries, timeline_mask, key=None, value=None, source_timeline_mask=None):
        queries *= ~timeline_mask.unsqueeze(-1)
        key *= ~source_timeline_mask.unsqueeze(-1)
        value *= ~source_timeline_mask.unsqueeze(-1)

        tl = queries.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        attention_mask = self.get_attn_mask(attention_mask, queries)

        attention_map_list = []

        for i in range(self.num_blocks):
            Q = self.query_attention_layernorms[i](queries)
            K = self.key_attention_layernorms[i](key)
            V = self.value_attention_layernorms[i](value)
            queries, attention_map = self.attention_layers[i](Q, K, V, attention_mask)
            queries += K
            queries = self.forward_layernorms[i](queries)
            queries = self.forward_layers[i](queries)
            attention_map_list.append(attention_map)
            queries *= ~source_timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(queries)
        return log_feats, attention_map_list


def SSL(sess_emb_hgnn, sess_emb_lgcn):
    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
        return corrupted_embedding

    def score(x1, x2):
        return torch.sum(torch.mul(x1, x2), 2)

    pos = score(sess_emb_hgnn, sess_emb_lgcn)
    neg1 = score(sess_emb_hgnn, row_column_shuffle(sess_emb_lgcn))
    one = torch.ones((neg1.shape[0], neg1.shape[1])).cuda()
    con_loss = torch.sum(-torch.log(1e-7 + torch.sigmoid(pos)) - torch.log(1e-7 + (one - torch.sigmoid(neg1))))
    return con_loss


class Model(nn.Module):
    def __init__(self, itemnum_target_domain, itemnum, ItemMeanFeatures, MetaFeatures, args):
        super().__init__()
        self.itemnum_target_domain = itemnum_target_domain
        self.itemnum = itemnum
        self.args = args
        self.dev = args.device
        self.hidden_units = args.hidden_units
        self.embedding_size = args.embedding_size
        self.review_emb = torch.tensor(ItemMeanFeatures, dtype=torch.float32, device=self.dev)
        self.meta_emb = torch.tensor(MetaFeatures, dtype=torch.float32, device=self.dev)
        self.item_emb = nn.Embedding(itemnum + 1, self.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen, self.hidden_units)
        self.layer_norm_eps = args.eps

        self.source_layer_norm = nn.LayerNorm(self.hidden_units, eps=self.layer_norm_eps)
        self.target_layer_norm = nn.LayerNorm(self.hidden_units, eps=self.layer_norm_eps)
        self.source_layer_dropout = nn.Dropout(args.dropout_rate)
        self.target_layer_dropout = nn.Dropout(args.dropout_rate)

        self.feat_target_Linear = nn.Linear(self.embedding_size, self.hidden_units)
        self.target_text_layer_norm = nn.LayerNorm(self.hidden_units, eps=self.layer_norm_eps)
        self.target_text_embed_dropout = nn.Dropout(args.dropout_rate)

        self.feat_source_Linear = nn.Linear(self.embedding_size, self.hidden_units)
        self.source_text_layer_norm = nn.LayerNorm(self.hidden_units, eps=self.layer_norm_eps)
        self.source_text_embed_dropout = nn.Dropout(args.dropout_rate)

        self.target_shared_transformer = Shared_Transformer(itemnum, args)
        self.source_text_aware_transformer = Text_aware_Transformer(itemnum, args)

        self.project_linear = nn.Linear(self.hidden_units, self.hidden_units)

        self.seqFeatLinear = nn.Linear(2 * self.hidden_units, self.hidden_units)
        self.seqFeat_layer_norm = nn.LayerNorm(self.hidden_units, eps=self.layer_norm_eps)

        self.last_linear = torch.nn.Linear(2 * self.hidden_units, self.hidden_units)
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=self.layer_norm_eps)

    def target_feat_emb(self, input_seq):
        seq_review = self.review_emb[input_seq]
        seq_meta = self.meta_emb[input_seq]
        seq_feat = seq_review + seq_meta
        seq_feat = self.feat_target_Linear(seq_feat)
        seq_feat = self.target_text_layer_norm(self.target_text_embed_dropout(seq_feat))
        return seq_feat

    def source_feat_emb(self, input_seq):
        seq_review = self.review_emb[input_seq]
        seq_meta = self.meta_emb[input_seq]
        seq_feat = seq_review + seq_meta
        seq_feat = self.feat_source_Linear(seq_feat)
        seq_feat = self.source_text_layer_norm(self.source_text_embed_dropout(seq_feat))
        return seq_feat

    def seq_self_attention_layer(self, targetSeq, sourceSeq):
        target_timeline_mask = torch.BoolTensor(targetSeq == 0).to(self.dev)
        source_timeline_mask = torch.BoolTensor(sourceSeq == 0).to(self.dev)

        source_text_feat = self.source_feat_emb(torch.LongTensor(sourceSeq).to(self.dev))
        target_text_feat = self.target_feat_emb(torch.LongTensor(targetSeq).to(self.dev))

        sourceEmb = self.source_layer_norm(self.source_layer_dropout(self.item_emb(torch.LongTensor(sourceSeq).to(self.dev))))  # 1 x item_num x h
        targetEmb = self.target_layer_norm(self.target_layer_dropout(self.item_emb(torch.LongTensor(targetSeq).to(self.dev)))) # 1 x item_num x h

        positions = np.tile(np.array(range(targetEmb.shape[1])), [targetEmb.shape[0], 1])
        targetSeqs = targetEmb + self.pos_emb(torch.LongTensor(positions).to(self.dev))
        target_text_feat = target_text_feat + self.pos_emb(torch.LongTensor(positions).to(self.dev))

        sourceSeqs = sourceEmb + self.pos_emb(torch.LongTensor(positions).to(self.dev))
        source_text_feat = source_text_feat + self.pos_emb(torch.LongTensor(positions).to(self.dev))

        # targetID -> SASRec
        targetSeqs = self.target_shared_transformer(targetSeqs, target_timeline_mask)

        # preference alignment mechanism
        cross_res, attention_map_list = self.source_text_aware_transformer(target_text_feat, target_timeline_mask,
                                                                           source_text_feat, sourceSeqs,
                                                                           source_timeline_mask)
        # target text Feat -> SASRec
        target_text_feat = self.target_shared_transformer(target_text_feat, target_timeline_mask, attention_map_list)

        # concat target domain text and id preference emb
        target_seq_feat = self.seq_feat_align(targetSeqs, target_text_feat)

        return target_seq_feat, cross_res

    def seq_feat_align(self, seq, feat):
        seq_feat = torch.cat((seq, feat), dim=-1)
        emb = self.seqFeatLinear(seq_feat)
        return emb

    def forward(self, targetSeq, sourceSeq, pos, neg):
        seq_sasrec, cross_res = self.seq_self_attention_layer(targetSeq, sourceSeq)
        pos_item_in = self.item_emb(torch.LongTensor(pos).to(self.dev))
        neg_item_in = self.item_emb(torch.LongTensor(neg).to(self.dev))

        pos_meta_feat = self.target_feat_emb(torch.LongTensor(pos).to(self.dev))
        pos_emb = self.seq_feat_align(pos_item_in, pos_meta_feat)

        neg_meta_feat = self.target_feat_emb(torch.LongTensor(neg).to(self.dev))
        neg_emb = self.seq_feat_align(neg_item_in,neg_meta_feat)

        # semantic contrastive learning
        cross_res = self.project_linear(cross_res)
        con_loss = SSL(seq_sasrec, cross_res)

        seqs = self.last_linear(torch.cat((seq_sasrec, cross_res), dim=-1))
        logits = self.last_layernorm(seqs)

        pos_logits = (logits * pos_emb).sum(dim=-1)
        neg_logits = (logits * neg_emb).sum(dim=-1)

        return pos_logits, neg_logits, con_loss/(seq_sasrec.shape[0]*seq_sasrec.shape[1])

    def full_predict(self, targetSeq, sourceSeq):
        seq_sasrec, cross_res = self.seq_self_attention_layer(targetSeq, sourceSeq)

        test_item = torch.arange(0, self.itemnum_target_domain).to(self.dev)

        test_item_in = self.item_emb(test_item)
        test_meta_feat = self.target_feat_emb(test_item)
        test_item_emb = self.seq_feat_align(test_item_in, test_meta_feat)

        final_feat = seq_sasrec[:, -1, :]  # 1 x h
        test_logits = test_item_emb.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return test_logits

    def predict(self, targetSeq, sourceSeq, test_item):
        seq_sasrec, cross_res = self.seq_self_attention_layer(targetSeq, sourceSeq)
        test_item_in = self.item_emb(torch.LongTensor(test_item).to(self.dev))
        test_meta_feat = self.target_feat_emb(torch.LongTensor(test_item).to(self.dev))
        test_item_emb = self.seq_feat_align(test_item_in, test_meta_feat)

        final_feat = seq_sasrec[:, -1, :]  # 1 x h
        test_logits = test_item_emb.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return test_logits
