import numpy as np
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from utils import get_slice
import scipy.sparse as sp


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


class SASRec(torch.nn.Module):
    def __init__(self, user_train1, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.train_data = user_train1

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.gnn = GNN(args.hidden_units, step=1)
        self.gnn2 = GNN(args.hidden_units, step=1)
        self.GCN = LightGCN(args, user_num, item_num, self.user_emb, self.item_emb, self.train_data)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()

        self.cross_attention_layernorms = torch.nn.ModuleList()
        self.cross_attention_layers = torch.nn.ModuleList()

        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.last_cross_layernorm = torch.nn.LayerNorm(2 * args.hidden_units, eps=1e-8)

        self.cross_forward_layernorms = torch.nn.ModuleList()
        self.cross_forward_layers = torch.nn.ModuleList()


        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_cross_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_attention_layernorms.append(new_cross_attn_layernorm)


            new_cross_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                               args.num_heads,
                                                               args.dropout_rate)
            self.cross_attention_layers.append(new_cross_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            new_cross_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_forward_layernorms.append(new_cross_fwd_layernorm)

            new_cross_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.cross_forward_layers.append(new_cross_fwd_layer)

        self.dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout2 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout3 = torch.nn.Dropout(p=args.dropout_rate)

        self.gating3 = torch.nn.Linear(2 * args.hidden_units, args.hidden_units)
        self.gating2 = torch.nn.Linear(1 * args.hidden_units, args.hidden_units)
        # self.gating = torch.nn.Linear(1 * args.hidden_units, args.hidden_units)

        self.w1gating = torch.nn.Linear(1 * args.hidden_units, args.hidden_units)
        self.w2gating = torch.nn.Linear(1 * args.hidden_units, args.hidden_units)
        # self.w3gating = torch.nn.Linear(1 * args.hidden_units, args.hidden_units)
        self.w4gating = torch.nn.Linear(1 * args.hidden_units, args.hidden_units)
        self.dropout4 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout5 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout6 = torch.nn.Dropout(p=args.dropout_rate)
        self.dropout7 = torch.nn.Dropout(p=args.dropout_rate)

        self.seq_layernorm1 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.seq_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.gate1 = torch.nn.Linear(2 * args.hidden_units, 2)
        self.gate2 = torch.nn.Linear(2 * args.hidden_units, 2)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Embedding):
            torch.nn.init.kaiming_normal_(module.weight.data)

    def log2feats(self, user_ids, log_seqs1, log_seqs2, mask, isTrain):

        u, gcn_hidden1 = self.GCN.get_embedding(torch.LongTensor(user_ids).to(self.dev),
                                                torch.LongTensor(log_seqs1).to(self.dev), isTrain)

        alias_inputs, A, items = get_slice(log_seqs1)

        A = torch.FloatTensor(A).to(self.dev)

        hidden = self.item_emb(torch.LongTensor(log_seqs1).to(self.dev))
        # timeline_mask1 = torch.BoolTensor(log_seqs1 == 0).to(self.dev)
        # hidden *= ~timeline_mask1.unsqueeze(-1)

        gnn_hidden = self.gnn(A, hidden)

        seqs1 = torch.cat((gcn_hidden1, gnn_hidden), dim=2)

        z = torch.nn.functional.softmax(self.gate1(seqs1), dim=2)

        z1 = torch.unsqueeze(z[:, :, 0], -1)
        z2 = torch.unsqueeze(z[:, :, 1], -1)

        seqs1 = z1 * gcn_hidden1 + z2 * gnn_hidden

        seqs1 = self.seq_layernorm1(seqs1)

        seqs1 *= self.item_emb.embedding_dim ** 0.5
        positions1 = np.tile(np.array(range(log_seqs1.shape[1])), [log_seqs1.shape[0], 1])
        seqs1 += self.pos_emb(torch.LongTensor(positions1).to(self.dev))
        seqs1 = self.emb_dropout(seqs1)

        timeline_mask1 = torch.BoolTensor(log_seqs1 == 0).to(self.dev)
        seqs1 *= ~timeline_mask1.unsqueeze(-1)

        tl1 = seqs1.shape[1]
        attention_mask1 = ~torch.tril(torch.ones((tl1, tl1), dtype=torch.bool, device=self.dev))

        u2, gcn_hidden2 = self.GCN.get_embedding(torch.LongTensor(user_ids).to(self.dev),
                                                torch.LongTensor(log_seqs2).to(self.dev), isTrain)

        alias_inputs2, A2, items2 = get_slice(log_seqs2)

        A2 = torch.FloatTensor(A2).to(self.dev)

        #
        hidden2 = self.item_emb(torch.LongTensor(log_seqs2).to(self.dev))
        # timeline_mask2 = torch.BoolTensor(log_seqs2 == 0).to(self.dev)
        # hidden2 *= ~timeline_mask2.unsqueeze(-1)

        gnn_hidden2 = self.gnn2(A2, hidden2)

        seqs2 = torch.cat((gcn_hidden2, gnn_hidden2), dim=2)

        zz = torch.nn.functional.softmax(self.gate2(seqs2), dim=2)

        zz1 = torch.unsqueeze(zz[:, :, 0], -1)
        zz2 = torch.unsqueeze(zz[:, :, 1], -1)

        seqs2 = zz1 * gcn_hidden2 + zz2 * gnn_hidden2

        seqs2 = self.seq_layernorm2(seqs2)

        seqs2 *= self.item_emb.embedding_dim ** 0.5
        positions2 = np.tile(np.array(range(log_seqs2.shape[1])), [log_seqs2.shape[0], 1])
        seqs2 += self.pos_emb(torch.LongTensor(positions2).to(self.dev))
        seqs2 = self.emb_dropout(seqs2)

        timeline_mask2 = torch.BoolTensor(log_seqs2 == 0).to(self.dev)
        seqs2 *= ~timeline_mask2.unsqueeze(-1)

        gcn_hidden1 = gcn_hidden1 * ~timeline_mask1.unsqueeze(-1)
        gnn_hidden = gnn_hidden * ~timeline_mask1.unsqueeze(-1)
        gcn_hidden2 = gcn_hidden2 * ~timeline_mask2.unsqueeze(-1)
        gnn_hidden2 = gnn_hidden2 * ~timeline_mask2.unsqueeze(-1)

        tl2 = seqs2.shape[1]
        batch_size = seqs2.shape[0]

        attention_mask2 = np.ones((batch_size, tl2, tl2), dtype=bool)

        for b in range(batch_size):
            for i in range(tl2):
                attention_mask2[b][i][0:mask[b][i]] = False

        attention_mask2[:, :, 0] = False

        seqs2_np = seqs2.cpu().detach().numpy()

        att_seq2_np = seqs2_np

        for b in range(batch_size):
            for i in range(tl2):
                att_seq2_np[b][i] = seqs2_np[b][mask[b][i] - 1]

        attention_mask2 = torch.from_numpy(attention_mask2).to(self.dev)
        # seqs2 = torch.from_numpy(seqs2).to(self.dev)
        att_seq2 = torch.from_numpy(att_seq2_np).to(self.dev)

        att_seq1 = seqs1
        for i in range(len(self.attention_layers)):
            att_seq1 = torch.transpose(att_seq1, 0, 1)

            Q = self.attention_layernorms[i](att_seq1)
            mha_outputs1, _ = self.attention_layers[i](Q, att_seq1, att_seq1,
                                                       attn_mask=attention_mask1)
            att_seq1 = Q + mha_outputs1

            att_seq1 = torch.transpose(att_seq1, 0, 1)

            att_seq1 = self.forward_layernorms[i](att_seq1)
            att_seq1 = self.forward_layers[i](att_seq1)
            att_seq1 *= ~timeline_mask1.unsqueeze(-1)

        for i in range(len(self.cross_attention_layers)):
            att_seq2 = torch.transpose(att_seq2, 0, 1)
            seqs2 = torch.transpose(seqs2, 0, 1)

            Q2 = self.cross_attention_layernorms[i](att_seq2)

            mha_outputs2, _ = self.cross_attention_layers[i](Q2, seqs2, seqs2,
                                                             attn_mask=attention_mask2)
            att_seq2 = Q2 + mha_outputs2

            seqs2 = torch.transpose(seqs2, 0, 1)
            att_seq2 = torch.transpose(att_seq2, 0, 1)

            att_seq2 = self.cross_forward_layernorms[i](att_seq2)
            att_seq2 = self.cross_forward_layers[i](att_seq2)
            att_seq2 *= ~timeline_mask2.unsqueeze(-1)


        att_seq2 = self.dropout2(self.gating2(att_seq2))
        con_loss = SSL(att_seq1, att_seq2)

        con_loss2 = SSL(self.dropout4(self.w1gating(gcn_hidden1)), self.dropout5(self.w2gating(gnn_hidden)))
        con_loss3 = SSL(self.dropout4(self.w1gating(gcn_hidden2)), self.dropout7(self.w4gating(gnn_hidden2)))

        seqs = torch.cat((att_seq1, att_seq2), dim=2)
        seqs = self.last_cross_layernorm(seqs)
        seqs = self.dropout3(self.gating3(seqs))
        log_feats = self.last_layernorm(seqs)

        return log_feats, con_loss, con_loss2, con_loss3

    def forward(self, user_ids, log_seqs, log_seqs2, pos_seqs, neg_seqs, mask):
        isTrain = True
        log_feats, con_loss, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, mask,
                                                                   isTrain)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, con_loss / (log_feats.shape[0] * log_feats.shape[1]), con_loss2 / (
                log_feats.shape[0] * log_feats.shape[1]), con_loss3 / (
                       log_feats.shape[0] * log_feats.shape[1])

    def predict(self, user_ids, log_seqs, log_seqs2, item_indices, mask):
        isTrain = False
        log_feats, con_loss, con_loss2, con_loss3 = self.log2feats(user_ids, log_seqs, log_seqs2, mask,
                                                                   isTrain)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.apply(self._init_weights)  # 初始化，效果略微提升

    # def _init_weights(self, module):
    #     if isinstance(module, torch.nn.Embedding):
    #         # torch.nn.init.xavier_normal_(module.weight.data)
    #         torch.nn.init.kaiming_normal_(module.weight.data)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)

        hy = hidden - inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


def SSL(sess_emb_hgnn, sess_emb_lgcn):
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

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


class LightGCN(torch.nn.Module):
    def __init__(self, args, user_num, item_num, user_emb, item_emb, data_list):
        super(LightGCN, self).__init__()
        self.device = args.device
        self.embedding_size = args.hidden_units
        self.user_count = user_num + 1
        self.item_count = item_num + 1
        self.n_layers = 3
        self.reg_weight = 1e-5

        self.user_embedding = user_emb
        self.item_embedding = item_emb
        self.data_list = data_list
        self.interaction_matrix = self.get_interaction_matrix()
        self.A_adj_matrix = self.get_a_adj_matrix()

        self.user_all_embedding, self.item_all_embedding = self.forward()

    def get_a_adj_matrix(self):
        A = sp.dok_matrix((self.user_count + self.item_count, self.user_count + self.item_count), dtype=np.float)
        inter_matrix = self.interaction_matrix
        inter_matrix_t = self.interaction_matrix.T

        data_dict = dict(zip(zip(inter_matrix.row, inter_matrix.col), [1] * inter_matrix.nnz))
        data_dict.update(dict(zip(zip(inter_matrix_t.row, inter_matrix_t.col), [1] * inter_matrix_t.nnz)))

        A._update(data_dict)
        sum_list = (A > 0).sum(axis=1)
        diag = np.array(sum_list.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)

        A_adj = D * A * D

        A_adj = sp.coo_matrix(A_adj)

        row = A_adj.row
        col = A_adj.col
        index = torch.LongTensor([row, col])
        data = torch.FloatTensor(A_adj.data)
        A_sparse = torch.sparse.FloatTensor(index, data, torch.Size(A_adj.shape))

        return A_sparse.to(self.device)

    def get_interaction_matrix(self):

        inter_list = []
        for key, value in self.data_list.items():
            user = key
            items = value
            for item in items:
                inter_list.append([int(user), int(item)])

        inter_list = np.array(inter_list)
        user_id = inter_list[:, 0]
        item_id = inter_list[:, 1]
        data = np.ones(len(inter_list))
        return sp.coo_matrix((data, (user_id, item_id)), shape=(self.user_count, self.item_count))

    def forward(self):
        user_embeddings = self.user_embedding.to(self.device).weight
        item_embeddings = self.item_embedding.to(self.device).weight
        all_embeddings = torch.cat([user_embeddings, item_embeddings]).to(self.device)

        embedding_list = [all_embeddings]
        for i in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.A_adj_matrix, all_embeddings)
            embedding_list.append(all_embeddings)

        total_E = torch.stack(embedding_list, dim=1)
        total_E = torch.mean(total_E, dim=1)
        user_all_embedding, item_all_embedding = torch.split(total_E, [self.user_count, self.item_count])

        return user_all_embedding, item_all_embedding

    def get_embedding(self, user_ids, log_seqs, isTrain):
        if isTrain:
            self.user_all_embedding, self.item_all_embedding = self.forward()
        return self.user_all_embedding[user_ids], self.item_all_embedding[log_seqs]
