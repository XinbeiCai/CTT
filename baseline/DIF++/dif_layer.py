
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as fn


class DIFMultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        maxlen=50
    ):
        super(DIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.query_f = nn.Linear(hidden_size, self.all_head_size)
        self.key_f = nn.Linear(hidden_size, self.all_head_size)



        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)



    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, query_states, key_states, value_states, attention_mask, position_embedding, feat_embedding):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        position_query_layer = self.query_p(position_embedding)
        position_key_layer = self.key_p(position_embedding)
        position_query_layer = self.transpose_for_scores(position_query_layer).permute(0, 2, 1, 3)
        position_key_layer = self.transpose_for_scores(position_key_layer).permute(0, 2, 3, 1)
        position_attention_map = torch.matmul(position_query_layer, position_key_layer)

        feat_query_layer = self.query_f(feat_embedding)
        feat_key_layer = self.key_f(feat_embedding)
        feat_query_layer = self.transpose_for_scores(feat_query_layer).permute(0, 2, 1, 3)
        feat_key_layer = self.transpose_for_scores(feat_key_layer).permute(0, 2, 3, 1)
        feat_attention_map = torch.matmul(feat_query_layer, feat_key_layer)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_map = torch.matmul(query_layer, key_layer)


        attention_scores = (attention_map + position_attention_map + feat_attention_map) / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        # hidden_states = self.out_dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + query_states)

        return hidden_states, attention_map


