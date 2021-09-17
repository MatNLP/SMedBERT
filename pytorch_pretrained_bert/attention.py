import enum
import math
import copy
import json
import torch
import torch.nn as nn
# from torch_geometric.nn import GraphConv,AGNNConv,FastRGCNConv,RGCNConv,DNAConv
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
# from apex.normalization import FusedLayerNorm

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        print(scores.size())
        print(mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)
    # In our case, query in shape (batch_size,max_len,e_dim)
    # Key in shape (batch_size,max_target, e_dim)
    # Use More aggresive stargy to caluate possible

    # p_attn_tmp = torch.exp(torch.softmax(scores, dim = -1))
    # p_attn = torch.softmax(p_attn_tmp*p_attn_tmp,dim = -1)

    p_attn = torch.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def attentionWithTypeScores(query, key, value, type_scores, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # Use More aggresive stargy to caluate possible

    # p_attn_tmp = torch.exp(torch.softmax(scores, dim = -1))
    # p_attn = torch.softmax(p_attn_tmp*p_attn_tmp,dim = -1)

    p_attn = torch.softmax(scores, dim=-1)
    p_attn *= type_scores
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.hidden_size = d_model
        self.d_k = self.hidden_size // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, self.hidden_size) for _ in range(3)])
        self.output = nn.Linear(self.hidden_size, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.num_attention_heads = h
        self.attention_head_size = self.d_k
        # for linears in self.linears:
        #     torch.nn.init.xavier_uniform(linears.weight)
        # torch.nn.init.xavier_uniform(self.output.weight)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = 1

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.output(x)


class MultiHeadedAttentionWithTypeScores(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionWithTypeScores, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.hidden_size = d_model
        self.d_k = self.hidden_size // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, self.hidden_size) for _ in range(3)])
        self.output = nn.Linear(self.hidden_size, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.num_attention_heads = h
        self.attention_head_size = self.d_k
        # for linears in self.linears:
        #     torch.nn.init.xavier_uniform(linears.weight)
        # torch.nn.init.xavier_uniform(self.output.weight)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, type_scores, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = 1

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attentionWithTypeScores(query, key, value, type_scores, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.output(x)


class SelfAttention(nn.Module):
    def __init__(self,head_num,hidden_size,dropout=.1):
        super(SelfAttention, self).__init__()
        if hidden_size % head_num != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, head_num))
        self.num_attention_heads = head_num
        self.attention_head_size = int(hidden_size / head_num)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q_in,k_in,v_in, attention_mask=None):
        mixed_query_layer = self.query(q_in)
        mixed_key_layer = self.key(k_in)
        mixed_value_layer = self.value(v_in)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if(attention_mask is not None):
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer



class TypeOneToManyAttention(nn.Module):
    def __init__(self,head_num,hidden_size,dropout=.1):
        super(TypeOneToManyAttention, self).__init__()
        if hidden_size % head_num != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, head_num))
        self.num_attention_heads = head_num
        self.attention_head_size = int(hidden_size // head_num)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    # def transpose_for_scores3(self, x):
    #     new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    #     x = x.view(*new_x_shape)
    #     return x.permute(0, 2, 1, 3)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, q_in,k_in,v_in,type_score,attention_mask=None):
        mixed_query_layer = self.query(q_in)
        mixed_key_layer = self.key(k_in)
        mixed_value_layer = self.value(v_in)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) # (batch_size,head_num,max_len,head_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if(attention_mask is not None):
            print(attention_scores.size())
            print(attention_mask.size())
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        type_attention_probs = attention_probs*type_score
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        type_attention_probs = self.dropout(type_attention_probs) # ((batch_size,head_num,max_len,max_len))

        context_layer = torch.matmul(type_attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape).squeeze(-2)
        return context_layer