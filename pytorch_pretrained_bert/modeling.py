# coding=utf-8
# dsafdsafdsa
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from allennlp.modules.span_extractors  import SelfAttentiveSpanExtractor

import copy
import json
import logging
import math
import os
import shutil
import sys
import tarfile
import tempfile
from io import open

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.attention import SelfAttention,TypeOneToManyAttention
from .file_utils import (CONFIG_NAME, MAX_NUM_PAIR, MAX_TARGET, WEIGHTS_NAME,
                         cached_path)

logger = logging.getLogger(__name__)

def check_has_nan(in_tensor):
    assert(torch.sum(torch.isnan(in_tensor))==0)


PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class ClsHead(nn.Module):
    def __init__(self, config, num_labels, dropout=0.3,need_rep=False):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.pre_dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.pre_dense_act = nn.Tanh()
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, num_labels, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_labels), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder.bias = self.bias
        self.need_rep = need_rep

    def forward(self, features, **kwargs):
        # features = self.pre_dense(features)
        # features = self.pre_dense_act(features)
        x = self.dense(features)
        x = gelu(x)
        x = self.dropout(x)
        v = self.layer_norm(x)
        x = self.decoder(v)
        return x,v if self.need_rep else x

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

try:
    from apex.normalization.fused_layer_norm import \
        FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
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


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)

        # We do not use the last two layer.
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        # self.mix_at = [10,11]
        # self.mix_one_hop_after = 8 
        self.mix_two_hop_after = 9
        # self.one_hop_mixer = ZeroCandicateKnowBERTLayer(config.hidden_size,100) # Careful! There is a hardcode num(e_dim asserted to be 100).
        self.two_hop_mixer = InfusionLimitKnowBERTLayer(config.hidden_size,100)

    def forward(self, hidden_states, attention_mask,kc_entity_one_hop_ids,kc_entity_one_hop_types,
                kc_entity_two_hop_labels,kc_entity_two_hop_types,kc_entity_infusion_pos,kc_entity_se_index,
                entity_embedding,entity_type_embedding, output_all_encoded_layers=True):
        
        all_encoder_layers = []

        kc_one_hop_entity_hiddens = entity_embedding(kc_entity_one_hop_ids)
        kc_one_hop_type_hiddens = entity_type_embedding(kc_entity_one_hop_types)
        
        kc_two_hop_entity_hiddens = entity_embedding(torch.where(kc_entity_two_hop_labels==-1,torch.zeros_like(kc_entity_two_hop_labels,device=kc_entity_two_hop_labels.device),kc_entity_two_hop_labels))
        kc_two_hop_type_hiddens = entity_type_embedding(kc_entity_two_hop_types)

        for index,layer_module in enumerate(self.layer): 
            hidden_states = layer_module(hidden_states, attention_mask)
            # if(index == self.mix_one_hop_after):
            #     hidden_states = self.one_hop_mixer(hidden_states, attention_mask, kc_entity_se_index,kc_entity_one_hop_ids,kc_one_hop_entity_hiddens,kc_one_hop_type_hiddens,kc_entity_infusion_pos)
            if(index == self.mix_two_hop_after):
                hidden_states = self.two_hop_mixer(hidden_states,kc_entity_se_index,torch.where(kc_entity_two_hop_labels==-1,torch.zeros_like(kc_entity_two_hop_labels,device=kc_entity_two_hop_labels.device),kc_entity_two_hop_labels),
                kc_entity_two_hop_types,
                kc_two_hop_entity_hiddens,kc_entity_infusion_pos,kc_one_hop_entity_hiddens,kc_one_hop_type_hiddens)
                
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        return (all_encoder_layers,hidden_states)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertSegPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertSegPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 2,
                                 bias=False)
        # self.decoder.weight = bert_model_embedding_weights
        # self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        # hidden_states = self.decoder(hidden_states) + self.bias
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class MLPWithLayerNorm(nn.Module):
    def __init__(self, config, input_size):
        super(MLPWithLayerNorm, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(input_size, config.hidden_size)
        self.non_lin1 = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.layer_norm1 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.non_lin2 = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        self.layer_norm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
    def forward(self, hidden):
        return self.layer_norm2(self.non_lin2(self.linear2(self.layer_norm1(self.non_lin1(self.linear1(hidden))))))

class BertPairDecodePredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights,max_targets=MAX_TARGET, position_embedding_size=200):
        super(BertPairDecodePredictionHead, self).__init__()
        self.position_embeddings = nn.Embedding(max_targets, position_embedding_size)
        self.mlp_layer_norm = MLPWithLayerNorm(config, config.hidden_size * 2 + position_embedding_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        self.max_targets = max_targets

    def forward(self, hidden_states, pairs):
        bs, num_pairs, _ = pairs.size()
        bs, seq_len, dim = hidden_states.size()
        # pair indices: (bs, num_pairs)
        left, right = pairs[:,:, 0], pairs[:, :, 1]
        # (bs, num_pairs, dim)
        left_hidden = torch.gather(hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim)) # This step get the target hidden for each pair postition.
        # pair states: bs * num_pairs, max_targets, dim
        left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1) # This step copy max_targets objs.
        right_hidden = torch.gather(hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim))
        # bs * num_pairs, max_targets, dim
        right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)
        # (max_targets, dim)
        position_embeddings = self.position_embeddings.weight
        hidden_states = self.mlp_layer_norm(torch.cat((left_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1)), -1))
        # target scores : bs * num_pairs, max_targets, vocab_size
        target_scores = self.decoder(hidden_states) + self.bias
        return target_scores

class BertPairClsPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights,max_targets=MAX_TARGET,cls_num=2, position_embedding_size=200):
        super(BertPairClsPredictionHead, self).__init__()
        if(max_targets==1): position_embedding_size=1 # Note that when max_target is 1, postition embedding is dumy.
        self.position_embeddings = nn.Embedding(max_targets, position_embedding_size)
        self.mlp_layer_norm = MLPWithLayerNorm(config, config.hidden_size * 2 + position_embedding_size)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 cls_num,
                                 bias=False)
        # self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(cls_num))
        self.max_targets = max_targets

    def forward(self, hidden_states, pairs):
        bs, num_pairs, _ = pairs.size()
        bs, seq_len, dim = hidden_states.size()
        # pair indices: (bs, num_pairs)
        left, right = pairs[:,:, 0], pairs[:, :, 1]
        # (bs, num_pairs, dim)
        left_hidden = torch.gather(hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim)) # This step get the target hidden for each pair postition.
        # pair states: bs * num_pairs, max_targets, dim
        left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1) # This step copy max_targets objs.
        right_hidden = torch.gather(hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim))
        # bs * num_pairs, max_targets, dim
        right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)
        # (max_targets, dim)
        position_embeddings = self.position_embeddings.weight
        hidden_states = self.mlp_layer_norm(torch.cat((left_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1)), -1))
        # target scores : bs * num_pairs, max_targets, vocab_size
        target_scores = self.decoder(hidden_states) + self.bias
        return target_scores


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                        from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            # print('Get Old:',key)
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it: getattr获取对象的属性值，可以直接进行调用
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model, missing_keys

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Add the new SegBERT pre-train model

class BertForSegPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
        `masked_seg_labels`: optional Segment masked modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, 1]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, 1], which represent the pos is a segment point(1) or not(0). 

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` and `next_sentence_label`are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss and the segment language modeling loss.
        # if `masked_lm_labels` or `next_sentence_label` is `None`:
        #     Outputs a tuple comprising
        #     - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
        #     - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForSegPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.seg_cls=BertSegPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,seg_lm_labels=None, next_sentence_label=None):
        assert(seg_lm_labels!=None) # We assume we will do the segment task.
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)

        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        seg_pre_scores= self.seg_cls(sequence_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            seg_lm_loss=loss_fct(seg_pre_scores.view(-1,2),seg_lm_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss + seg_lm_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score



class BertForHyperPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
        `masked_seg_labels`: optional Segment masked modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, 1]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, 1], which represent the pos is a segment point(1) or not(0). 

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` and `next_sentence_label`are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss and the segment language modeling loss.
        # if `masked_lm_labels` or `next_sentence_label` is `None`:
        #     Outputs a tuple comprising
        #     - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
        #     - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForHyperPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        # self.pos_neg_head=
        # self.seg_cls=BertSegPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
        self.pos_neg_cls=BertPairClsPredictionHead(config,self.bert.embeddings.word_embeddings.weight,max_targets= 1,cls_num= 2)
        self.seg_cls=BertPairClsPredictionHead(config,self.bert.embeddings.word_embeddings.weight,max_targets= MAX_TARGET,cls_num= 2)
        self.atom2modify_cls=BertPairDecodePredictionHead(config,self.bert.embeddings.word_embeddings.weight)
        self.modify2atom_cls=BertPairDecodePredictionHead(config,self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                    pairs_tuple=None,pairs_label_tuple=None ,next_sentence_label=None,reduce=True):
        # assert(seg_lm_labels!=None) # We assume we will do the segment task.
        # assert(pairs_tuple!=None)
        # assert(pairs_label_tuple!=None)
        pos_neg_pairs,seg_pairs,atom2modify_pairs,modify2atom_pairs=pairs_tuple
        pos_neg_labels,seg_labels,atom2modify_labels,modify2atom_labels=pairs_label_tuple

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)

        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        pos_neg_score = self.pos_neg_cls(sequence_output,pos_neg_pairs)
        seg_cls_score = self.seg_cls(sequence_output,seg_pairs)
        a2m_score = self.atom2modify_cls(sequence_output,atom2modify_pairs)
        m2a_score = self.modify2atom_cls(sequence_output,modify2atom_pairs)
        # seg_pre_scores= self.seg_cls(sequence_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            pos_neg_loss=F.cross_entropy(pos_neg_score.view(-1,pos_neg_score.size(-1)),pos_neg_labels.view(-1),size_average=False,reduce=reduce,ignore_index=-1)
            seg_cls_loss=F.cross_entropy(seg_cls_score.view(-1,seg_cls_score.size(-1)),seg_labels.view(-1),size_average=False,reduce=reduce,ignore_index=-1)
            a2m_loss=F.cross_entropy(a2m_score.view(-1,a2m_score.size(-1)),atom2modify_labels.view(-1),size_average=False,reduce=reduce,ignore_index=-1)
            m2a_loss=F.cross_entropy(m2a_score.view(-1,m2a_score.size(-1)),modify2atom_labels.view(-1),size_average=False,reduce=reduce,ignore_index=-1)
            
            # seg_lm_loss=loss_fct(seg_pre_scores.view(-1,2),seg_lm_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss + pos_neg_loss + seg_cls_loss + a2m_loss + m2a_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                kc_entity_one_hop_ids=None, kc_entity_one_hop_types=None,
                kc_entity_two_hop_labels=None,kc_entity_two_hop_types=None,kc_entity_infusion_pos=None,kc_entity_se_index=None,
                entity_embedding=None,entity_type_embedding=None, output_all_encoded_layers=True):

        # self.bert(input_ids, segment_ids, input_mask, one_hop_input_ent,
        #                                            one_hop_input_entity_type, # two_hop_entity_type_mask,two_hop_entity_type_index_select,
        #                                            one_hop_ent_mask, two_hop_input_ent, two_hop_ent_mask,
        #                                            two_hop_entity_types, output_all_encoded_layers=False)

        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(input_ids)

        
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_ent_mask = ent_mask.unsqueeze(1).unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # extended_ent_mask = extended_ent_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_ent_mask = (1.0 - extended_ent_mask) * -10000.0
        # extended_two_hop_ent_attention_mask = (1.0 - token_condidate_enttiy_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # assert(input_ent is not None)
        # assert(ent_mask is not None)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      kc_entity_one_hop_ids,
                                      kc_entity_one_hop_types,
                                      kc_entity_two_hop_labels,kc_entity_two_hop_types,kc_entity_infusion_pos,kc_entity_se_index,
                                      entity_embedding,entity_type_embedding,
                                      output_all_encoded_layers=output_all_encoded_layers)

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls( pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForMultipleChoice(BertPreTrainedModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_choices):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, entity_embedding, entity_type_embedding):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.entity_embedding = entity_embedding
        torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        self.entity_type_embedding = entity_type_embedding
        torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def parameter_complete(self, one_hop_entity_ids, one_hop_entity_types, two_hop_entity_ids):
        one_hop_input_ent = self.entity_embedding(one_hop_entity_ids+1)
        one_hop_input_entity_type = self.entity_type_embedding(one_hop_entity_types)
        one_hop_ent_mask = one_hop_entity_ids.ne_(-1)
        two_hop_input_ent = self.entity_embedding(two_hop_entity_ids+1)
        two_hop_ent_mask = two_hop_entity_ids.ne_(-1)
        return one_hop_input_ent, one_hop_input_entity_type, one_hop_ent_mask, two_hop_input_ent, two_hop_ent_mask

    def forward(self, input_ids, segment_ids=None, input_mask=None,
                one_hop_entity_ids=None, one_hop_entity_types=None, two_hop_entity_ids=None,
                two_hop_entity_type_mask=None, two_hop_entity_type_index_select=None, two_hop_entity_type=None,
                start_positions=None, end_positions=None, output_all_encoded_layers=True):

        one_hop_input_ent, one_hop_input_entity_type, one_hop_ent_mask, two_hop_input_ent, two_hop_ent_mask = \
            self.parameter_complete(one_hop_entity_ids, one_hop_entity_types, two_hop_entity_ids)
        sequence_output, _ = self.bert(input_ids, segment_ids, input_mask, one_hop_input_ent,
                                                   one_hop_input_entity_type, two_hop_entity_type_mask,
                                                   two_hop_entity_type_index_select,
                                                   one_hop_ent_mask, two_hop_input_ent, two_hop_ent_mask,
                                                   two_hop_entity_type, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

### /* Start Insert Code For ERNIE */

class BertEntitySelfAttention(nn.Module):
    def __init__(self, config):
        super(BertEntitySelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
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

class BertAttention_simple(nn.Module):
    def __init__(self, config):
        super(BertAttention_simple, self).__init__()
        self.self = BertEntitySelfAttention(config)
        self.output = BertEntitySelfOutput(config)
    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertEntitySelfOutput(nn.Module):
    def __init__(self, config):
        super(BertEntitySelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertEntityIntermediate(nn.Module):
    def __init__(self, config):
        super(BertEntityIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_ent = nn.Linear(100, config.intermediate_size, bias=True)
        torch.nn.init.xavier_uniform_(self.dense_ent.weight)
        torch.nn.init.constant_(self.dense_ent.bias, 0)

        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
    def forward(self, hidden_states, hidden_states_ent):
        hidden_states_ = self.dense(hidden_states)
        hidden_states_ent_ = self.dense_ent(hidden_states_ent)
        hidden_states = self.intermediate_act_fn(hidden_states_+hidden_states_ent_)
        return hidden_states#, hidden_states_ent
    
class BertEntityOutput(nn.Module):
    def __init__(self, config):
        super(BertEntityOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense_ent = nn.Linear(config.intermediate_size, 100)
        torch.nn.init.xavier_uniform_(self.dense_ent.weight)
        torch.nn.init.constant_(self.dense_ent.bias, 0)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm_ent = BertLayerNorm(100, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states_, input_tensor, input_tensor_ent):
        hidden_states = self.dense(hidden_states_)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        hidden_states_ent = self.dense_ent(hidden_states_)
        hidden_states_ent = self.dropout(hidden_states_ent)
        hidden_states_ent = self.LayerNorm_ent(hidden_states_ent + input_tensor_ent)

        return hidden_states, hidden_states_ent

class BertEntityAttention(nn.Module):
    def __init__(self, config):
        super(BertEntityAttention, self).__init__()
        self.self = BertEntitySelfAttention(config)
        self.output = BertEntitySelfOutput(config)

        config_ent = copy.deepcopy(config)
        config_ent.hidden_size = 100
        config_ent.num_attention_heads = 4

        self.self_ent = BertEntitySelfAttention(config_ent)
        self.output_ent = BertEntitySelfOutput(config_ent)

        self.self_two_hop_ent = BertEntitySelfAttention(config_ent)
        self.output_two_hop_ent = BertEntitySelfOutput(config_ent)


    def forward(self, input_tensor, attention_mask, input_tensor_ent, attention_mask_ent):

        self_output = self.self(input_tensor, attention_mask)
        self_output_ent = self.self_ent(input_tensor_ent, attention_mask_ent)
        # self_output_two_hop_ent = self.self_two_hop_ent(input_tensor_two_hop_ent, attention_mask_two_hop_ent)

        attention_output = self.output(self_output, input_tensor)
        attention_output_ent = self.output_ent(self_output_ent, input_tensor_ent)
        # attention_output_two_hop_ent = self.output_two_hop_ent(self_output_two_hop_ent, input_tensor_two_hop_ent)

        return attention_output, attention_output_ent

class AdditiveAttention(nn.Module):
    def __init__(self,q_dim,kv_dim):
        super().__init__()
        self.q_proj = nn.Linear(q_dim,q_dim,False)
        self.k_proj = nn.Linear(kv_dim,q_dim,False)
        self.v_proj = nn.Linear(q_dim,1,bias=False)
        self.act = nn.Tanh()
    def forward(self,query,kv):
        q = self.q_proj(query).unsqueeze(-2)
        k = self.k_proj(kv)
        intermediate = self.act(k+q)
        return self.v_proj(intermediate).squeeze(-1)

class ZeroCandicateKnowBERTLayer(nn.Module):
    def __init__(self,h_dim,e_dim): # h_dim means BERT hidden status dim, which is 768 in BERT-base.
        head_num = 4
        super().__init__()
        self.span_extractor = SelfAttentiveSpanExtractor(h_dim)
        self.bert2entity = nn.Linear(h_dim,e_dim)
        self.back_projecter = nn.Linear(e_dim,h_dim)
        self.MHA = SelfAttention(head_num,e_dim)
        self.info_mix_dense = nn.Linear(e_dim,e_dim)
        self.act = nn.GELU()
        self.layer_norm = BertLayerNorm(h_dim)
        self.eh_mixer = nn.Linear(2*h_dim,h_dim)
        self.update_gate = nn.Linear(2*h_dim,h_dim)
        # self.out_layer_norm = BertLayerNorm(h_dim)
        assert(e_dim % head_num == 0)
    
    def forward(self, hidden_states, attention_mask, kc_entity_se_index,one_hop_entity_ids,
                one_hop_entity_hiddens,one_hop_type_hiddens,kc_entity_infusion_pos):
        batch_size, max_target ,_ = kc_entity_se_index.size()
        h_span_rep = self.span_extractor(hidden_states,kc_entity_se_index)
        # print(h_span_rep.size())
        # print(hidden_states.size())
        # print(kc_entity_se_index.size())
        h_span_rep = self.bert2entity(h_span_rep)
        e_hidden_states = self.bert2entity(hidden_states)
        batch_size,max_len,e_dim = hidden_states.size()
        h_dim = hidden_states.size(-1)
        # print(one_hop_entity_hiddens.size())
        # print(one_hop_type_hiddens.size())
        # print(h_span_rep.size())
        span_rep = h_span_rep + one_hop_entity_hiddens + one_hop_type_hiddens
        span_rep = self.act(self.info_mix_dense(span_rep))
        # print(span_rep.size())
        batch_size,max_target,e_dim = span_rep.size()
        # print(one_hop_entity_ids.size())
        # print(span_rep.size())
        assert(one_hop_entity_ids.size() == (batch_size,max_target))
        # print(e_hidden_states.size())
        # print(span_rep.size())
        # print(one_hop_entity_ids.unsqueeze(-2).expand(-1,max_len,-1).size())
        attention_mask = (one_hop_entity_ids==0).unsqueeze(1).unsqueeze(2)*(-1e5)
        e_hidden_states = self.MHA(e_hidden_states,span_rep,span_rep,attention_mask=attention_mask) # Test Dropout Later!
        e_hidden_states = self.layer_norm(self.back_projecter(e_hidden_states))
        
        gate_score = self.update_gate(torch.cat((hidden_states,e_hidden_states),dim=-1))
        gate_score = F.tanh(gate_score)
        kc_entity_infusion_pos  = kc_entity_infusion_pos.detach().unsqueeze(-1).expand(-1,-1,h_dim)
        gate_mask = torch.where(kc_entity_infusion_pos!=0,torch.ones_like(kc_entity_infusion_pos),torch.zeros_like(kc_entity_infusion_pos))
        e_hidden_states = gate_score * e_hidden_states * gate_mask
        final_hidden = self.act(self.eh_mixer(torch.cat((hidden_states,e_hidden_states),dim=-1)))
        return  hidden_states + final_hidden



class InfusionLimitKnowBERTLayer(nn.Module):
    def __init__(self,h_dim,e_dim):
        head_num = 4
        super().__init__()
        self.type_attention = TypeOneToManyAttention(head_num,e_dim) # Used For P2P attention
        self.span_extractor = SelfAttentiveSpanExtractor(h_dim)
        
        self.bert2entity = nn.Linear(h_dim,e_dim)
        self.entity22entity4 = nn.Linear(2*e_dim,4*e_dim)
        self.entity42entity = nn.Linear(4*e_dim,e_dim)
        self.entity32entity = nn.Linear(3*e_dim,e_dim)
        # self.entity_layer_norm = BertLayerNorm(e_dim)
        # self.ffn_dropout = nn.Dropout(.1)

        self.ta_amp = nn.Linear(e_dim,e_dim*4) # 'ta' means type attention.
        self.ta_nar = nn.Linear(e_dim*4,e_dim)
        self.ta_layer_norm = BertLayerNorm(e_dim)
        
        
        self.add_attention = AdditiveAttention(e_dim,e_dim)
        self.infusion_back_project = nn.Linear(2*e_dim,h_dim)
        # self.entity_type_linear = nn.Linear(2*e_dim,1)
        self.one_two_hop_mix = nn.Linear(2*e_dim,2*e_dim)
        self.layer_norm = BertLayerNorm(h_dim)
        self.eh_mixer = nn.Linear(h_dim*2,h_dim)
        self.update_gate = nn.Linear(2*h_dim,h_dim)
        assert(e_dim % head_num == 0)

    def forward(self,hidden_states,kc_entity_se_index,
                two_hop_entity_ids,
                two_hop_entity_types,
                two_hop_entity_hiddens,kc_entity_infusion_pos,kc_one_hop_entity_hiddens,kc_one_hop_type_hiddens):
        # 方案三：将entity type信息加入到原始的attention中进行计算
        two_hop_one_hot = F.one_hot(two_hop_entity_types)
        
        dim = two_hop_entity_hiddens.size(-1)
        batch_size, max_target, two_hop_num, class_num = two_hop_one_hot.size()
        assert(two_hop_entity_hiddens.size() == (batch_size, max_target, two_hop_num,dim))
        class_num = class_num - 1 # exclude the padding type, which is 0.

        two_hop_one_hot = two_hop_one_hot[:,:,:,1:].unsqueeze(-1).expand(-1,-1,-1,-1,dim)
        expand_two_hop_entity_hiddens = two_hop_entity_hiddens.unsqueeze(-2).expand(-1,-1,-1,class_num,-1)
        # print(two_hop_one_hot.size())
        # print(expand_two_hop_entity_hiddens.size())
        sum_two_hop_entity_hiddens = torch.sum(two_hop_one_hot*expand_two_hop_entity_hiddens,dim=2)
        two_hop_num_factor = torch.sum(two_hop_one_hot,dim=2)
        pad_two_hop_num_factor = torch.where( two_hop_num_factor==0 ,torch.ones_like(two_hop_num_factor,device=two_hop_num_factor.device),two_hop_num_factor)
        assert(pad_two_hop_num_factor.size() == (batch_size,max_target,class_num,dim))
        sum_two_hop_entity_hiddens /= pad_two_hop_num_factor
        # assert(sum_two_hop_entity_hiddens.requires_grad)
        assert(sum_two_hop_entity_hiddens.size() == (batch_size,max_target,class_num,dim))

        h_span_rep = self.span_extractor(hidden_states,kc_entity_se_index)
        h_span_rep = self.bert2entity(h_span_rep)
        ke_h_span_rep = F.gelu(self.entity32entity(torch.cat((h_span_rep,kc_one_hop_entity_hiddens,kc_one_hop_type_hiddens),dim=-1)))
        
        h_span_rep = self.entity42entity(F.gelu(self.entity22entity4(torch.cat((ke_h_span_rep,h_span_rep),dim=-1)))) + h_span_rep

        # assert(one_hop_entity_hiddens.size() == (bathc_size,max_len,dim))
        assert(h_span_rep.size() == (batch_size,max_target,dim))
        two_hop_type_logits = self.add_attention(h_span_rep,sum_two_hop_entity_hiddens)
        # print(two_hop_type_logits.size())
        # two_hop_one_hop_contcat_rep = torch.cat((h_span_rep.unsqueeze(-2).expand(-1,-1,class_num,-1),sum_two_hop_entity_hiddens),dim=-1)
        # two_hop_type_logits = F.gelu(self.entity_type_linear(two_hop_one_hop_contcat_rep)).squeeze(-1) # squeeze the last single dim.
        assert(two_hop_type_logits.size() == (batch_size,max_target,class_num))
        two_hop_type_logits  -= (two_hop_num_factor[:,:,:,0].squeeze(-1)==0)*10000
        two_hop_type_scores = F.softmax(two_hop_type_logits,dim=-1)
        assert(two_hop_type_scores.size() == (batch_size,max_target,class_num))
        dumy_two_hop_type_score = torch.zeros((batch_size,max_target,1),device=two_hop_type_scores.device)
        two_hop_type_scores = torch.cat((dumy_two_hop_type_score,two_hop_type_scores),dim=-1) # We recover the padding type score(0).
        two_hop_type_scores_for_each_entity = torch.gather(two_hop_type_scores,dim=-1,index=two_hop_entity_types)
        assert(two_hop_type_scores_for_each_entity.size() == (batch_size,max_target,two_hop_num))
        p2p_attention_rep = self.type_attention(h_span_rep.unsqueeze(-2),two_hop_entity_hiddens,two_hop_entity_hiddens,two_hop_type_scores_for_each_entity.unsqueeze(-2).unsqueeze(-2))
        
        p2p_attention_rep = self.ta_layer_norm(p2p_attention_rep+(self.ta_nar(F.gelu(self.ta_amp(p2p_attention_rep)))))

        assert(p2p_attention_rep.size() == (batch_size,max_target,dim))
        p2p_attention_rep = torch.cat((p2p_attention_rep,h_span_rep),dim=-1)
        pad_p2p_attention_rep = torch.cat( (torch.zeros((batch_size,1,2*dim),device=p2p_attention_rep.device),p2p_attention_rep),dim=1)
        assert(kc_entity_infusion_pos.size() == (batch_size,512)) # Care This!
        pad_p2p_attention_rep = F.gelu(self.one_two_hop_mix(pad_p2p_attention_rep))
        kc_entity_infusion_pos = kc_entity_infusion_pos.unsqueeze(-1).expand(-1,-1,2*dim)
        sequence_p2p_attention_rep = torch.gather(pad_p2p_attention_rep,1,kc_entity_infusion_pos)
        sequence_p2p_attention_rep = self.infusion_back_project(sequence_p2p_attention_rep)
        sequence_p2p_attention_rep = self.layer_norm(sequence_p2p_attention_rep)
        
        gate_score = self.update_gate(torch.cat((hidden_states,sequence_p2p_attention_rep),dim=-1))
        gate_score = F.tanh(gate_score)
        sequence_p2p_attention_rep = gate_score * sequence_p2p_attention_rep
        
        sequence_p2p_attention_rep = (sequence_p2p_attention_rep)
        assert(hidden_states.size() == sequence_p2p_attention_rep.size())
        final_hidden = F.gelu(self.eh_mixer(torch.cat((hidden_states,sequence_p2p_attention_rep),dim=-1)))
        # sequence_p2p_attention_fusion_hidden_rep = torch.cat((sequence_p2p_attention_rep,hidden_states),dim=-1)
        # print(sequence_p2p_attention_fusion_hidden_rep.size())
        return hidden_states + final_hidden


class BertLayerMix(nn.Module):
    def __init__(self, config):
        super(BertLayerMix, self).__init__()
        self.attention = BertAttention_simple(config)
        self.intermediate = BertEntityIntermediate(config)
        self.output = BertEntityOutput(config)

        self.entity_linear = nn.Linear(100*2, 100, bias=True)
        # torch.nn.init.xavier_uniform_(self.entity_linear.weight)
        # torch.nn.init.constant_(self.entity_linear.bias, 0)

        self.entity_type_linear = nn.Linear(100, 100, bias=True)
        # torch.nn.init.xavier_uniform_(self.entity_type_linear.weight)
        # torch.nn.init.constant_(self.entity_type_linear.bias, 0)

        self.attention_linear = nn.Linear(100*2, 1, bias=True)
        # torch.nn.init.xavier_uniform_(self.attention_linear.weight)
        # torch.nn.init.constant_(self.attention_linear.bias, 0)

    def forward(self, hidden_states, attention_mask, one_hop_entity_ids,two_hop_entity_ids,
                one_hop_entity_types,two_hop_entity_type,
                one_hop_entity_hiddens,one_hop_type_hiddens,
                two_hop_entity_hiddens,two_hop_type_hiddens
                ): 
                # Warning Some Parameter is unsed and is not consisented with actually input!!!.
        attention_output = self.attention(hidden_states, attention_mask)
        # 在ERNIE这个地方进行融合是错误的，这个地方只是进行了融合前的维度操作
        # hidden_states_ent = self.two_hops_entity_enhanced(hidden_states_ent, token_candidate_entity,
        #                                   extended_token_candidate_entity_mask, two_hop_entity_type)
        one_hop_entity_hiddens = F.gelu(self.entity_type_linear(one_hop_entity_hiddens + one_hop_type_hiddens))
        
        attention_output_ent = one_hop_entity_hiddens
        # attention_output_ent = hidden_states_ent * ent_mask 
        # In this way, it keep the padding entity hiddens to be all zero, while it is not nessary to do so? 
        
        intermediate_output = self.intermediate(attention_output, attention_output_ent) # Get H_i
        layer_output, layer_output_ent = self.output(intermediate_output, attention_output, attention_output_ent) # Get W and E from H
        return layer_output, layer_output_ent
        
    def two_hops_entity_enhanced(self, hidden_state_ent, token_candidate_entity,
                                 extended_token_candidate_entity_mask, two_hop_entity_type):
        # 方案三：将entity type信息加入到原始的attention中进行计算
        hidden_state_ent_copy = hidden_state_ent.clone()
        node_attention_value = hidden_state_ent.view(-1, 1, hidden_state_ent.size(2)).bmm(token_candidate_entity.view(-1,
                                                                    token_candidate_entity.size(3),
                                                                    token_candidate_entity.size(2)))
        type_attention_value = hidden_state_ent.view(-1, 1, hidden_state_ent.size(2)).bmm(two_hop_entity_type.view(-1,
                                                                                 two_hop_entity_type.size(3),
                                                                                 two_hop_entity_type.size(2)))
        attention_value_softmax = F.softmax(node_attention_value.mul(type_attention_value), dim=2)

        # 方案二
        # hidden_state_ent_copy = hidden_state_ent.clone()
        # hidden_state_ent = hidden_state_ent.unsqueeze(2)
        # hidden_state_ent = hidden_state_ent.expand(hidden_state_ent.size(0), hidden_state_ent.size(1),
        #                                            token_candidate_entity.size(2), hidden_state_ent.size(3))
        # # batch * max_seq * top_K * entity_embedding
        # entity_hop_1_2_cat = torch.cat((hidden_state_ent, token_candidate_entity), dim=3)
        # attention_value_softmax = F.softmax(F.sigmoid(self.attention_linear(entity_hop_1_2_cat).squeeze(3)), dim=2)
        # attention_value_softmax = attention_value_softmax.view(-1, 1, attention_value_softmax.size(2))

        # 方案一
        # attention_value = hidden_state_ent.view(-1, 1, hidden_state_ent.size(2)).bmm(token_candidate_entity.view(-1,
        #                                                             token_candidate_entity.size(3),
        #                                                             token_candidate_entity.size(2)))
        # attention_value_softmax = F.softmax(attention_value, dim=2)

        attention_mask_value = attention_value_softmax.mul(extended_token_candidate_entity_mask.view(-1, 1,
                                                                    extended_token_candidate_entity_mask.size(2)))
        two_hops_enhanced_entity_repre = attention_mask_value.bmm(
                            token_candidate_entity.view(-1, token_candidate_entity.size(2),
                            token_candidate_entity.size(3)))
        batch_size = two_hops_enhanced_entity_repre.size(0)
        two_hops_enhanced_entity_repre = two_hops_enhanced_entity_repre.squeeze().view(
                                                    int(batch_size/512), 512, 100)
        return self.entity_linear(torch.cat((hidden_state_ent_copy, two_hops_enhanced_entity_repre), dim=2))

class BertEntLayer(nn.Module):
    def __init__(self, config):
        super(BertEntLayer, self).__init__()
        self.attention = BertEntityAttention(config)
        self.intermediate = BertEntityIntermediate(config)
        self.output = BertEntityOutput(config)

        self.two_hop_linear = nn.Linear(100, 100, bias=True)
        # self.two_hop_linear.weight.data.uniform_(-1, 1)

        self.entity_linear = nn.Linear(100*2, 100, bias=True)
        # self.entity_linear.weight.data.uniform_(-1, 1)

        self.entity_type_linear = nn.Linear(100*2, 1, bias=True)
        # self.entity_type_linear.weight.data.uniform_(-1, 1)

        self.entity_node_linear = nn.Linear(100*2, 1, bias=True)
        # self.entity_node_linear.weight.data.uniform_(-1, 1)
        self.two_hop_q_linear = nn.Linear(100,100)
        
    def forward(self, hidden_states, attention_mask, one_hop_entity_ids,kc_entity_two_hop_labels,
                one_hop_entity_types,kc_entity_two_hop_types,
                one_hop_entity_hiddens,one_hop_type_hiddens,
                kc_two_hop_entity_hiddens,kc_two_hop_type_hiddens):
        one_hop_entity_mask = (-10000)*(one_hop_entity_ids==0).unsqueeze(1).unsqueeze(2)
        attention_output, attention_output_ent = self.attention(hidden_states,
                                                                attention_mask,
                                                                one_hop_entity_hiddens,one_hop_entity_mask)
        # 注入二阶HOP知识 增强attention_output_ent
        # 为什么不和上面一样先过一遍self-attention？  因为自注意力是通过上下文进行的计算，two-hop已经没有上下文的概念，最多只有5个相邻的实体，
        # 所以应该只需要进行简单的MLP映射即可注入到相关one-hop实体中去
        # two_hop_entity_hiddens = F.gelu(self.two_hop_linear(two_hop_entity_hiddens))

        # 二阶实体对一阶实体的知识增强
        attention_output_ent = self.two_hops_entity_enhanced(one_hop_entity_ids,two_hop_entity_ids,
                                                                one_hop_entity_types,two_hop_entity_types,
                                                                one_hop_entity_hiddens,one_hop_type_hiddens,
                                                                two_hop_entity_hiddens,two_hop_type_hiddens
                                                             )
        # ent_mask = one_hop_entity_ids != 0
        # attention_output_ent = attention_output_ent * ent_mask
        intermediate_output = self.intermediate(attention_output, attention_output_ent)
        layer_output, layer_output_ent = self.output(intermediate_output, attention_output, attention_output_ent)
        # layer_output_ent = layer_output_ent * ent_mask
        return layer_output, layer_output_ent

    def two_hops_entity_enhanced(self,
                                one_hop_entity_ids,two_hop_entity_ids,
                                one_hop_entity_types,two_hop_entity_types,
                                one_hop_entity_hiddens,one_hop_type_hiddens,
                                two_hop_entity_hiddens,two_hop_type_hiddens
                                #  hidden_state_ent, token_candidate_entity,
                                #  extended_token_candidate_entity_mask, two_hop_entity_type,
                                #  two_hop_entity_type_mask, two_hop_entity_type_index_select
                                ):
        
        # 方案三：将entity type信息加入到原始的attention中进行计算
        two_hop_one_hot = F.one_hot(two_hop_entity_types)
        
        dim = two_hop_entity_hiddens.size(-1)
        bathc_size, max_len, two_hop_num, class_num = two_hop_one_hot.size()
        class_num = class_num - 1 # exclude the padding type, which is 0.

        two_hop_one_hot = two_hop_one_hot[:,:,:,1:].unsqueeze(-1).expand(-1,-1,-1,-1,dim).detach()
        expand_two_hop_entity_hiddens = two_hop_entity_hiddens.unsqueeze(-2).expand(-1,-1,-1,class_num,-1)
        # print(two_hop_one_hot.size())
        # print(expand_two_hop_entity_hiddens.size())
        sum_two_hop_entity_hiddens = torch.sum(two_hop_one_hot*expand_two_hop_entity_hiddens,dim=2)
        two_hop_num_factor = torch.sum(two_hop_one_hot,dim=2).detach()
        two_hop_num_factor = torch.where( two_hop_num_factor==0 ,torch.ones_like(two_hop_num_factor,device=two_hop_num_factor.device),two_hop_num_factor)

        # print(two_hop_num_factor.size())
        # print(sum_two_hop_entity_hiddens.size())
        # print(non_zero_pos.size())
        sum_two_hop_entity_hiddens /= two_hop_num_factor
        assert(sum_two_hop_entity_hiddens.requires_grad)
        assert(sum_two_hop_entity_hiddens.size() == (bathc_size,max_len,class_num,dim))
        assert(one_hop_entity_hiddens.size() == (bathc_size,max_len,dim))
        two_hop_one_hop_contcat_rep = torch.cat((one_hop_entity_hiddens.unsqueeze(-2).expand(-1,-1,class_num,-1),sum_two_hop_entity_hiddens),dim=-1)
        two_hop_type_logits = F.gelu(self.entity_type_linear(two_hop_one_hop_contcat_rep)).squeeze(-1) # squeeze the last single dim.
        assert(two_hop_type_logits.size() == (bathc_size,max_len,class_num))
        two_hop_type_logits  -= (two_hop_num_factor[:,:,:,0].squeeze(-1)==0)*10000
        two_hop_type_scores = F.softmax(two_hop_type_logits,dim=-1)
        assert(two_hop_type_scores.size() == (bathc_size,max_len,class_num))

        # There would like to have mulit-head attention, while we implement the single head version for now.
        two_hop_entity_hiddens_q = self.two_hop_q_linear(two_hop_entity_hiddens) # Proj to Query vector.
        one_two_hop_concate = torch.cat((one_hop_entity_hiddens.unsqueeze(-2).expand(-1,-1,two_hop_num,-1),two_hop_entity_hiddens_q),dim=-1)
        
        p2p_attention_logist = F.gelu(self.entity_node_linear(one_two_hop_concate)).squeeze(-1) # p2p means one-hop entity attention with its neghibors
        two_hop_attention_mask = ((two_hop_entity_ids == 0 )*10000).detach()

        p2p_attention_logist -= two_hop_attention_mask
        p2p_attention_score = F.softmax(p2p_attention_logist,dim=-1)
        assert(p2p_attention_score.size() == (bathc_size,max_len,two_hop_num))
        dumy_two_hop_type_score = torch.zeros((bathc_size,max_len,1),device=two_hop_type_scores.device).detach()
        two_hop_type_scores = torch.cat((dumy_two_hop_type_score,two_hop_type_scores),dim=-1) # We recover the padding type score(0).
        two_hop_type_scores_for_each_entity = torch.gather(two_hop_type_scores,dim=-1,index=two_hop_entity_types)
        p2p_attention_score_norm = p2p_attention_score*two_hop_type_scores_for_each_entity
        p2p_attention_score_norm = p2p_attention_score_norm.view(-1,two_hop_num).unsqueeze(1)
        assert(p2p_attention_score_norm.size() == (bathc_size*max_len,1,two_hop_num))
        two_hop_fused_rep = torch.bmm(p2p_attention_score_norm,two_hop_entity_hiddens.view(bathc_size*max_len,two_hop_num,dim)).squeeze(1)
        one_two_hop_fused_rep = torch.cat((one_hop_entity_hiddens,two_hop_fused_rep.view(bathc_size,max_len,dim)),dim=-1)
        return F.gelu(self.entity_linear(one_two_hop_fused_rep))


        # hidden_state_ent_copy = hidden_state_ent.clone()
        
        # # two_hop_entity_type 只是用来聚合entity type信息
        # two_hop_entity_type = two_hop_entity_type.permute(0, 1, 3, 2)
        # # (batch * max_len) * entity_type_num * dim
        # two_hop_type_repre = torch.bmm(two_hop_entity_type.reshape(-1, two_hop_entity_type.size(2),
        #                                                            two_hop_entity_type.size(3)).float(),
        #                                token_candidate_entity.reshape(-1, token_candidate_entity.size(2),
        #                                                               token_candidate_entity.size(3)))
        # # 和中心实体concate计算type attention
        # # (batch * max_len) * entity_type_num
        # alpha_type_attention = F.gelu(self.entity_type_linear(torch.cat((two_hop_type_repre,
        #                             hidden_state_ent.view(-1, 1, hidden_state_ent.size(2)).expand(-1,
        #                             two_hop_type_repre.size(1), hidden_state_ent.size(2))), dim=2))).squeeze(dim=2)
        # # batch * max_len * entity_type_num
        # alpha_type_attention_softmax = F.softmax(alpha_type_attention.reshape(hidden_state_ent_copy.size(0),
        #                                                                       hidden_state_ent_copy.size(1),
        #                                                                       alpha_type_attention.size(1)), dim=2)

        # # 将type attention 分配到对应的每个二阶实体中（二阶实体个数与entity type个数不一样，需要重构index）
        # # alpha_type_attention_softmax = batch * max_len * two_hop_num
        # alpha_type_attention_softmax = alpha_type_attention_softmax.mul(two_hop_entity_type_mask)
        # alpha_type_attention_softmax = alpha_type_attention_softmax.view(-1, alpha_type_attention_softmax.size(2))
        # alpha_type_attention_softmax = torch.gather(alpha_type_attention_softmax.reshape(two_hop_entity_type_mask.size()),
        #                                             2, two_hop_entity_type_index_select)

        # # alpha_type_attention_softmax = torch.stack([torch.index_select(a, 0, i) for a, i in zip(
        # #     alpha_type_attention_softmax, two_hop_entity_type_index_select)], dim=0).reshape(
        # #     two_hop_entity_type_index_select_size)

        # # Node-level attention
        # one_two_hop_concate = torch.cat((hidden_state_ent_copy.reshape(hidden_state_ent_copy.size(0),
        #                                                                hidden_state_ent_copy.size(1), 1,
        #                                          hidden_state_ent_copy.size(2)).expand(-1, -1,
        #                                          token_candidate_entity.size(2),
        #                                          hidden_state_ent_copy.size(2)),
        #                                          token_candidate_entity), dim=3)
        # # batch * max_len * two_hop_num
        # beta_node_attention = F.gelu(self.entity_node_linear(alpha_type_attention_softmax.unsqueeze(dim=3).expand(-1, -1, -1,
        #                                                                         one_two_hop_concate.size(3)).mul(
        #                                                                         one_two_hop_concate)).squeeze(dim=3))
        # beta_node_attention_softmax = F.softmax(beta_node_attention, dim=2)

        # # batch * max_len * entity_dim
        # # 将token_candidate_entity补全的实体给mask
        # token_candidate_entity = extended_token_candidate_entity_mask.unsqueeze(dim=3).mul(token_candidate_entity)

        # # .unsqueeze(dim=3).mul(token_candidate_entity)
        # two_hup2one_hup_fusion_entity_repre = beta_node_attention_softmax.unsqueeze(dim=2).view(-1, 1, 
        # beta_node_attention_softmax.size(2)).bmm(token_candidate_entity.view(-1, token_candidate_entity.size(2), token_candidate_entity.size(3))
        # ).squeeze(dim=1).view(beta_node_attention_softmax.size(0), beta_node_attention_softmax.size(1),
        #                       token_candidate_entity.size(3))
        # return F.gelu(self.entity_linear(torch.cat((hidden_state_ent_copy, two_hup2one_hup_fusion_entity_repre), dim=2)))

        # 方案二
        # hidden_state_ent_copy = hidden_state_ent.clone()
        # hidden_state_ent = hidden_state_ent.unsqueeze(2)
        # hidden_state_ent = hidden_state_ent.expand(hidden_state_ent.size(0), hidden_state_ent.size(1),
        #                                            token_candidate_entity.size(2), hidden_state_ent.size(3))
        # # batch * max_seq * top_K * entity_embedding
        # entity_hop_1_2_cat = torch.cat((hidden_state_ent, token_candidate_entity), dim=3)
        # attention_value_softmax = F.softmax(F.sigmoid(self.attention_linear(entity_hop_1_2_cat).squeeze(3)), dim=2)
        # attention_value_softmax = attention_value_softmax.view(-1, 1, attention_value_softmax.size(2))

        # 方案一
        # attention_value = hidden_state_ent.view(-1, 1, hidden_state_ent.size(2)).bmm(token_candidate_entity.view(-1,
        #                                                             token_candidate_entity.size(3),
        #                                                             token_candidate_entity.size(2)))
        # attention_value_softmax = F.softmax(attention_value, dim=2)


        # attention_value_softmax = attention_value_softmax.reshape(-1, 1, attention_value_softmax.size(2))
        # attention_mask_value = attention_value_softmax.mul(extended_token_candidate_entity_mask.view(-1, 1,
        #                                                             extended_token_candidate_entity_mask.size(2)))
        # two_hops_enhanced_entity_repre = attention_mask_value.bmm(
        #                     token_candidate_entity.view(-1, token_candidate_entity.size(2),
        #                     token_candidate_entity.size(3)))
        # batch_size = two_hops_enhanced_entity_repre.size(0)
        # two_hops_enhanced_entity_repre = two_hops_enhanced_entity_repre.squeeze().view(
        #                                             int(batch_size/512), 512, 100)
        # return self.entity_linear(torch.cat((hidden_state_ent_copy, two_hops_enhanced_entity_repre), dim=2))

class BertEntityPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertEntityPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(768, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertEntPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertEntPredictionHead, self).__init__()
        config_ent = copy.deepcopy(config)
        config_ent.hidden_size = 100
        self.transform = BertEntityPredictionHeadTransform(config_ent)

    def forward(self, hidden_states, candidate):
        # print('Test BertEntPredictionHead')
        hidden_states = self.transform(hidden_states)
        candidate = torch.squeeze(candidate, 0)
        return torch.matmul(hidden_states, candidate.t())

class BertEntityPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertEntityPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        # self.predictions_ent = BertEntPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score
        # prediction_scores_ent = self.predictions_ent(sequence_output, candidate)
        # return prediction_scores, seq_relationship_score, prediction_scores_ent

class cMeForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head, and
        - the entity predict head.
    Params:
        config: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    ```
    """
    def __init__(self, config, entity_embedding=None, entity_type_embedding=None,transfer_matrix=None,e_dim=None):
        super(cMeForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        # self.cls = BertEntityPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        # torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        # torch.nn.init.xavier_uniform_(self.entity_embedding.weight) # In this way, loading weights bring no benfit
        self.e_dim = e_dim
        self.bert2entity = nn.Linear(config.hidden_size,e_dim)
        self.apply(self.init_bert_weights)
        self.entity_type_embedding = entity_type_embedding
        self.entity_embedding = entity_embedding
        entity_num , entity_dim =entity_embedding.weight.size()
        self.pooler = nn.Linear(config.hidden_size,config.hidden_size)
        self.cls = nn.Linear(config.hidden_size,2)
        self.dropout = nn.Dropout(p=.15)
        # self.transfer_matrix = transfer_matrix
        # self.span_extractor = SelfAttentiveSpanExtractor(config.hidden_size)
        # self.entity_classifer = entity_embedding.weight
        # self.act = nn.GELU()
        # self.FFN1 = nn.Linear(e_dim,e_dim*4)
        # self.FFN2 = nn.Linear(e_dim*4,e_dim)
        # self.neg_sample_num = 24
        # self.entity_num = entity_num
        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        # assert(self.entity_classifer.size() == (entity_num,entity_dim))

    def forward(self, input_ids, segment_ids, input_mask, kc_entity_one_hop_ids,
                kc_entity_one_hop_types, # two_hop_entity_ids, two_hop_entity_types,
                kc_entity_se_index, kc_entity_two_hop_labels, kc_entity_out_or_in, kc_entity_two_hop_rel_types,kc_entity_two_hop_types,kc_entity_infusion_pos,labels):

        sequence_output, pooled_output = self.bert(input_ids, segment_ids, input_mask, kc_entity_one_hop_ids,
                                                   kc_entity_one_hop_types,kc_entity_two_hop_labels,kc_entity_two_hop_types,kc_entity_infusion_pos,kc_entity_se_index,
                                                   self.entity_embedding,self.entity_type_embedding,output_all_encoded_layers=False)
        pooled_output = self.pooler(pooled_output)
        pooled_output = F.gelu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.cls(pooled_output).view(-1,2)
        loss = F.cross_entropy(logits,labels.view(-1))
        return loss,logits

class cMeForSoftMaxNER(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head, and
        - the entity predict head.
    Params:
        config: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    ```
    """
    def __init__(self, config, entity_embedding=None, entity_type_embedding=None,transfer_matrix=None,e_dim=None,labels_num=None):
        super(cMeForSoftMaxNER, self).__init__(config)
        self.bert = BertModel(config)
        
        self.e_dim = e_dim
        self.bert2entity = nn.Linear(config.hidden_size,e_dim)
        self.apply(self.init_bert_weights)
        self.entity_type_embedding = entity_type_embedding
        self.entity_embedding = entity_embedding
        # entity_num , entity_dim =entity_embedding.weight.size()
        self.cls = ClsHead(config,labels_num)
        # self.ffn = nn.Linear(config.hidden_size, config.hidden_size)
        self.labels_num = labels_num
        self.dropout = nn.Dropout(p=.1)
        

    def forward(self, input_ids, segment_ids, input_mask, kc_entity_one_hop_ids,
                kc_entity_one_hop_types,
                kc_entity_se_index, kc_entity_two_hop_labels, kc_entity_out_or_in, kc_entity_two_hop_rel_types,kc_entity_two_hop_types,kc_entity_infusion_pos,labels):

        sequence_output, pooled_output = self.bert(input_ids, segment_ids, input_mask, kc_entity_one_hop_ids,
                                                   kc_entity_one_hop_types,kc_entity_two_hop_labels,kc_entity_two_hop_types,kc_entity_infusion_pos,kc_entity_se_index,
                                                   self.entity_embedding,self.entity_type_embedding,output_all_encoded_layers=False)

        x,v = self.cls(sequence_output)
        logits = v.view(-1,self.labels_num)
        loss = F.cross_entropy(logits,labels.view(-1),ignore_index=-1)
        return loss,logits



class cMeForRE(BertPreTrainedModel):
    def __init__(self, config, entity_embedding=None, entity_type_embedding=None,transfer_matrix=None,e_dim=None,label_num = None):
        super(cMeForRE, self).__init__(config)
        self.bert = BertModel(config)
        # self.cls = BertEntityPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        # torch.nn.init.xavier_uniform_(self.entity_embedding.weight)
        # torch.nn.init.xavier_uniform_(self.entity_embedding.weight) # In this way, loading weights bring no benfit
        self.e_dim = e_dim
        self.bert2entity = nn.Linear(config.hidden_size,e_dim)
        self.apply(self.init_bert_weights)
        self.entity_type_embedding = entity_type_embedding
        self.entity_embedding = entity_embedding
        entity_num , entity_dim =entity_embedding.weight.size()
        # self.pooler = nn.Linear(config.hidden_size,config.hidden_size)
        self.cls = ClsHead(config,label_num)
        self.dropout = nn.Dropout(p=.05)
        self.label_num = label_num

    def forward(self, input_ids, segment_ids, input_mask, kc_entity_one_hop_ids,
                kc_entity_one_hop_types, # two_hop_entity_ids, two_hop_entity_types,
                kc_entity_se_index, kc_entity_two_hop_labels, kc_entity_out_or_in, kc_entity_two_hop_rel_types,kc_entity_two_hop_types,kc_entity_infusion_pos,labels):

        sequence_output, pooled_output = self.bert(input_ids, segment_ids, input_mask, kc_entity_one_hop_ids,
                                                   kc_entity_one_hop_types,kc_entity_two_hop_labels,
                                                   kc_entity_two_hop_types,kc_entity_infusion_pos,kc_entity_se_index,
                                                   self.entity_embedding,self.entity_type_embedding,output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        # logits = self.cls(pooled_output).view(-1,self.label_num)
        x,v = self.cls(pooled_output)
        # print(x.size(),v.size())
        logits = v.view(-1,self.label_num)
        # print(labels.view(-1).size(),logits.size())
        loss = F.cross_entropy(logits,labels.view(-1))
        return loss,logits

class cMeForIR(BertPreTrainedModel):
    def __init__(self, config, entity_embedding=None, entity_type_embedding=None,transfer_matrix=None,e_dim=None,label_num = None):
        super(cMeForIR, self).__init__(config)
        self.bert = BertModel(config)
        self.e_dim = e_dim
        self.bert2entity = nn.Linear(config.hidden_size,e_dim)
        self.apply(self.init_bert_weights)
        self.entity_type_embedding = entity_type_embedding
        self.entity_embedding = entity_embedding
        entity_num , entity_dim =entity_embedding.weight.size()
        self.cls = nn.Linear(config.hidden_size,1)
        self.dropout = nn.Dropout(p=.1)
        # self.label_num = label_num

    def forward(self, input_ids, segment_ids, input_mask, kc_entity_one_hop_ids,
                kc_entity_one_hop_types, # two_hop_entity_ids, two_hop_entity_types,
                kc_entity_se_index, kc_entity_two_hop_labels, kc_entity_out_or_in, kc_entity_two_hop_rel_types,kc_entity_two_hop_types,kc_entity_infusion_pos,labels):

        sequence_output, pooled_output = self.bert(input_ids, segment_ids, input_mask, kc_entity_one_hop_ids,
                                                   kc_entity_one_hop_types,kc_entity_two_hop_labels,kc_entity_two_hop_types,kc_entity_infusion_pos,kc_entity_se_index,
                                                   self.entity_embedding,self.entity_type_embedding,output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = F.sigmoid(self.cls(pooled_output))
        return (logits,)


###### /* End of Insertion */
