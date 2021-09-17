# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unicodedata
import argparse
import collections
import logging
import json
import math
import os
import random
import six
from tqdm import tqdm, trange
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

import pytorch_pretrained_bert.tokenization as tokenization
from pytorch_pretrained_bert.modeling import BertConfig, BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam

from multiprocessing import Pool
from functools import partial
from MRC_pakage.evaluate import score
from shutil import copyfile
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.Trie import Trie
from random import Random as Rd
from MRC_pakage.Data_process import feature_list_expand

# pre-training 的KG设置：全局
entity_file = open('/home/dell/PycharmProjects/tmp_czr/KIM/OpenKE-master/ch_entity2id.txt', 'r',
                   encoding='utf-8')  # Note that we must add the first entity as EMPETY.
entity_dict = {}
entity_file.readline()
for line in entity_file:
    name, idx = line.rstrip().split('\t')
    entity_dict[name] = int(idx)
ww_tree = Trie()
MAX_PRE_ENTITY_MASK = 100
mask_count = dict.fromkeys(entity_dict.keys(), MAX_PRE_ENTITY_MASK)

for key in entity_dict.keys():
    if (len(key) > 1 and not key.isdigit()):
        ww_tree.insert(key)
    # entity_dict[key] += 1  # For the zero embedding is used as Pad ebmedding.
entity_file.close()
entity_dict_index2str = {value: key for key, value in entity_dict.items()}

# keys = ['ent_embeddings', 'rel_embeddings']
js_file = open('/home/dell/PycharmProjects/tmp_czr/KIM/OpenKE-master/temp/embedding.vec.json', 'r',
               encoding='utf-8')  # Note that we must add the first entity as EMPTY.
embedding_list = json.load(js_file)
js_file.close()
embedding_list = embedding_list['ent_embeddings']

vecs = []
vecs.append([0] * 100)  # CLS
for vec in embedding_list:
    vecs.append(vec)
embed = torch.FloatTensor(vecs)
embed = torch.nn.Embedding.from_pretrained(embed)
del vec, js_file, entity_file
rng = Rd(43)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class SquadTrainExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class SquadDevExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 answerList):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.answerList = answerList
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.answerList:
            s += ", start_position: %d" % (self.answerList)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 entity_ids,
                 entity_ids_type,
                 entity_ids_mapping,
                 two_hop_entity_type,
                 two_hop_entity_type_mask,
                 two_hop_entity_type_index_select,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.entity_ids = entity_ids
        self.entity_ids_type = entity_ids_type
        self.entity_ids_mapping = entity_ids_mapping
        self.two_hop_entity_type = two_hop_entity_type
        self.two_hop_entity_type_mask = two_hop_entity_type_mask
        self.two_hop_entity_type_index_select = two_hop_entity_type_index_select

        self.start_position = start_position
        self.end_position = end_position

#
# def read_squad_dev_examples(entry, tokenizer, is_training):
#     """
#     Read a SQuAD json file into a list of SquadExample.
#     这个函数将input_data[i]["paragraphs"]["context"]变成一个list，词的list
#     然后遍历"qas"，对于每一个qa，提取
#     {
#         qas_id: qa['id'],
#         question_text: qa["question"],
#         orig_answer_text: answer["text"],
#         start_position: start_position,
#         end_position: end_position
#     }
#     """
#     # with open(args.train_file, "r") as reader:
#     #     input_data = json.load(reader)["data"]
#     # input_data = input_data[:20]
#     def is_whitespace(c):
#         if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
#             return True
#         return False
#
#     def is_control(char):
#         """Checks whether `chars` is a control character."""
#         # These are technically control characters but we count them as whitespace
#         # characters.
#         if char == "\t" or char == "\n" or char == "\r":
#             return False
#         cat = unicodedata.category(char)
#         if cat.startswith("C"):
#             return True
#         return False
#
#     def clean_text(text):
#         """Performs invalid character removal and whitespace cleanup on text."""
#         output = []
#         for char in text:
#             cp = ord(char)
#             if cp == 0 or cp == 0xfffd or is_control(char):
#                 continue
#             if is_whitespace(char):
#                 output.append(" ")
#             else:
#                 output.append(char)
#         return "".join(output)
#
#     examples = []
#     # for entry in tqdm(input_data):
#     for paragraph in entry["paragraphs"]:
#             paragraph_text = " ".join(tokenization.whitespace_tokenize(clean_text(paragraph["context"])))
#
#             for qa in paragraph["qas"]:
#                 doc_tokens = tokenizer.basic_tokenizer.tokenize(paragraph_text)
#                 qas_id = qa["id"]
#                 question_text = qa["question"]
#                 if is_training:
#                     if len(qa["answers"]) != 3:
#                         raise ValueError(
#                             "For dev, each question should have exactly 1 answer.")
#                     orig_answer_text_list = [qa["answers"][0]["text"], qa["answers"][1]["text"], qa["answers"][2]["text"]]
#
#                 example = SquadDevExample(
#                     qas_id=qas_id,
#                     question_text=question_text,
#                     doc_tokens=doc_tokens,
#                     answerList=orig_answer_text_list)
#                 examples.append(example)
#     return examples

def read_squad_train_examples(entry, tokenizer, is_training):
    """
    Read a SQuAD json file into a list of SquadExample.
    这个函数将input_data[i]["paragraphs"]["context"]变成一个list，词的list
    然后遍历"qas"，对于每一个qa，提取
    {
        qas_id: qa['id'],
        question_text: qa["question"],
        orig_answer_text: answer["text"],
        start_position: start_position,
        end_position: end_position
    }
    """
    # with open(args.train_file, "r") as reader:
    #     input_data = json.load(reader)["data"]
    # input_data = input_data[:20]
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def is_control(char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def clean_text(text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    examples = []
    for paragraph in entry["paragraphs"]:
            paragraph_text = " ".join(tokenization.whitespace_tokenize(clean_text(paragraph["context"])))
            for qa in paragraph["qas"]:
                doc_tokens = tokenizer.basic_tokenizer.tokenize(paragraph_text)
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                if is_training:
                    if len(qa["answers"]) != 1:
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    if len(orig_answer_text) == 0:
                        continue
                    cleaned_answer_text = "".join(tokenizer.basic_tokenizer.tokenize(orig_answer_text))
                    ori_start_position = "".join(doc_tokens).find(cleaned_answer_text)
                    if ori_start_position == -1:
                        print("Could not find answer: '%s' vs. '%s'",  ''.join(doc_tokens), cleaned_answer_text)
                        continue
                    ori_end_position = ori_start_position + len(cleaned_answer_text) - 1
                    char_to_word_offset = {}
                    start = 0
                    for idx, token in enumerate(doc_tokens):
                        for _ in token:
                            char_to_word_offset[start] = idx
                            start += 1
                    start_position = char_to_word_offset[ori_start_position]
                    try:
                        end_position = char_to_word_offset[ori_end_position]
                    except KeyError:
                        continue
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = "".join(doc_tokens[start_position:(end_position + 1)])
                    if actual_text != cleaned_answer_text:
                        # print("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                        continue
                    orig_answer_text = cleaned_answer_text
                example = SquadTrainExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                examples.append(example)
    return examples

def judge_two_entity(temp_entity_ids_mapping, entity2type, type_embedd, entity_dict_init):
        # 方案二：做成按照类别分类的type attention聚合
        entity_dict_reverse = {value: key for key, value in entity_dict_init.items()}
        two_hop_entity_type_list = []
        # 二阶实体类别初始化
        for i in range(len(temp_entity_ids_mapping)):
            two_hop_entity_type_list.append([0 for _ in range(type_embedd.num_embeddings)])
        two_hop_entity_type_mask = [0 for _ in range(type_embedd.num_embeddings)]
        two_hop_entity_type_index_select = []
        for index, ids in enumerate(temp_entity_ids_mapping):
            if ids == -1:
                two_hop_entity_type_index_select.append(4)
                two_hop_entity_type_list[index][4] = 1
                # if two_hop_entity_type_mask[4] == 1:
                #     pass
                # else:
                # 缺省的entity type 永远都把它mask
                two_hop_entity_type_mask[4] = 0
            else:
                if entity2type[entity_dict_reverse[ids]] == '药品':
                    two_hop_entity_type_index_select.append(0)
                    two_hop_entity_type_list[index][0] = 1
                    if two_hop_entity_type_mask[0] == 1:
                        pass
                    else:
                        two_hop_entity_type_mask[0] = 1
                elif entity2type[entity_dict_reverse[ids]] == '疾病':
                    two_hop_entity_type_index_select.append(1)
                    two_hop_entity_type_list[index][1] = 1
                    if two_hop_entity_type_mask[1] == 1:
                        pass
                    else:
                        two_hop_entity_type_mask[1] = 1
                elif entity2type[entity_dict_reverse[ids]] == '症状':
                    two_hop_entity_type_index_select.append(2)
                    two_hop_entity_type_list[index][2] = 1
                    if two_hop_entity_type_mask[2] == 1:
                        pass
                    else:
                        two_hop_entity_type_mask[2] = 1
                else:
                    two_hop_entity_type_index_select.append(3)
                    two_hop_entity_type_list[index][3] = 1
                    if two_hop_entity_type_mask[3] == 1:
                        pass
                    else:
                        two_hop_entity_type_mask[3] = 1
        # 三个变量分别返回的数值意义为：
        # two_hop_entity_type_list： 由于不能直接用一个vector进行Tensor操作，将vector拆开成N个候选二阶实体列表组成，
        # 元素对应位置为1的表示类型表中那个index的类型是这个二阶实体的类型。
        # two_hop_entity_type_mask: 因为最后要进行softmax计算，所以要将softmax到底需要那几个实体类型进行mask，计算精确。
        # two_hop_entity_type_index_select： 由于每个二阶实体对应的实体类型是随机的实体类型，所以用这个列表进行tensor变量的select
        # 操作，组成最后的type attention数值。
        return two_hop_entity_type_list, two_hop_entity_type_mask, two_hop_entity_type_index_select

def convert_examples_to_features(train_examples, args, node2entity, entity_dict_init,
                                 entity_type, type_embedd, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # print(tokenizer.vocab['[CLS]'])
    features = []
    for (example_index, example) in tqdm(enumerate(tqdm(train_examples)), desc='feature transform....'):
        query_tokens = tokenizer.tokenize(example.question_text)
        # context_tokens = tokenizer.tokenize(example.context)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # 将原文变成token之后，token和原始文本index的对应关系
        tok_to_orig_index = []
        # 将原文变成token之后，原始文本和token的index的对应关系
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training:
            # 从原始的start_position转换到变为token之后的index
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            # 如果length是小于max_tokens_for_doc的话，那么就会直接break
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            # segment_ids中0代表[CLS]、第一个[SEP]和query_tokens，1代表doc和第二个[SEP]
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                # len(tokens)为[CLS]+query_tokens+[SEP]的大小，应该是doc_tokens第i个token
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                if (example.start_position < doc_start or
                        example.end_position < doc_start or
                        example.start_position > doc_end or example.end_position > doc_end):
                    continue

                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
            entity_ids, entity_ids_type, entity_ids_mapping, two_hop_entity_type, \
            two_hop_entity_type_mask, two_hop_entity_type_index_select = two_hop_entity_data(args,
                                                                                             tokens,
                                                                                             node2entity,
                                                                                             entity_dict_init,
                                                                                             entity_type,
                                                                                             type_embedd)
            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    entity_ids=entity_ids,
                    entity_ids_type=entity_ids_type,
                    entity_ids_mapping=entity_ids_mapping,
                    two_hop_entity_type=two_hop_entity_type,
                    two_hop_entity_type_mask=two_hop_entity_type_mask,
                    two_hop_entity_type_index_select=two_hop_entity_type_index_select,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position))
            unique_id += 1

    return features

def create_wwm_lm_predictions(tokens):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    chars = [token for token in tokens]
    entity_ids = []  # This list would like [ [0] , [1,2] , [3,4,5]] to pack wwm word toghter
    # We use it to 保存分词预测和特征词原子词相互预测的位置
    cand_index = 0
    assert (chars[0] == '[CLS]')
    while (cand_index < len(chars)):
        if (isSkipToken(chars[cand_index])):
            entity_ids.append(-1)
        elif (ww_tree.startsWith(chars[cand_index])):
            c = ww_tree.get_lengest_match(chars, cand_index)
            if (c == None):
                entity_ids.append(-1)
            else:
                word = ''.join(chars[cand_index:c + 1])
                assert (word in entity_dict)
                # 考虑到里面有一个嵌套实体(其实理论上有多个，但是目前只考虑实体里面有一个嵌套实体)
                nested_end = -1
                nested_start = -1
                for temp_i in range(1, len(word), 1):
                    nested_index = ww_tree.get_lengest_match(chars, cand_index+temp_i)
                    if nested_index != None:
                        nested_end = nested_index
                        if (nested_end >=0) & (nested_end<=c):
                            nested_start = cand_index + temp_i
                            break
                    else:
                        pass
                if (nested_start >= 0) & (nested_end >=0) & (nested_start < nested_end):
                    # 里面检测到一个嵌套实体
                    nested_word = ''.join(chars[nested_start:nested_end + 1])
                    for temp_ii in range(cand_index, nested_start, 1):
                        entity_ids.append(entity_dict[word])
                    for temp_ii in range(nested_start, nested_end+1, 1):
                        entity_ids.append(entity_dict[nested_word])
                    for temp_ii in range(nested_end+1, c+1, 1):
                        entity_ids.append(entity_dict[word])
                else:
                    # 给匹配到的实体所有token位置都增强相同的实体
                    for temp_i in range(len(word)):
                        entity_ids.append(entity_dict[word])
                cand_index += len(word)
                continue
        else:
            entity_ids.append(-1)
        cand_index += 1
    return entity_ids

def isSkipToken(token):
    return token == "[CLS]" or token == "[SEP]" or (not token.isalnum() and len(token) == 1)

def two_hop_entity_data(args, tokens, node2entity, entity_dict_init, entity_type, type_embedd):
    entity_ids = create_wwm_lm_predictions(tokens)
    # We use the sop task. the simple implement that.
    entity_mapping = node2entity

    # 每一个token对应的个数由
    entity_ids_mapping = []
    entity_ids_mapping_mask = []

    two_hop_entity_type_list = []
    two_hop_entity_type_mask_list = []
    two_hop_entity_type_index_select_list = []
    entity_ids_type = []
    for index_id, entity_id in enumerate(entity_ids):
        temp_entity_ids_mapping = []
        temp_entity_ids_mapping_mask = []
        if entity_id == -1:
            entity_ids_type.append(4)
            for i in range(args.two_hop_entity_num):
                temp_entity_ids_mapping.append(-1)
                temp_entity_ids_mapping_mask.append(0)
        else:
            # 给一阶实体增加entity type
            if entity_type[entity_dict_index2str[entity_id]] == '药品':
                entity_ids_type.append(0)
            elif entity_type[entity_dict_index2str[entity_id]] == '疾病':
                entity_ids_type.append(1)
            elif entity_type[entity_dict_index2str[entity_id]] == '症状':
                entity_ids_type.append(2)
            else:
                entity_ids_type.append(3)

            if len(entity_mapping[entity_dict_index2str[entity_id]]) == 0:
                for i in range(args.two_hop_entity_num):
                    temp_entity_ids_mapping.append(-1)
                    temp_entity_ids_mapping_mask.append(0)
            else:
                if len(entity_mapping[entity_dict_index2str[entity_id]]) > args.two_hop_entity_num:
                    # 因为相关的实体列表在存储的时候，感觉重要的周围实体在list后面，前面的是entity对应的标签，所以使用这个倒叙进行输出
                    for two_hop_entity in reversed(node2entity[entity_dict_index2str[entity_id]]):
                        if len(temp_entity_ids_mapping) != args.two_hop_entity_num:
                            temp_entity_ids_mapping.append(entity_dict_init[two_hop_entity])
                            temp_entity_ids_mapping_mask.append(1)
                        if len(temp_entity_ids_mapping) == args.two_hop_entity_num:
                            break
                elif len(entity_mapping[entity_dict_index2str[entity_id]]) < args.two_hop_entity_num:
                    for two_hop_entity in node2entity[entity_dict_index2str[entity_id]]:
                        temp_entity_ids_mapping.append(entity_dict_init[two_hop_entity])
                        temp_entity_ids_mapping_mask.append(1)
                    for i in range(args.two_hop_entity_num - len(entity_mapping[entity_dict_index2str[entity_id]])):
                        temp_entity_ids_mapping.append(-1)
                        temp_entity_ids_mapping_mask.append(0)
                    assert len(temp_entity_ids_mapping) == args.two_hop_entity_num
                else:
                    for two_hop_entity in node2entity[entity_dict_index2str[entity_id]]:
                        temp_entity_ids_mapping.append(entity_dict_init[two_hop_entity])
                        temp_entity_ids_mapping_mask.append(1)

        entity_ids_mapping.append(temp_entity_ids_mapping)
        entity_ids_mapping_mask.append(temp_entity_ids_mapping_mask)

        # 加入二阶实体类型的 representation 和 label 标签
        two_hop_entity_type, two_hop_entity_type_mask, two_hop_entity_type_index_select = judge_two_entity(
                                                                            temp_entity_ids_mapping,
                                                                            entity_type,
                                                                            type_embedd,
                                                                            entity_dict)
        two_hop_entity_type_list.append(two_hop_entity_type)
        two_hop_entity_type_mask_list.append(two_hop_entity_type_mask)
        two_hop_entity_type_index_select_list.append(two_hop_entity_type_index_select)

    # 补全各个实体参数的长度
    entity_ids, entity_ids_type, entity_ids_mapping, \
    two_hop_entity_type_list, two_hop_entity_type_mask_list, \
    two_hop_entity_type_index_select_list = two_hop_entity_parameter_complete(args,
                                                                              type_embedd,
                                                                              entity_ids,
                                                                              entity_ids_type,
                                                                              entity_ids_mapping,
                                                                              two_hop_entity_type_list,
                                                                              two_hop_entity_type_mask_list,
                                                                              two_hop_entity_type_index_select_list)

    return entity_ids, entity_ids_type, entity_ids_mapping, \
           two_hop_entity_type_list, \
           two_hop_entity_type_mask_list, two_hop_entity_type_index_select_list

def two_hop_entity_parameter_complete(args, type_embedd, entity_ids, entity_ids_type, entity_ids_mapping,
                                      two_hop_entity_type_list, two_hop_entity_type_mask_list,
                                                                              two_hop_entity_type_index_select_list):
    add_default_value = args.max_seq_length - len(entity_ids_mapping)
    for _ in range(add_default_value):
        entity_ids_type.append(4)
        number_hop_list = [-1 for _ in range(args.two_hop_entity_num)]
        entity_ids_mapping.append(number_hop_list)

        number_default_two_hop_list = [0 for _ in range(type_embedd.num_embeddings - 1)]
        number_default_two_hop_list.append(1)
        two_hop_entity_type_list.append([number_default_two_hop_list for _ in range(args.two_hop_entity_num)])

        number_default_two_hop_type = [0 for _ in range(type_embedd.num_embeddings - 1)]
        number_default_two_hop_type.append(1)
        two_hop_entity_type_mask_list.append(number_default_two_hop_type)

        two_hop_entity_type_index_select_list.append(
            [(type_embedd.num_embeddings - 1) for _ in range(args.two_hop_entity_num)])
        entity_ids_type.append(4)

    entity_array = np.full(args.max_seq_length, dtype=np.int, fill_value=-1)
    entity_array[:len(entity_ids)] = entity_ids

    entity_ids_type = np.array(entity_ids_type)
    entity_ids_mapping = np.array(entity_ids_mapping)
    two_hop_entity_type_list = np.array(two_hop_entity_type_list)
    two_hop_entity_type_mask_list = np.array(two_hop_entity_type_mask_list)
    two_hop_entity_type_index_select_list = np.array(two_hop_entity_type_index_select_list)


    return entity_array, entity_ids_type, entity_ids_mapping, two_hop_entity_type_list, \
           two_hop_entity_type_mask_list, two_hop_entity_type_index_select_list


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index



RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, verbose_logging):
    """Write final predictions to the json file."""
    # logger.info("Writing predictions to: %s" % (output_prediction_file))
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = "".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = "".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json
    return all_predictions, all_nbest_json


def write_to_file(all_predictions, all_nbest_json, output_prediction_file, output_nbest_file):
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = "".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    # 通过logits的值进行排序，然后把前n_best_size保存下来
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def run_evaluate(args, model, eval_features, device, eval_examples):
    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.dev_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    try:
        with tqdm(eval_dataloader) as t:
            for input_ids, input_mask, segment_ids, example_indices in t:
            # for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader):
            # if len(all_results) % 1000 == 0:
            #     logger.info("Processing example: %d" % (len(all_results)))
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                with torch.no_grad():
                    batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    all_results.append(RawResult(unique_id=unique_id,
                                                 start_logits=start_logits,
                                                 end_logits=end_logits))
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()
    # output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    # output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    # predict, nbest = write_predictions(eval_examples, eval_features, all_results,
    #                                    args.n_best_size, args.max_answer_length,
    #                                    args.do_lower_case, output_prediction_file,
    #                                    output_nbest_file, args.verbose_logging)
    predict, nbest = write_predictions(eval_examples, eval_features, all_results,
                                       args.n_best_size, args.max_answer_length,
                                       args.do_lower_case, args.verbose_logging)
    return predict

def entity_info(args):
    # 实体类型：{'西药', '疾病', '疾', '分类', '中药', '检查', '中西症状', '药品', '科室', '中医症状', '部位', '西医症状', '症状', '检验'}
    with open(args.entity_type, 'rb') as fo:
        entity_type_dict = pickle.load(fo, encoding='utf-8')
    with open(args.node2entity, 'rb') as fo:
        node2entity = pickle.load(fo, encoding='utf-8')
    new_entity_type_dict = {}
    total_entity = []
    for one_hop_entity, two_hop_entity_list in node2entity.items():
        total_entity.append(one_hop_entity)
        total_entity += two_hop_entity_list
    for entity in set(total_entity):
        if entity not in new_entity_type_dict.keys():
            if len(entity_type_dict[entity]) == 0:
                # 如果没有entity type的实体默认为 ”症状“
                new_entity_type_dict[entity] = '症状'
            else:
                # 如果有多个实体类型，直接取第一个
                new_entity_type_dict[entity] = entity_type_dict[entity][0]

    # 将实体类型进行合并，保持实体类型的个数尽可能集中，同时显存占用少。
    # 经过实体数据统计，可以分为： 药品（包含中药，西药，药品），症状（中医症状，西医症状，症状），疾病，其他
    combine_entity_type_dict = {}
    for key, value in new_entity_type_dict.items():
        if value == '药品' or value == '中药' or value == '西药':
            combine_entity_type_dict[key] = '药品'
        elif value == '中医症状' or value == '西医症状' or value == '症状':
            combine_entity_type_dict[key] = '症状'
        elif value == '疾病':
            combine_entity_type_dict[key] = '疾病'
        else:
            combine_entity_type_dict[key] = '其他'

    return entity_type_dict, node2entity, combine_entity_type_dict

def entity_type_initialize(combine_entity_type_dict):
    # predefined_entity_type = ['药品', '疾病', '症状', '其他', '缺省']
    medicine = np.zeros(100)
    medicine_count = 0
    illness = np.zeros(100)
    illness_count = 0
    zhengzhuang = np.zeros(100)
    zhengzhuang_count = 0
    other = np.zeros(100)
    other_count = 0
    defualt = np.zeros(100)

    for entity_name, entity_type in combine_entity_type_dict.items():
        if entity_type == '药品':
            medicine += embedding_list[entity_dict[entity_name]]
            medicine_count += 1
        elif entity_type == '疾病':
            illness += embedding_list[entity_dict[entity_name]]
            illness_count += 1
        elif entity_type == '症状':
            zhengzhuang += embedding_list[entity_dict[entity_name]]
            zhengzhuang_count += 1
        elif entity_type == '其他':
            other += embedding_list[entity_dict[entity_name]]
            other_count += 1
        # else:
        #     defualt += embedding_list[entity_dict[entity_name]]
        #     defualt_count += 1
    medicine = np.divide(medicine, medicine_count)
    illness = np.divide(illness, illness_count)
    zhengzhuang = np.divide(zhengzhuang, zhengzhuang_count)
    other = np.divide(other, other_count)

    return_result = np.vstack((medicine, illness, zhengzhuang, other, defualt))
    return_result = torch.from_numpy(return_result).float()
    return_result = torch.nn.Embedding.from_pretrained(return_result)
    return return_result

# class OurCMBERTDataset(Dataset):
#     def __init__(self, args, data_path, tokenizer, node2entity, entity_dict, entity_type, type_embedd):
#         self.unique_id = 1000000000
#         self.example_index = 0
#         self.args = args
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.node2entity = node2entity
#         self.entity_dict = entity_dict
#         self.examples = []
#         self.entity_type = entity_type
#         self.type_embedd = type_embedd
#         self.entity_dict_reverse = {value: key for key, value in entity_dict.items()}
#         self.__read_data__()
#     def __mrc_feature__(self, example):
#         query_tokens = self.tokenizer.tokenize(example.question_text)
#         # context_tokens = tokenizer.tokenize(example.context)
#         if len(query_tokens) > self.args.max_query_length:
#             query_tokens = query_tokens[0:self.args.max_query_length]
#
#         # 将原文变成token之后，token和原始文本index的对应关系
#         tok_to_orig_index = []
#         # 将原文变成token之后，原始文本和token的index的对应关系
#         orig_to_tok_index = []
#         all_doc_tokens = []
#         for (i, token) in enumerate(example.doc_tokens):
#             orig_to_tok_index.append(len(all_doc_tokens))
#             sub_tokens = self.tokenizer.tokenize(token)
#             for sub_token in sub_tokens:
#                 tok_to_orig_index.append(i)
#                 all_doc_tokens.append(sub_token)
#
#         tok_start_position = None
#         tok_end_position = None
#         if self.args.do_train:
#             # 从原始的start_position转换到变为token之后的index
#             tok_start_position = orig_to_tok_index[example.start_position]
#             if example.end_position < len(example.doc_tokens) - 1:
#                 tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
#             else:
#                 tok_end_position = len(all_doc_tokens) - 1
#             (tok_start_position, tok_end_position) = _improve_answer_span(
#                 all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
#                 example.orig_answer_text)
#
#         # The -3 accounts for [CLS], [SEP] and [SEP]
#         max_tokens_for_doc = self.args.max_seq_length - len(query_tokens) - 3
#
#         # We can have documents that are longer than the maximum sequence length.
#         # To deal with this we do a sliding window approach, where we take chunks
#         # of the up to our max length with a stride of `doc_stride`.
#         _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
#             "DocSpan", ["start", "length"])
#         doc_spans = []
#         start_offset = 0
#         while start_offset < len(all_doc_tokens):
#             length = len(all_doc_tokens) - start_offset
#             if length > max_tokens_for_doc:
#                 length = max_tokens_for_doc
#             doc_spans.append(_DocSpan(start=start_offset, length=length))
#             # 如果length是小于max_tokens_for_doc的话，那么就会直接break
#             if start_offset + length == len(all_doc_tokens):
#                 break
#             start_offset += min(length, self.args.doc_stride)
#
#         for (doc_span_index, doc_span) in enumerate(doc_spans):
#             tokens = []
#             token_to_orig_map = {}
#             token_is_max_context = {}
#             # segment_ids中0代表[CLS]、第一个[SEP]和query_tokens，1代表doc和第二个[SEP]
#             segment_ids = []
#             tokens.append("[CLS]")
#             segment_ids.append(0)
#             for token in query_tokens:
#                 tokens.append(token)
#                 segment_ids.append(0)
#             tokens.append("[SEP]")
#             segment_ids.append(0)
#
#             for i in range(doc_span.length):
#                 split_token_index = doc_span.start + i
#                 # len(tokens)为[CLS]+query_tokens+[SEP]的大小，应该是doc_tokens第i个token
#                 token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
#
#                 is_max_context = _check_is_max_context(doc_spans, doc_span_index,
#                                                        split_token_index)
#                 token_is_max_context[len(tokens)] = is_max_context
#                 tokens.append(all_doc_tokens[split_token_index])
#                 segment_ids.append(1)
#             tokens.append("[SEP]")
#             segment_ids.append(1)
#
#             input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#
#             # The mask has 1 for real tokens and 0 for padding tokens. Only real
#             # tokens are attended to.
#             input_mask = [1] * len(input_ids)
#
#             # Zero-pad up to the sequence length.
#             while len(input_ids) < self.args.max_seq_length:
#                 input_ids.append(0)
#                 input_mask.append(0)
#                 segment_ids.append(0)
#
#             assert len(input_ids) == self.args.max_seq_length
#             assert len(input_mask) == self.args.max_seq_length
#             assert len(segment_ids) == self.args.max_seq_length
#
#             start_position = None
#             end_position = None
#             if self.args.do_train:
#                 # For training, if our document chunk does not contain an annotation
#                 # we throw it out, since there is nothing to predict.
#                 doc_start = doc_span.start
#                 doc_end = doc_span.start + doc_span.length - 1
#                 if (example.start_position < doc_start or
#                         example.end_position < doc_start or
#                         example.start_position > doc_end or example.end_position > doc_end):
#                     continue
#
#                 doc_offset = len(query_tokens) + 2
#                 start_position = tok_start_position - doc_start + doc_offset
#                 end_position = tok_end_position - doc_start + doc_offset
#
#         return doc_span_index
#     def __read_data__(self):
#         def is_whitespace(c):
#             if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
#                 return True
#             return False
#
#         def is_control(char):
#             """Checks whether `chars` is a control character."""
#             # These are technically control characters but we count them as whitespace
#             # characters.
#             if char == "\t" or char == "\n" or char == "\r":
#                 return False
#             cat = unicodedata.category(char)
#             if cat.startswith("C"):
#                 return True
#             return False
#
#         def clean_text(text):
#             """Performs invalid character removal and whitespace cleanup on text."""
#             output = []
#             for char in text:
#                 cp = ord(char)
#                 if cp == 0 or cp == 0xfffd or is_control(char):
#                     continue
#                 if is_whitespace(char):
#                     output.append(" ")
#                 else:
#                     output.append(char)
#             return "".join(output)
#
#         examples = []
#         with open(self.data_path, "r") as reader:
#             input_train_data = json.load(reader)["data"]
#             input_train_data = input_train_data[:50]
#         for paragraph in input_train_data["paragraphs"]:
#             paragraph_text = " ".join(tokenization.whitespace_tokenize(clean_text(paragraph["context"])))
#             for qa in paragraph["qas"]:
#                 doc_tokens = self.tokenizer.tokenize(paragraph_text)
#                 qas_id = qa["id"]
#                 question_text = qa["question"]
#                 start_position = None
#                 end_position = None
#                 orig_answer_text = None
#                 if self.args.do_train:
#                     if len(qa["answers"]) != 1:
#                         raise ValueError(
#                             "For training, each question should have exactly 1 answer.")
#                     answer = qa["answers"][0]
#                     orig_answer_text = answer["text"]
#                     if len(orig_answer_text) == 0:
#                         continue
#                     cleaned_answer_text = "".join(self.tokenizer.tokenize(orig_answer_text))
#                     ori_start_position = "".join(doc_tokens).find(cleaned_answer_text)
#                     if ori_start_position == -1:
#                         print("Could not find answer: '%s' vs. '%s'", ''.join(doc_tokens), cleaned_answer_text)
#                         continue
#                     ori_end_position = ori_start_position + len(cleaned_answer_text) - 1
#                     char_to_word_offset = {}
#                     start = 0
#                     for idx, token in enumerate(doc_tokens):
#                         for _ in token:
#                             char_to_word_offset[start] = idx
#                             start += 1
#                     start_position = char_to_word_offset[ori_start_position]
#                     try:
#                         end_position = char_to_word_offset[ori_end_position]
#                     except KeyError:
#                         continue
#                     # Only add answers where the text can be exactly recovered from the
#                     # document. If this CAN'T happen it's likely due to weird Unicode
#                     # stuff so we will just skip the example.
#                     #
#                     # Note that this means for training mode, every example is NOT
#                     # guaranteed to be preserved.
#                     actual_text = "".join(doc_tokens[start_position:(end_position + 1)])
#                     if actual_text != cleaned_answer_text:
#                         # print("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
#                         continue
#                     orig_answer_text = cleaned_answer_text
#                 example = SquadTrainExample(
#                     qas_id=qas_id,
#                     question_text=question_text,
#                     doc_tokens=doc_tokens,
#                     orig_answer_text=orig_answer_text,
#                     start_position=start_position,
#                     end_position=end_position)
#                 examples.append(example)
#         self.examples = examples
#     def __getitem__(self, index):
#         example = self.examples[index]
#         self.example_index += 1


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--model_name_or_path', default='/ecnu105/ztl/our_medBERT_version2_knowledge_diff/MRC_pakage/pytorch_pretrained_bert')
    parser.add_argument("--bert_config_file", default='/ecnu105/ztl/our_medBERT_version2_knowledge_diff/MRC_pakage/pytorch_pretrained_bert/config.json', type=str,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default='/ecnu105/ztl/our_medBERT_version2_knowledge_diff/MRC_pakage/pytorch_pretrained_bert/vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default='/ecnu105/ztl/our_medBERT_version2_knowledge_diff/MRC_pakage/output', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--processed_data", default='/ecnu105/ztl/our_medBERT_version2_knowledge_diff/MRC_pakage/processed_data', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true',
                        help="Whether to run eval on the dev set.")

    ## Other parameters
    parser.add_argument("--model_dir", default='/ecnu105/ztl/our_medBERT_version2_knowledge_diff/MRC_pakage/model_save', type=str,
                        help="save best_model and checkpoint_model dir")
    parser.add_argument("--train_file", default='/home/dell/PycharmProjects/Data/dxy_mrc_data/multi_task_data/modify_question_data/train.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default='/home/dell/PycharmProjects/Data/dxy_mrc_data/multi_task_data/modify_question_data/test.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--dev_file", default='/home/dell/PycharmProjects/Data/dxy_mrc_data/multi_task_data/modify_question_data/dev.json', type=str,
                        help="SQuAD json for develop. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--init_checkpoint",
                        default='/ecnu105/ztl/our_medBERT_version2_knowledge_diff/MRC_pakage/pytorch_pretrained_bert/pytorch_model.bin', type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    # parser.add_argument("--finetuned_checkpoint",
    #                     default='/home/gpu-105/PycharmProjects/Bert_DXY_MRC/ft_dir', type=str,
    #                     help="finetuned checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--best_model",
                        default='/ecnu105/ztl/our_medBERT_version2_knowledge_diff/MRC_pakage/model_save/best_model.pt', type=str,
                        help="finetuned checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_answer_length", default=100, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--dev_batch_size", default=8, type=int, help="Total batch size for dev.")
    parser.add_argument("--predict_batch_size", default=6, type=int, help="Total batch size for predictions.")
    parser.add_argument("--n_best_size", default=10, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--learning_rate", default=7e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_checkpoints_steps", default=10, type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--iterations_per_loop", default=1000, type=int,
                        help="How many steps to make in each estimator call.")
    parser.add_argument("--n-best", default=5, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument('--seed',
                        type=int,
                        default=2020,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--threads',
                        type=int,
                        default=10,
                        help="Number of threads to process data")
    parser.add_argument('--entity_type',
                        default='/home/dell/PycharmProjects/our_medBERT_version2_knowledge_diff/kg_embed/type_set.pkl',
                        type=str, help='entity type in knowledge graph')
    parser.add_argument('--node2entity', default='/home/dell/PycharmProjects/tmp_czr/KIM/OpenKE-master/new_n2e.pkl',
                        type=str, help='target node to other entity relationship')
    parser.add_argument('--two_hop_entity_num', default=7, type=int, help='target node to other entity relationship')
    args = parser.parse_args()

    # 获取实体相关信息
    entity_type_dict, node2entity, combine_entity_type_dict = entity_info(args)
    type_embedd = entity_type_initialize(combine_entity_type_dict)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory () already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)

    num_train_steps = None
    if args.do_train:
        logger.info('Load and process train examples')
        if os.path.exists(os.path.join(args.processed_data, 'processed_train.pkl')):
            # 表示将处理好的数据保存到这个结果里面，不用每次重新进行处理
            with open(os.path.join(args.processed_data, 'processed_train.pkl'), 'rb') as f:
                train_features, train_examples, num_train_steps = pickle.load(f)
        else:
            with open(args.train_file, "r") as reader:
                input_train_data = json.load(reader)["data"]
                input_train_data = input_train_data[:10]
            with Pool(args.threads) as pool:
                read_squad_train_examples_ = partial(read_squad_train_examples, tokenizer=tokenizer, is_training=True)
                train_examples = list(tqdm(pool.imap(read_squad_train_examples_, input_train_data,
                                                     chunksize=args.train_batch_size), total=len(input_train_data),
                                                     desc='read train examples'))
            train_examples = [x for sample in train_examples for x in sample]

            num_train_steps = int(
                len(train_examples) / args.train_batch_size * args.num_train_epochs)

            # with Pool(args.threads) as pool:
            #     read_squad_train_feature_ = partial(convert_examples_to_features,
            #                                          args=args,
            #                                          node2entity=node2entity,
            #                                          entity_dict_init=entity_dict,
            #                                          entity_type=combine_entity_type_dict,
            #                                          type_embedd=type_embedd,
            #                                          tokenizer=tokenizer,
            #                                          max_seq_length=args.max_seq_length,
            #                                          doc_stride=args.doc_stride,
            #                                          max_query_length=args.max_query_length,
            #                                          is_training=True)
            #     train_features = list(tqdm(pool.imap(read_squad_train_feature_, train_examples,
            #                                          chunksize=args.train_batch_size), total=len(train_examples),
            #                                          desc='convert train examples'))
            # train_features = feature_list_expand(train_features, train_examples)
            # raise ValueError('error')
            train_features = convert_examples_to_features(
                train_examples=train_examples,
                args=args,
                node2entity=node2entity,
                entity_dict_init=entity_dict,
                entity_type=combine_entity_type_dict,
                type_embedd=type_embedd,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
            train_path = os.path.join(args.processed_data, 'processed_train.pkl')
            with open(train_path, 'wb') as f:
                pickle.dump([train_features, train_examples, num_train_steps], f)

        #   处理dev类型的数据
        if os.path.exists(os.path.join(args.processed_data, 'processed_dev.pkl')):
            with open(os.path.join(args.processed_data, 'processed_dev.pkl'), 'rb') as f:
                dev_features, dev_examples, dev_three_answer = pickle.load(f)
        else:
            with open(args.dev_file, "r") as reader:
                input_dev_data = json.load(reader)["data"]
                input_dev_data = input_dev_data[:5]
            # dev里面的数据有三个答案，但是我们用第一个答案起始位置作为确定文本哪个范围,所以保存一个答案列表,其中有三个答案，作为调整模型参数的依据
            _, dev_three_answer = devData2trainType(input_dev_data)
            with Pool(args.threads) as pool:
                read_squad_train_examples_ = partial(read_squad_train_examples, tokenizer=tokenizer, is_training=False)
                dev_examples = list(tqdm(pool.imap(read_squad_train_examples_, input_dev_data,
                                                   chunksize=args.dev_batch_size),
                                     total=len(input_dev_data), desc='dev'))
            dev_examples = [x for sample in dev_examples for x in sample]

            dev_features = convert_examples_to_features(
                train_examples=dev_examples,
                args=args,
                node2entity=node2entity,
                entity_dict_init=entity_dict,
                entity_type=combine_entity_type_dict,
                type_embedd=type_embedd,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
            dev_path = os.path.join(args.processed_data, 'processed_dev.pkl')
            with open(dev_path, 'wb') as f:
                pickle.dump([dev_features, dev_examples, dev_three_answer], f)

    if args.do_predict:
        logger.info('Load and process test examples')
        if os.path.exists(os.path.join(args.processed_data, 'processed_test.pkl')):
            with open(os.path.join(args.processed_data, 'processed_test.pkl'), 'rb') as f:
                eval_features, eval_examples, eval_answer = pickle.load(f)
        else:
            with open(args.predict_file, "r") as reader:
                eval_examples = json.load(reader)["data"]
                # eval_examples = eval_examples[:50]
            _, eval_answer = extractEvalAnswer(eval_examples)
            with Pool(args.threads) as pool:
                read_squad_examples_ = partial(read_squad_train_examples, tokenizer=tokenizer, is_training=False)
                eval_examples = list(tqdm(pool.imap(read_squad_examples_, eval_examples,
                                                    chunksize=args.predict_batch_size),
                                     total=len(eval_examples), desc='test'))
            eval_examples = [x for sample in eval_examples for x in sample]

            eval_features = convert_examples_to_features(
                train_examples=eval_examples,
                args=args,
                node2entity=node2entity,
                entity_dict_init=entity_dict,
                entity_type=combine_entity_type_dict,
                type_embedd=type_embedd,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
            eval_path = os.path.join(args.processed_data, 'processed_test.pkl')
            with open(eval_path, 'wb') as f:
                pickle.dump([eval_features, eval_examples, eval_answer], f)

    model = BertForQuestionAnswering(bert_config, embed, type_embedd)
    if args.do_train and args.init_checkpoint is not None:
        logger.info('Loading init checkpoint')
        # model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
        # 加载的参数名称与源代码中的不一样，需要进行一些处理 (1)把加载的模型参数名称前面的bert.这几个字符去掉   （2）把cls开头的参数删除
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')
        new_state_dict = collections.OrderedDict()
        for key, value in state_dict.items():
            if key[0:5] == 'bert.':
                key = key[5:]
                new_state_dict[key] = value
            elif key[0:3] == 'cls':
                continue
            else:
                print('error')
        my_list = ["embeddings.LayerNorm.gamma", "embeddings.LayerNorm.beta", "encoder.layer.0.attention.output.LayerNorm.gamma", "encoder.layer.0.attention.output.LayerNorm.beta", "encoder.layer.0.output.LayerNorm.gamma", "encoder.layer.0.output.LayerNorm.beta", "encoder.layer.1.attention.output.LayerNorm.gamma", "encoder.layer.1.attention.output.LayerNorm.beta", "encoder.layer.1.output.LayerNorm.gamma", "encoder.layer.1.output.LayerNorm.beta", "encoder.layer.2.attention.output.LayerNorm.gamma", "encoder.layer.2.attention.output.LayerNorm.beta", "encoder.layer.2.output.LayerNorm.gamma", "encoder.layer.2.output.LayerNorm.beta", "encoder.layer.3.attention.output.LayerNorm.gamma", "encoder.layer.3.attention.output.LayerNorm.beta", "encoder.layer.3.output.LayerNorm.gamma", "encoder.layer.3.output.LayerNorm.beta", "encoder.layer.4.attention.output.LayerNorm.gamma", "encoder.layer.4.attention.output.LayerNorm.beta", "encoder.layer.4.output.LayerNorm.gamma", "encoder.layer.4.output.LayerNorm.beta", "encoder.layer.5.attention.output.LayerNorm.gamma", "encoder.layer.5.attention.output.LayerNorm.beta", "encoder.layer.5.output.LayerNorm.gamma", "encoder.layer.5.output.LayerNorm.beta", "encoder.layer.6.attention.output.LayerNorm.gamma", "encoder.layer.6.attention.output.LayerNorm.beta", "encoder.layer.6.output.LayerNorm.gamma", "encoder.layer.6.output.LayerNorm.beta", "encoder.layer.7.attention.output.LayerNorm.gamma", "encoder.layer.7.attention.output.LayerNorm.beta", "encoder.layer.7.output.LayerNorm.gamma", "encoder.layer.7.output.LayerNorm.beta", "encoder.layer.8.attention.output.LayerNorm.gamma", "encoder.layer.8.attention.output.LayerNorm.beta", "encoder.layer.8.output.LayerNorm.gamma", "encoder.layer.8.output.LayerNorm.beta", "encoder.layer.9.attention.output.LayerNorm.gamma", "encoder.layer.9.attention.output.LayerNorm.beta", "encoder.layer.9.output.LayerNorm.gamma", "encoder.layer.9.output.LayerNorm.beta", "encoder.layer.10.attention.output.LayerNorm.gamma", "encoder.layer.10.attention.output.LayerNorm.beta", "encoder.layer.10.output.LayerNorm.gamma", "encoder.layer.10.output.LayerNorm.beta", "encoder.layer.11.attention.output.LayerNorm.gamma", "encoder.layer.11.attention.output.LayerNorm.beta", "encoder.layer.11.output.LayerNorm.gamma", "encoder.layer.11.output.LayerNorm.beta"]

        bert_state_dict = collections.OrderedDict()

        for key, value in new_state_dict.items():
            if key in my_list:
                if 'beta' in key:
                    key = key.replace('beta', 'bias')
                elif 'gamma' in key:
                    key = key.replace('gamma', 'weight')
                bert_state_dict[key] = value
            else:
                bert_state_dict[key] = value
        model.bert.load_state_dict(bert_state_dict)

        logger.info('Loaded init checkpoint')
    elif args.do_predict:
        logger.info('Loading best model')
        model = torch.load(args.best_model)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        logger.info('Loaded best model')
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
        ]

    optimizer = BertAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

        all_entity_ids = torch.tensor([f.entity_ids for f in train_features], dtype=torch.long)
        all_entity_ids_type = torch.tensor([f.all_entity_ids_type for f in train_features], dtype=torch.long)
        all_two_hop_entity_type_mask = torch.tensor([f.two_hop_entity_type_mask for f in train_features], dtype=torch.long)
        all_two_hop_entity_type_index_select = torch.tensor([f.two_hop_entity_type_index_select for f in train_features], dtype=torch.long)
        all_two_hop_entity_ids = torch.tensor([f.entity_ids_mapping for f in train_features], dtype=torch.long)
        all_two_hop_entity_type = torch.tensor([f.two_hop_entity_type for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions, all_entity_ids, all_entity_ids_type,
                                   all_two_hop_entity_type_mask, all_two_hop_entity_type_index_select,
                                   all_two_hop_entity_ids, all_two_hop_entity_type)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        best_val_score = 0
        epoch_list = []
        for epoch in range(int(args.num_train_epochs)):
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                all_input_ids, all_input_mask, all_segment_ids, \
                all_start_positions, all_end_positions, all_entity_ids, all_entity_ids_type, \
                all_two_hop_entity_type_mask, all_two_hop_entity_type_index_select, \
                all_two_hop_entity_ids, all_two_hop_entity_type = batch

                loss = model(all_input_ids, all_segment_ids, all_input_mask, all_entity_ids, all_entity_ids_type,
                             all_two_hop_entity_ids, all_two_hop_entity_type_mask, all_two_hop_entity_type_index_select,
                             all_two_hop_entity_type, all_start_positions, all_end_positions)
                if n_gpu > 1:
                    loss = loss.mean()# mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                logger.info("Step: {} / loss value: {}".format(step, loss.item()))
                epoch_list.append(loss.item())
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1
            # save_checkpoints_steps = 20
            # if epoch % args.save_checkpoints_steps == 0:

            predict_result = run_evaluate(args, model, dev_features, device, dev_examples)
            predict_result_list = []
            for key, value in predict_result.items():
                predict_result_list.append(value)
            em, f1 = score(predict_result_list, dev_three_answer)
            logger.info("dev EM: {} F1: {}".format(em, f1))
            # save
            # if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            model_file = os.path.join(args.model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            torch.save(model, model_file)
            # model.module.save(model_file, epoch, [em, f1, best_val_score])
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                    model_file,
                    os.path.join(args.model_dir, 'best_model.pt'))
                logger.info('[new best model saved.]')
            logger.info('Best dev score {} in train_epochs {}:'.format(best_val_score, epoch))
       # plot_x = np.linspace(1, len(epoch_list), len(epoch_list))
       # plt.plot(plot_x, epoch_list)
       # plt.show()
    if args.do_predict:
        predict_result = run_evaluate(args, model, eval_features, device, eval_examples)
        predict_result_list = []
        for key, value in predict_result.items():
            predict_result_list.append(value)
        em, f1 = score(predict_result_list, eval_answer)
        logger.info("test EM: {} F1: {}".format(em, f1))

def devData2trainType(input_dev_data):
    dev_three_answer = []
    for i, dev_example_data in enumerate(input_dev_data):
        qas = dev_example_data['paragraphs'][0]['qas']
        for j, qas_temp in enumerate(qas):
            answer_temp = input_dev_data[i]['paragraphs'][0]['qas'][j]['answers']
            input_dev_data[i]['paragraphs'][0]['qas'][j]['answers'] = []
            input_dev_data[i]['paragraphs'][0]['qas'][j]['answers'].append(answer_temp[0])
            dev_three_answer.append([answer_temp[0]['text'], answer_temp[1]['text'], answer_temp[2]['text']])
    return input_dev_data, dev_three_answer
def extractEvalAnswer(eval_examples):
    eval_answer = []
    for i, dev_example_data in enumerate(eval_examples):
        qas = dev_example_data['paragraphs'][0]['qas']
        for j, qas_temp in enumerate(qas):
            answer_temp = eval_examples[i]['paragraphs'][0]['qas'][j]['answers']
            eval_examples[i]['paragraphs'][0]['qas'][j]['answers'] = []
            eval_examples[i]['paragraphs'][0]['qas'][j]['answers'].append(answer_temp[0])
            eval_answer.append([answer_temp[0]['text'], answer_temp[1]['text'], answer_temp[2]['text']])
    return eval_examples, eval_answer
if __name__ == "__main__":
    main()
