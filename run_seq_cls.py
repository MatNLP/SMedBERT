from __future__ import absolute_import, division, print_function

import json
# import pretraining_args as args
import csv
import logging
import os
# import random
import sys
from glob import glob
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from random import random, randrange, randint, shuffle, choice, sample, uniform, randint
from random import Random as Rd
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score,precision_score,recall_score
import jieba
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME, MAX_TARGET, MAX_NUM_PAIR, MAX_LONG_WORD_USE, \
    MAX_SHORT_WORD_USE, MAX_SEG_USE

from pytorch_pretrained_bert.modeling import BertForHyperPreTraining, cMeForRE, \
    BertConfig  # Note that we use the sop, rather than nsp task.
# from code.knowledge_bert.modeling import BertForPreTraining

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.Trie import Trie
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_double_linear_schedule_with_warmup
from pytorch_pretrained_bert.optimization import BertAdam
import argparse
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Process
import gc
import pickle
import torch.distributed as dist
import time
from torch.nn.parallel import DistributedDataParallel as DDP

# torch.autograd.set_detect_anomaly(True)

# ww_lables = None
# with open('ww_sim/labels.pkl','rb') as wf:
#     ww_lables = pickle.load(wf)


entity_file = open('kgs/ch_entity2id.txt', 'r',
                   encoding='utf-8')  # Note that we must add the first entity as EMPETY.
entity_dict = {}
entity_file.readline()
id2entity = {}

for line in entity_file:
    name, idx = line.rstrip().split('\t')
    entity_dict[name] = int(idx) + 1
    id2entity[idx] = name
entity_file.close()

entity_file = open('kgs/entityId2weight.json', 'r',
                   encoding='utf-8')
idx2weight = json.load(entity_file)
entity_file.close()

rel_file = open('kgs/ch_relation2id.txt', 'r',
                   encoding='utf-8')
rel_file.readline()
rel_dict = {}
for line in rel_file:
    name, idx = line.rstrip().split('\t')
    rel_dict[name] = int(idx) + 1

ww_tree = Trie()
MAX_PRE_ENTITY_MASK = 100
mask_count = dict.fromkeys(entity_dict.keys(), MAX_PRE_ENTITY_MASK)

for key in entity_dict.keys():
    if (len(key) > 1 and not key.isdigit()):
        ww_tree.insert(key)
    # entity_dict[key] += 1  # For the zero embedding is used as Pad ebmedding.

entity_dict_index2str = {value: key for key, value in entity_dict.items()}

# keys = ['ent_embeddings', 'rel_embeddings']
js_file = open('kgs/transr_transr.json', 'r',
               encoding='utf-8')  # Note that we must add the first entity as EMPTY.
js_dict = json.load(js_file)
js_file.close()
embedding_list = js_dict['ent_embeddings.weight']
transfer_matrix_list = js_dict['transfer_matrix.weight']
relation_list = js_dict['rel_embeddings.weight']
e_dim = len(embedding_list[0])
assert(len(transfer_matrix_list[0]) % e_dim == 0)
r_dim = len(transfer_matrix_list[0]) // e_dim
assert(len(transfer_matrix_list) == len(relation_list))
for i in range(len(transfer_matrix_list)):
    transfer_matrix_list[i].extend(relation_list[i])

# transfer_matrix_list = [[0]*(r_dim*e_dim)] + transfer_matrix_list
transfer_matrix = torch.FloatTensor(transfer_matrix_list)
transfer_matrix = transfer_matrix.view(transfer_matrix.size(0),r_dim,e_dim+1)
transfer_matrix = torch.bmm(transfer_matrix.transpose(-1,-2),transfer_matrix)
transfer_matrix = torch.cat((torch.zeros(1,e_dim+1,e_dim+1),transfer_matrix),dim=0)
transfer_matrix = torch.nn.Embedding.from_pretrained(transfer_matrix.view(-1,(e_dim+1)*(e_dim+1)),freeze=False)

# ['zero_const', 'pi_const', 'ent_embeddings.weight', 'rel_embeddings.weight', 'transfer_matrix.weight']

def euclidean(p, q):
    # 计算欧几里德距离,并将其标准化
    e = sum([(p[i] - q[i]) ** 2 for i in range(len(p))])
    # return 1 / (1 + e ** .5)
    return e


vecs = []
vecs.append([0] * 100)  # CLS
for vec in embedding_list:
    vecs.append(vec)
embedding_list = vecs
embed = torch.FloatTensor(vecs)
embed = torch.nn.Embedding.from_pretrained(embed,freeze=False)

# print(len(embedding_list))
# print(len(entity_dict))
# print(max(entity_dict.values()))
# del vecs, embedding_list, js_file, entity_file
# predefined_entity_type = ['药品', '疾病', '症状', '其他', '缺省']
# type_embed = torch.randn(len(predefined_entity_type), 100).float()
# type_embed = torch.nn.Embedding.from_pretrained(type_embed)
del  js_file, entity_file
MAX_SEQ_LEN = 512
WORD_CUTTING_MAX_PAIR = 50
GUEESS_ATOM_MAX_PAIR = 50
POS_NEG_MAX_PAIR = 10

# MAX_TARGET=32
# MAX_NUM_PAIR=25

SAVE_THELD = .1
logger = logging.getLogger(__name__)
rng = Rd(43)
import re

# def collect_fn(x):
#     print(type(x))
#     x = torch.cat( tuple(xx.unsqueeze(0) for xx in x) , dim=0 )
#     entity_idx = x[:, 4*args.max_seq_length:5*args.max_seq_length]
#     # Build candidate
#     uniq_idx = np.unique(entity_idx.numpy())
#     ent_candidate = embed(torch.LongTensor(uniq_idx+1))
#     ent_candidate = ent_candidate.repeat([n_gpu, 1])
#     # build entity labels
#     d = {}
#     dd = []
#     for i, idx in enumerate(uniq_idx):
#         d[idx] = i
#         dd.append(idx)
#     ent_size = len(uniq_idx)-1
#     def map(x):
#         if x == -1:
#             return -1
#         else:
#             rnd = random.uniform(0, 1)
#             if rnd < 0.05:
#                 return dd[random.randint(1, ent_size)]
#             elif rnd < 0.2:
#                 return -1
#             else:
#                 return x
#     ent_labels = entity_idx.clone()
#     d[-1] = -1
#     ent_labels = ent_labels.apply_(lambda x: d[x])
#     entity_idx.apply_(map)
#     ent_emb = embed(entity_idx+1)
#     mask = entity_idx.clone()
#     mask.apply_(lambda x: 0 if x == -1 else 1)
#     mask[:,0] = 1
#     return x[:,:args.max_seq_length], x[:,args.max_seq_length:2*args.max_seq_length], x[:,2*args.max_seq_length:3*args.max_seq_length], x[:,3*args.max_seq_length:4*args.max_seq_length], ent_emb, mask, x[:,6*args.max_seq_length:], ent_candidate, ent_labels

def key_fn(obj):
    idx = entity_dict[obj[0]]-1
    weight = idx2weight[str(idx)]
    return weight

class OurENRIEDataset(Dataset):
    def __init__(self, args, data_path, max_seq_length, masked_lm_prob,
                 max_predictions_per_seq, tokenizer, node2entity, entity_dict_init, entity_type, type_embedd,type2id,entityOutNegbhor,entityInNegbhor, min_len=128):
        self.args = args
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.tokenizer = tokenizer
        self.node2entity = node2entity
        self.entity_dict = entity_dict_init
        self.min_len = min_len
        self.type2id = type2id
        self.max_num_tokens = max_seq_length - 3
        self.lines = []
        self.vocab = list(tokenizer.vocab.keys())
        self.entity_type = entity_type
        self.type_embedd = type_embedd
        self.entity_dict_reverse = {value: key for key, value in entity_dict_init.items()}
        self.entityInNegbhor = entityInNegbhor
        self.entityOutNegbhor = entityOutNegbhor
        self.__read_data__()

    def __getitem__(self, index):
        line = self.lines[index]
        htl = self.__data2tokens__(line)
        feature = self.__tokens2feature__(*htl)
        # example_a = {
        #     "tokens": example['tokens_a'],
        #     "segment_ids": example['segment_ids_a'],
        #     'entity_pos': example['entity_pos_a'] }
        f = self.__get_feature__(feature)
        
        # example_b = {
        #     "tokens": example['tokens_b'],
        #     "segment_ids": example['segment_ids_b'],
        #     'entity_pos': example['entity_pos_b'] }
        # f_b = self.__get_feature__(example_b)
        tensor_tuple = self.__feature2tensor__(f)
        # tensor_tuple_b = self.__feature2tensor__(f_b)

        return tensor_tuple

    def __data2tokens__(self,data):
        label,head,tail = data.strip().split('\t')
        label = int(label)
        assert(label==0 or label==1)
        head = self.tokenizer.tokenize(head)
        tail = self.tokenizer.tokenize(tail)
        # head = self.tokenizer.tokenize(head)
        # tail = None
        truncate_seq_pair(head,tail,MAX_SEQ_LEN-3,rng)
        return (head,tail,label)
    
    def __tokens2feature__(self,head,tail,label):
        # tokens = self.tokenizer.convert_ids_to_tokens(head)
        # if('[PAD]' in tokens):
        #     tokens = tokens[:tokens.index('[PAD]')]
        # try:
        #     assert( tokens[0] == '[CLS]')
        #     assert ( tokens[-1] == '[SEP]')
        # except:
        #     print(tokens)
        #     assert(False)
        tokens = ['[CLS]'] + head + ['[SEP]']
        segment_ids = [0] * len(tokens)
        if(tail): 
            tail_tokens = tail + ['[SEP]']
            segment_ids += [1]*len(tail_tokens)
            tokens += tail_tokens
        tokens, entity_pos = convert_sentence_to_tokens(tokens,self.tokenizer)
        example ={
            'tokens': tokens,
            'segment_ids': segment_ids,
            'entity_pos': entity_pos,
            'label': label
        }
        return example

    def __get_example__(self, line):
        line = line.rstrip()
        a,b = line.split('\t')
        tokens_a = self.tokenizer.tokenize(a)
        tokens_b = self.tokenizer.tokenize(b)

        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = ['[CLS]'] + tokens_b + ['[SEP]']
        
        tokens_a, entity_pos_a = convert_sentence_to_tokens(
            tokens_a,self.tokenizer)
        tokens_b, entity_pos_b = convert_sentence_to_tokens(
            tokens_b,self.tokenizer)
        
        segment_ids_a = [0] * len(tokens_a)
        segment_ids_b = [0] * len(tokens_b)

        # one_hop_entity_ids = np.array(len(tokens),dtype=np.int)
        # one_hop_entity_types = np.zeros(len(tokens),args.two_hop_entity_num)
        # two_hop_entity_ids = np.zeros(len(tokens),args.two_hop_entity_num)
        # two_hop_entity_types = np.zeros(len(tokens),args.two_hop_entity_num)

        example = {
            "tokens_a": tokens_a,
            "segment_ids_a": segment_ids_a,
            'entity_pos_a': entity_pos_a,
            "tokens_b": tokens_b,
            "segment_ids_b": segment_ids_b,
            'entity_pos_b': entity_pos_b
        }
        return example

    def __get_feature__(self, example):
        args = self.args
        max_seq_length = self.max_seq_length

        tokens = example["tokens"]
        segment_ids = example["segment_ids"]
        entity_pos = example['entity_pos']
        label = example['label']
        args = self.args
        
        kc_entity_se_index_array = np.zeros((MAX_TARGET,2),dtype=np.int)
        kc_entity_two_hop_labels_array = np.full((MAX_TARGET,args.two_hop_entity_num),fill_value=-1,dtype=np.int)
        
        kc_entity_out_or_in_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        kc_entity_two_hop_rel_types_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        kc_entity_infusion_pos_array = np.zeros(max_seq_length,dtype=np.int)
        kc_entity_two_hop_types_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        
        kc_entity_one_hop_ids_array = np.zeros(MAX_TARGET,dtype=np.int) # Note that Label has -1 as padding while ids use 0.
        kc_entity_one_hop_types_array = np.zeros(MAX_TARGET,dtype=np.int)


        for index,key in  enumerate(entity_pos):
            word = entity_pos[key]
            start,end = key,key+len(word)
            
            
            tmp_set = [] 
            if(word in self.entityInNegbhor):
                for rel,e in self.entityInNegbhor[word]:
                    tmp_set.append((e,rel,-1))
            if(word in self.entityOutNegbhor):
                for rel,e in self.entityOutNegbhor[word]:
                    tmp_set.append((e,rel,1))
            
            tmp_set = sorted(tmp_set,key=key_fn,reverse=True)
            tmp_set = tmp_set[:args.two_hop_entity_num]

            kc_entity_se_index_array[index] = [start,end-1]
            tmp = list(t[2] for t in tmp_set)
            kc_entity_out_or_in_array[index][:len(tmp)] = tmp
            tmp = list(self.entity_dict[t[0]] for t in tmp_set)
            kc_entity_two_hop_labels_array[index][:len(tmp)] = tmp
            tmp = list(rel_dict[t[1]] for t in tmp_set)
            kc_entity_two_hop_rel_types_array[index][:len(tmp)] = tmp
            tmp = list(self.type2id[self.entity_type[t[0]]] for t in tmp_set)
            kc_entity_two_hop_types_array[index][:len(tmp)] = tmp

            kc_entity_one_hop_ids_array[index] = self.entity_dict[word]
            kc_entity_one_hop_types_array[index] = self.type2id[self.entity_type[word]]
            kc_entity_infusion_pos_array[start:end] = index + 1


        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        
        input_array = np.full(max_seq_length, dtype=np.int, fill_value=self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0] )
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.int)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.int)
        segment_array[:len(segment_ids)] = segment_ids

        feature = InputFeatures(input_ids=input_array,
                                input_mask=mask_array,
                                segment_ids=segment_array,
                                kc_entity_one_hop_ids=kc_entity_one_hop_ids_array,
                                kc_entity_one_hop_types=kc_entity_one_hop_types_array,
                                kc_entity_se_index=kc_entity_se_index_array,
                                kc_entity_two_hop_labels=kc_entity_two_hop_labels_array,
                                kc_entity_out_or_in=kc_entity_out_or_in_array,
                                kc_entity_two_hop_rel_types=kc_entity_two_hop_rel_types_array,
                                kc_entity_two_hop_types_array=kc_entity_two_hop_types_array,
                                kc_entity_infusion_pos = kc_entity_infusion_pos_array,
                                label = [label])
        return feature

    def __feature2tensor__(self, feature):
        f = feature
        all_input_ids = torch.tensor(f.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(f.input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(f.segment_ids, dtype=torch.long)
        # all_label_ids = torch.tensor(f.label_id, dtype=torch.long)
        # all_sop_labels = torch.tensor(f.sop_label, dtype=torch.long).unsqueeze(-1)

        # 一阶实体相关的变量：一阶实体每个token位置就一个，不需要mask，直接最后用input_mask即可
        # all_one_hop_entity_ids = torch.tensor(f.kc_entity_one_hop_ids, dtype=torch.long)
        # all_one_hop_entity_types = torch.tensor(f.kc_entity_one_hop_types, dtype=torch.long)
        # all_one_hop_entity_types_repre = self.type_embedd(all_one_hop_entity_types)

        # 二阶实体相关变量
        # 8*512*7
        # all_two_hop_entity_ids = torch.tensor(f.two_hop_entity_ids, dtype=torch.long)
        # all_two_hop_entity_repre = embed(all_two_hop_entity_ids + 1)

        # 8*512*7 (取-1进行mask)
        # all_two_hop_entity_mask = torch.tensor(f.entity_mapping_mask, dtype=torch.long)

        # 8*512*7*5
        # all_two_hop_entity_types = torch.tensor(f.two_hop_entity_types, dtype=torch.long)

        # 8*512*5
        # all_two_hop_entity_type_mask = torch.tensor(f.two_hop_entity_type_mask, dtype=torch.long)

        # 8*512*7
        # all_two_hop_entity_type_index_select = torch.tensor(f.two_hop_entity_type_index_select, dtype=torch.long)
        # print(f.kc_entity_one_hop_ids.shape)
        kc_entity_one_hop_ids = torch.tensor(f.kc_entity_one_hop_ids, dtype=torch.long)
        kc_entity_one_hop_types = torch.tensor(f.kc_entity_one_hop_types, dtype=torch.long)
        # print(kc_entity_one_hop_ids.size())
        # print(kc_entity_one_hop_types.size())
        kc_entity_se_index = torch.tensor(f.kc_entity_se_index, dtype=torch.long)
        kc_entity_two_hop_labels = torch.tensor(f.kc_entity_two_hop_labels, dtype=torch.long)
        kc_entity_out_or_in = torch.tensor(f.kc_entity_out_or_in, dtype=torch.long)
        kc_entity_two_hop_rel_types = torch.tensor(f.kc_entity_two_hop_rel_types, dtype=torch.long)
        kc_entity_two_hop_types = torch.tensor(f.kc_entity_two_hop_types)
        kc_entity_infusion_pos = torch.tensor(f.kc_entity_infusion_pos)
        label = torch.tensor(f.label,dtype=torch.long)
        # print(all_input_ids)
        return ((all_input_ids, all_segment_ids,all_input_mask, kc_entity_one_hop_ids,
                 kc_entity_one_hop_types) #+ (all_two_hop_entity_ids, all_two_hop_entity_types)
                 +(kc_entity_se_index, kc_entity_two_hop_labels, kc_entity_out_or_in, kc_entity_two_hop_rel_types,kc_entity_two_hop_types,kc_entity_infusion_pos)
                 +(label,)
                 )
        # one example has 11 features.

    def __len__(self):
        return len(self.lines)

    def __read_data__(self):
        fr = open(self.data_path, "r",encoding='utf-8')
        self.lines = fr.readlines()
        fr.close()



### End

### Load the synset dict and len2word dict
# MAX_SYN_USEAGE=30 # Note that since our synset is samll, we actually replace syn word as far as we can, and limit the useage.

# 现有替换策略为预处理时先生成并标记增强样本，对于增强样本中的同义词，不进行mask预测，但同样可进行切分和分类以及特征词原子词相互预测。
# syn_file=open('data_aug/syn.pkl','rb')
# synset=pickle.load(syn_file)
# syn_file.close()


# len2word={} # use to produce negative sample, note that we can always use the corrpording word as postive sample.

### End

### creat the converter for type and entitiy
# l2w_file=open('cut_word/l2w.pkl','rb')
# len2word=pickle.load(l2w_file) # length of word and type back to word
# l2w_file.close()
### End

# logging.basicConfig(filename='logger.log', level=logging.INFO)
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(cp)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids,  kc_entity_one_hop_ids, 
                kc_entity_one_hop_types,
                kc_entity_se_index,kc_entity_two_hop_labels,kc_entity_out_or_in,kc_entity_two_hop_rel_types,kc_entity_two_hop_types_array,kc_entity_infusion_pos,label=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.kc_entity_one_hop_ids=kc_entity_one_hop_ids
        self.kc_entity_one_hop_types=kc_entity_one_hop_types

        self.kc_entity_se_index = kc_entity_se_index
        self.kc_entity_two_hop_labels = kc_entity_two_hop_labels
        self.kc_entity_out_or_in = kc_entity_out_or_in
        self.kc_entity_two_hop_rel_types = kc_entity_two_hop_rel_types
        self.kc_entity_two_hop_types = kc_entity_two_hop_types_array
        self.kc_entity_infusion_pos = kc_entity_infusion_pos
        self.label = label

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        # if rng.random() < 0.01:  # I do not want you delete front because you cause the head always produce [UNK]
        #     del trunc_tokens[0]
        # else:
        trunc_tokens.pop()


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]" or not token.isalnum():
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    # print(num_to_mask)
    # print("tokens", len(tokens))
    # print("cand", len(cand_indices))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels


def isSkipToken(token):
    return token == "[CLS]" or token == "[SEP]" or (not token.isalnum() and len(token) == 1)


def create_wwm_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab, tokenizer):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""

    chars = [token for token in tokens]
    cand_map = dict(zip(range(len(tokens)), range(len(tokens))))

    entity_pos = {}
    # entity_ids = []  # This list would like [ [0] , [1,2] , [3,4,5]] to pack wwm word toghter
    # We use it to 保存分词预测和特征词原子词相互预测的位置
    cand_index = 0
    skip_index = set()

    assert (chars[0] == '[CLS]')

    while (cand_index < len(chars)):
        if (isSkipToken(chars[cand_index])):
            skip_index.add(cand_index)
            # entity_ids.append(-1)
        elif ( ww_tree.startsWith(chars[cand_index])):
            c = ww_tree.get_lengest_match(chars, cand_index)
            if (c == None):
                # entity_ids.append(-1)
                pass
            else:
                word = ''.join(chars[cand_index:c + 1])
                assert (word in entity_dict)
                # mask_count[word] -= 1
                entity_pos[cand_index]=word
                cand_index += len(word)
                continue
        cand_index += 1
    words_hit= list(entity_pos.items())
    if(len(words_hit)>MAX_TARGET):
        shuffle(words_hit)
    entity_pos = dict(words_hit[:MAX_TARGET])
    words_hit = [ w[1] for w in words_hit[:MAX_TARGET] ] 
    for w in words_hit:
        if(w in mask_count):
            mask_count[w]-=1
            if(mask_count[w]==0):
                mask_count.pop(w)
                ww_tree.delete(w)
    cand_indices = [i for i in range(cand_index) if i not in skip_index ]
    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(cand_indices) * masked_lm_prob))))
    
    
    shuffle(cand_indices)
    mask_indices = sorted(cand_indices[:num_to_mask])
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        chars[index] = masked_token

    assert (len(mask_indices) <= MAX_SEQ_LEN)
    assert (len(masked_token_labels) == len(mask_indices))

    assert (len(chars) <= MAX_SEQ_LEN)
    assert (len(set(mask_indices)) == len(mask_indices))
    # print(entity_pos)
    assert (len(entity_pos)<=MAX_TARGET)
    return chars, mask_indices, masked_token_labels, entity_pos

def convert_sentence_to_tokens(tokens,  tokenizer):
    
    chars = [token for token in tokens]
    entity_pos = {}
    cand_index = 0
    assert (chars[0] == '[CLS]')

    while (cand_index < len(chars)):
        if( (not isSkipToken(chars[cand_index])) and  ww_tree.startsWith(chars[cand_index])):
            c = ww_tree.get_lengest_match(chars, cand_index)
            if(c is not None):
                word = ''.join(chars[cand_index:c + 1])
                assert (word in entity_dict)
                entity_pos[cand_index]=word
                cand_index += len(word)
                continue
        cand_index += 1
    
    words_hit= list(entity_pos.items())
    if(len(words_hit)>MAX_TARGET):
        shuffle(words_hit)
    entity_pos = dict(words_hit[:MAX_TARGET])
    words_hit = [ w[1] for w in words_hit[:MAX_TARGET] ]

    assert (len(chars) <= MAX_SEQ_LEN)
    assert (len(entity_pos)<=MAX_TARGET)
    return chars, entity_pos

def convert_examples_to_features(args, lines, max_seq_length, tokenizer, do_gc=False):
    features = []
    example_num = len(lines)
    names_list = []
    save_pre_step = max(int(.25 * example_num), 1)
    # print(save_pre_step)
    # example = {
    #         "tokens": tokens,
    #         "segment_ids": segment_ids,
    #         "masked_lm_positions": masked_lm_positions,
    #         "masked_lm_labels": masked_lm_labels,
    #         "entiy_ids": entiy_ids,
    #         'sop_label':sop_label
    #         }
    for f_index in tqdm(range(example_num), desc="Converting Feature"):
        #    for i, example in enumerate(lines):
        # print(f_index)
        example = lines[-1]
        tokens = example["tokens"]
        segment_ids = example["segment_ids"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_labels = example["masked_lm_labels"]
        entity_ids_mapping = example["entity_ids_mapping"]
        entity_ids_mapping_mask = example["entity_ids_mapping_mask"]

        add_default_value = args.max_seq_length - len(entity_ids_mapping)
        for _ in range(add_default_value):
            number_hop_list = [-1 for _ in range(args.two_hop_entity_num)]
            entity_ids_mapping.append(number_hop_list)
            number_default_list = [0 for _ in range(args.two_hop_entity_num)]
            entity_ids_mapping_mask.append(number_default_list)
        assert len(entity_ids_mapping) == args.max_seq_length
        assert len(entity_ids_mapping_mask) == args.max_seq_length

        entity_ids_mapping = np.array(entity_ids_mapping)
        entity_ids_mapping_mask = np.array(entity_ids_mapping_mask)

        entiy_ids = example["entiy_ids"]
        sop_label = example['sop_label']
        # print(list(zip(tokens,range(len(tokens)))))

        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
        assert (len(masked_label_ids) == len(masked_lm_positions))
        input_array = np.zeros(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.bool)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.bool)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
        lm_label_array[masked_lm_positions] = masked_label_ids

        entity_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
        entity_array[:len(entiy_ids)] = entiy_ids
        # seg_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
        # seg_label_array[seg_positions] = seg_labels

        feature = InputFeatures(input_ids=input_array,
                                input_mask=mask_array,
                                segment_ids=segment_array,
                                label_id=lm_label_array,
                                entiy_ids=entity_array,
                                entity_mapping=entity_ids_mapping,
                                entity_mapping_mask=entity_ids_mapping_mask,
                                sop_label=sop_label)
        features.append(feature)
        lines.pop()
        del example
        if (((f_index + 1) % save_pre_step) == 0 or (f_index + 1) == example_num):
            print("Do Save There")
            name = 'run_tmp/{}_f.pklf'.format(f_index)
            sf = open(name, 'wb+')
            pickle.dump(features, sf)
            sf.close()
            names_list.append(name)
            features.clear()
            del name
    del features
    features = []
    lines = []
    for name in tqdm(names_list, desc='Loading features'):
        sf = open(name, 'rb')
        f = pickle.load(sf)
        sf.close()
        features.extend(f)
        del f
    return features


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + 2)
    rng.seed(seed + 4)


def evaluate(args, model, eval_dataloader, device, loss_bag, eval_step):
    torch.cuda.empty_cache()
    best_loss, epoch, tr_loss = loss_bag
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    labels = []
    for batch in eval_dataloader:
        # logger.info('Testing step: {} / {}'.format(nb_eval_steps, eval_step))
        batch0 = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            output = model(*batch0)
            loss = output[0]
            logits = output[1]

            pred = torch.argmax(logits,dim=-1)
            preds.extend(pred.cpu().tolist())
            labels.extend(batch[-1].view(-1).cpu().tolist())
        eval_loss += loss.mean().item()
        nb_eval_steps += 1
    # print(preds)
    # print(labels)
    assert(len(labels) == len(preds))
    P = precision_score(labels,preds,[i for i in range(args.label_num)],average='micro')
    R = recall_score(labels,preds,[i for i in range(args.label_num)],average='micro')
    F1 = f1_score(labels,preds,[i for i in range(args.label_num)],average='micro')
    
    logger.info('P:{} R:{} F1:{}'.format(P,R,F1))

    eval_loss = eval_loss / nb_eval_steps
    if eval_loss < best_loss:
        # Save a trained model, configuration and tokenizer
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        # output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        # torch.save(model_to_save.state_dict(), output_model_file)
        best_loss = eval_loss
    if(args.local_rank <=0 ):
        logger.info(
        "============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n" % (epoch, tr_loss, eval_loss))
    torch.cuda.empty_cache()
    return min(best_loss, eval_loss)

def entity_info(args):
    # #实体类型：{'西药', '疾病', '疾', '分类', '中药', '检查', '中西症状', '药品', '科室', '中医症状', '部位', '西医症状', '症状', '检验'}
    # DXY 实体类型: {'药品', '日常用品', '地区', '疾病', '检验', '特殊人群', '大众医学词汇', '身体部位', '微生物病原体', '检查检验结果', '医院', '症状', '非手术治疗', '季节', '行为活动', '食品', '检查', '科室', '基因', '科室传染性湿疹样', '机构', '外界环境因素', '症状"', '手术治疗', '疾病"', '医疗器械'}
    with open(args.entity_type, 'rb') as fo:
        entity_type_dict = pickle.load(fo, encoding='utf-8')
    with open(args.entityOutNegbhor,'rb') as fo:
        entityOutNegbhor = pickle.load(fo)
    with open(args.entityInNegbhor,'rb') as fo:
        entityInNegbhor = pickle.load(fo)
    
    entities = set(entityInNegbhor.keys())
    entities = entities.union(entityOutNegbhor.keys())

    node2entity = {}
    for key in entities:
        tmp_set =set()
        if(key in entityInNegbhor):
            for rel,e in entityInNegbhor[key]:
                tmp_set.add(e)
        if(key in entityOutNegbhor):
            for rel,e in entityOutNegbhor[key]:
                tmp_set.add(e)
        node2entity[key] = list(tmp_set)


    # new_entity_type_dict = {}
    # total_entity = []
    # for one_hop_entity, two_hop_entity_list in node2entity.items():
    #     total_entity.append(one_hop_entity)
    #     total_entity += two_hop_entity_list
    # for entity in set(total_entity):
    #     if entity not in new_entity_type_dict.keys():
    #         if len(entity_type_dict[entity]) == 0:
    #             # 如果没有entity type的实体默认为 ”症状“
    #             new_entity_type_dict[entity] = '症状'
    #         else:
    #             # 如果有多个实体类型，直接取第一个
    #             new_entity_type_dict[entity] = entity_type_dict[entity][0]

    # 将实体类型进行合并，保持实体类型的个数尽可能集中，同时显存占用少。
    # 经过实体数据统计，可以分为： 药品（包含中药，西药，药品），症状（中医症状，西医症状，症状），疾病，其他
    # combine_entity_type_dict = {}
    # for key, value in new_entity_type_dict.items():
    #     if value == '药品' or value == '中药' or value == '西药':
    #         combine_entity_type_dict[key] = '药品'
    #     elif value == '中医症状' or value == '西医症状' or value == '症状':
    #         combine_entity_type_dict[key] = '症状'
    #     elif value == '疾病':
    #         combine_entity_type_dict[key] = '疾病'
    #     else:
    #         combine_entity_type_dict[key] = '其他'

    # entity_type_dict: 每一个entity对应一个entity type
    # node2entity: 每一个entity对应一个二阶entity list
    # combine_entity_type_dict： 经过entity type合并的字典，将所有的entity type合并成了四种类型
    return  node2entity, entity_type_dict,entityOutNegbhor,entityInNegbhor

def entity_type_initialize(entity2type):
    # predefined_entity_type = ['药品', '疾病', '症状', '其他', '缺省']
    
    type_set = set(entity2type.values())
    type2embed = {}
    type2count = {}
    dim = len(embedding_list[0])

    for key in type_set:
        type2embed[key] = np.zeros(dim)
        type2count[key] = 0
    for e in entity2type:
        e_type = entity2type[e]
        type2embed[e_type] += embedding_list[entity_dict[e]]
        type2count[e_type] += 1
    type2id = {}
    weights = [np.zeros(dim)] # Note: 0 is the index for padding entity

    for index,key in enumerate(type2embed):
        weights.append( type2embed[e_type]/type2count[e_type] )
        type2id[key]=index+1
    return_result = torch.Tensor(weights)
    return_result = torch.nn.Embedding.from_pretrained(return_result)
    return type2id,return_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_train_path", type=str,
                        default="./datasets/cMedQQ/train.txt",
                        help="pretrain train path to file")
    parser.add_argument("--pretrain_dev_path", type=str,
                        default="./datasets/cMedQQ/test.txt",
                        help="pretrain dev path to file")
    parser.add_argument("--max_seq_length", type=int, default=512, help="max seq length of input sequences")
    parser.add_argument("--do_train", type=bool, default=True, help="If do train")
    parser.add_argument("--do_lower_case", type=bool, default=True, help="If do case lower")
    parser.add_argument("--train_batch_size", type=int, default=16, help="train_batch_size")  # May Need to finetune
    parser.add_argument("--eval_batch_size", type=int, default=32, help="eval_batch_size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="num_train_epochs")
    parser.add_argument("--learning_rate", type=float, default=1.25e-5, help="learning rate")  # May Need to finetune
    parser.add_argument("--warmup_proportion", type=float, default=.1,
                        help="warmup_proportion")  # May Need to finetune
    parser.add_argument("--no_cuda", type=bool, default=False, help="prevent use GPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="If we are using cluster for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="gradient_accumulation_steps")  # May Need to finetune
    parser.add_argument("--fp16", type=bool, default=False, help="If use apex to train")
    parser.add_argument("--loss_scale", type=int, default=0, help="loss_scale")
    parser.add_argument("--bert_config_json", type=str, default="pytorch_pretrained_bert/bert_config.json",
                        help="bert_config_json")
    parser.add_argument("--vocab_file", type=str, default="pytorch_pretrained_bert/vocab.txt",
                        help="Path to vocab file")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs",
                        help="output_dir")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="masked_lm_prob")
    parser.add_argument("--max_predictions_per_seq", type=int, default=72, help="max_predictions_per_seq")
    parser.add_argument("--cache_dir", type=str, default='pytorch_pretrained_bert1', help="cache_dir")
    parser.add_argument("--model_name_or_path", type=str, default="pytorch_pretrained_bert1", help="model_name_or_path")
    parser.add_argument('--eval_pre_step', type=float, default=.35,
                        help="The percent of how many train with one eval run")
    parser.add_argument('--finetune_proportion', type=float, default=.25,
                        help="Detemind the proportion of the first training stage")
    parser.add_argument('--two_hop_entity_num', default=7, type=int,
                        help='The threshold value of two hop entities of each entities in knowledge graph')
    
    parser.add_argument('--entity_type',
                        default='./kgs/type_set.pkl',
                        type=str, help='entity type in knowledge graph')
    parser.add_argument('--entityOutNegbhor', default='./kgs/ent2outRel.pkl',
                        type=str, help='target node to other entity relationship')
    parser.add_argument('--entityInNegbhor', default='./kgs/ent2inRel.pkl',
                        type=str, help='target node to other entity relationship')
    parser.add_argument('--label_num', default=32,
                        type=int, help='The number of classifier labels')
    
    
    args = parser.parse_args()
    # 获取实体相关信息
    node2entity, combine_entity_type_dict,entityOutNegbhor,entityInNegbhor = entity_info(args)
    type2id,type_embedd = entity_type_initialize(combine_entity_type_dict)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
    # train_examples = None
    num_train_optimization_steps = None

    if args.do_train:
        model, missing_keys = cMeForRE.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            cache_dir=args.cache_dir,
            entity_embedding=embed,
            entity_type_embedding=type_embedd,
            transfer_matrix = transfer_matrix,
            e_dim=e_dim,
            label_num = args.label_num
        )

        train_dataset = OurENRIEDataset(args=args,
                                        data_path=args.pretrain_train_path,
                                        max_seq_length=args.max_seq_length,
                                        masked_lm_prob=args.masked_lm_prob,
                                        max_predictions_per_seq=args.max_predictions_per_seq,
                                        tokenizer=tokenizer,
                                        node2entity=node2entity,
                                        entity_dict_init =entity_dict,
                                        entity_type=combine_entity_type_dict,
                                        type_embedd=type_embedd,type2id=type2id,
                                        entityOutNegbhor=entityOutNegbhor,entityInNegbhor=entityInNegbhor)
        eval_dataset = OurENRIEDataset(args=args,
                                       data_path=args.pretrain_dev_path,
                                       max_seq_length=args.max_seq_length,
                                       masked_lm_prob=args.masked_lm_prob,
                                       max_predictions_per_seq=args.max_predictions_per_seq,
                                       tokenizer=tokenizer,
                                       node2entity=node2entity,
                                       entity_dict_init=entity_dict,
                                       entity_type=combine_entity_type_dict,
                                       type_embedd=type_embedd,type2id=type2id,
                                       entityOutNegbhor=entityOutNegbhor,entityInNegbhor=entityInNegbhor)

        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()


    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        # try:
        #     from apex.parallel import DistributedDataParallel as DDP
        # except ImportError:
        #     raise ImportError(
        #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model,device_ids=[args.local_rank], output_device=args.local_rank)
        # pass
    elif n_gpu > 1:
        assert(False)
        model = torch.nn.DataParallel(model)
    n_gpu = max(n_gpu, 1)
    # for key,param in model.named_parameters(recurse=True):
    # print(key)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    new_add_param = [(n, p) for n, p in param_optimizer if n in missing_keys]
    pretrain_parm = [(n, p) for n, p in param_optimizer if n not in missing_keys]

    LR_AMP = 3
    new_optimizer_grouped_parameters = [
        {'params': [p for n, p in new_add_param if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,'lr':args.learning_rate * LR_AMP},
        {'params': [p for n, p in new_add_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':args.learning_rate * LR_AMP}
    ]
    old_optimizer_grouped_parameters = [
        {'params': [p for n, p in pretrain_parm if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,'lr':args.learning_rate },
        {'params': [p for n, p in pretrain_parm if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':args.learning_rate}
    ]

    optimizer = None
    scheduler = None
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(new_optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = AdamW(new_optimizer_grouped_parameters + old_optimizer_grouped_parameters,
                          lr=args.learning_rate)
        # scheduler = get_double_linear_schedule_with_warmup(optimizer,args.num_training_steps,args.warmup_proportion)
        # Note The schedule set the new warm start right at the half of training process.
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_proportion*num_train_optimization_steps,
        # num_training_steps=num_train_optimization_steps)
    global_step = 0
    best_loss = 100000
    #    eval_pern_steps=570

    if args.do_train:

        total_eval_step = int(len(eval_dataset) / args.eval_batch_size)
        # train_features_len = len(train_dataset)
        
        if(args.local_rank != -1):
            train_sampler = DistributedSampler(train_dataset,shuffle=True)
        else:
            train_sampler = RandomSampler(train_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                       num_workers=3,pin_memory=True)
        # train_dataloader2 = DataLoader(train_data2, sampler=train_sampler2, batch_size=args.train_batch_size, shuffle=False)
        model.train()
        # Run prediction for full data
        # eval_sampler1 = SequentialSampler(eval_data1)
        # eval_sampler2 = SequentialSampler(eval_data2)
        eval_sampler = None
        if(args.local_rank != -1):
            eval_sampler = DistributedSampler(eval_dataset,shuffle=False)
        else:
            eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                      num_workers=3,pin_memory=True)
        # total_step = len(train_dataloader) 
        # eval_step = int(total_step * args.eval_pre_step)
        # scheduler = get_double_linear_schedule_with_warmup(optimizer, total_step, args.warmup_proportion,
        #                                                    args.finetune_proportion)
        total_step = len(train_dataloader) * args.num_train_epochs
        eval_step = int( len(train_dataloader) * args.eval_pre_step)
        scheduler = get_linear_schedule_with_warmup(optimizer, total_step*args.warmup_proportion, total_step)

        if(args.local_rank <=0):
            logger.info("Start Train...")
        for e in trange(int(args.num_train_epochs), desc="Epoch"):
            nb_tr_examples = 0
            tr_loss = 0
            loss_step = 0
            if(args.local_rank != -1):
                train_dataloader.sampler.set_epoch(e)
            for step, batch in enumerate(tqdm(train_dataloader,desc='Training')):
                batch0 = tuple(t.to(device) for t in batch)
                output = model(*batch0)
                loss = output[0]
                logist = output[1]
                # print(torch.argmax(logist,dim=-1))
                # print(loss.item())
                # print(logist)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                nb_tr_examples += train_dataloader.batch_size

                tr_loss += loss.item() * args.gradient_accumulation_steps
                loss_step += 1
                if( (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader)-1):
                    if args.fp16:
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    # if (global_step == int(total_step * args.finetune_proportion)):
                    #     optimizer.add_param_group(old_optimizer_grouped_parameters)

                if (((step + 1) % eval_step ) == 0 or (step+1)==len(train_dataloader)):
                    best_loss = evaluate(args, model, eval_dataloader, device, (best_loss, e, tr_loss / loss_step),
                                         total_eval_step)
                    tr_loss = 0
                    loss_step = 0
        # Run evaluation when finishing the training process.
        best_loss = evaluate(args, model, eval_dataloader, device, (best_loss, e, tr_loss / loss_step),
                                         total_eval_step)
        if(args.local_rank <=0):
            logger.info('Training Done!')

if __name__ == "__main__":
    main()

