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

from pytorch_pretrained_bert.modeling import BertForHyperPreTraining, cMeForPreTraining,cMeForSoftMaxNER, \
    BertConfig  # Note that we use the sop, rather than nsp task.
# from code.knowledge_bert.modeling import BertForPreTraining

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.Trie import Trie
from transformers.optimization import AdamW, get_linear_schedule_with_warmup # , get_double_linear_schedule_with_warmup
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
from metric import SeqEntityScore
import re
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/NER')

# torch.autograd.set_detect_anomaly(True)

labels_list = {'zh_msra': ['B-NS', 'B-NT', 'B-NR', 'I-NS', 'I-NT', 'I-NR','O'],
               'cmedqaner': ['B-crowd', 'I-crowd', 'B-symptom', 'I-symptom',
                'B-body', 'B-treatment', 'I-treatment', 'I-body', 'B-time', 'I-time', 'B-drug',
                'I-drug', 'B-feature', 'I-feature', 'B-physiology', 'I-physiology', 'B-test',
                'I-test', 'B-department', 'I-department','B-disease', 'I-disease','O'],
                'dxy_clean_ner': ['B-XDEPARTMENT', 'B-XFOOD', 'B-XSEASON', 'B-XEQUIPMENT', 'B-XPOPULATION', 'B-XEXAMINATION', 
                'B-XDISEASE', 'B-XESSENTIAL', 'B-XDRUG', 'B-XENVIRONMENT', 'B-XBODY', 'B-XGENE', 'B-XRESULT', 'B-XSYMPTOM', 
                'B-XTREATMENT', 'B-XINSPECTION', 'B-XPATHOGEN', 'B-XREGION', 'B-XORGANIZATION', 'B-XACTIVITY', 'B-XSURGERY',
                'I-XDEPARTMENT', 'I-XFOOD', 'I-XSEASON', 'I-XEQUIPMENT', 'I-XPOPULATION', 'I-XEXAMINATION', 'I-XDISEASE', 
                'I-XESSENTIAL', 'I-XDRUG', 'I-XENVIRONMENT', 'I-XBODY', 'I-XGENE', 'I-XRESULT', 'I-XSYMPTOM', 'I-XTREATMENT', 
                'I-XINSPECTION', 'I-XPATHOGEN', 'I-XREGION', 'I-XORGANIZATION', 'I-XACTIVITY', 'I-XSURGERY','O'] }

CLIP_UNTIL = 105
CLIP_VALUE = 5
SKIP_VALUE = 7

entity_file = open('kgs/ch_entity2id.txt', 'r',encoding='utf-8') 
entity_dict = {}
entity_file.readline()
max_count = 0
for line in entity_file:
    name, idx = line.rstrip().split('\t')
    entity_dict[name] = int(idx) + 1
entity_file.close()

idx2weight = {}
entity_file = open('global_PageRank.json', 'r',
                   encoding='utf-8')
name2weight = json.load(entity_file)
entity_file.close()
name2weight['響'] = 0
for key in entity_dict:
    idx2weight[str(entity_dict[key])] = name2weight[key]
idx2weight['0'] = 0

rel_file = open('kgs/ch_relation2id.txt', 'r',
                   encoding='utf-8')
rel_file.readline()
rel_dict = {}
for line in rel_file:
    name, idx = line.rstrip().split('\t')
    rel_dict[name] = int(idx) + 1

ww_tree = Trie()
MAX_PRE_ENTITY_MASK = 100
with open('meta_ent.pkl','rb') as tf:
    abstract_obj = set(pickle.load(tf))

# for key in entity_dict.keys():
#     if (len(key) > 1 and not key.isdigit() and key not in abstract_obj):
#         ww_tree.insert(key)
    # entity_dict[key] += 1  # For the zero embedding is used as Pad ebmedding.

# entity_dict_index2str = {value: key for key, value in entity_dict.items()}

# keys = ['ent_embeddings', 'rel_embeddings']
js_file = open('kgs/transr_transr.json', 'r',
               encoding='utf-8')  # Note that we must add the first entity as EMPTY.
js_dict = json.load(js_file)
js_file.close()
embedding_list = js_dict['ent_embeddings.weight']
transfer_matrix_list = js_dict['transfer_matrix.weight']
print(len(embedding_list))

relation_list = js_dict['rel_embeddings.weight']
e_dim = len(embedding_list[0])
assert(len(transfer_matrix_list[0]) % e_dim == 0)
r_dim = len(transfer_matrix_list[0]) // e_dim
assert(len(transfer_matrix_list) == len(relation_list))
# print('There are:',len(transfer_matrix_list),' entities in total.')
for i in range(len(transfer_matrix_list)):
    transfer_matrix_list[i].extend(relation_list[i])

# transfer_matrix_list = [[0]*(r_dim*e_dim)] + transfer_matrix_list
transfer_matrix = torch.FloatTensor(transfer_matrix_list)
transfer_matrix = transfer_matrix.view(transfer_matrix.size(0),r_dim,e_dim+1)
transfer_matrix = torch.bmm(transfer_matrix.transpose(-1,-2),transfer_matrix)
transfer_matrix = torch.cat((torch.zeros(1,e_dim+1,e_dim+1),transfer_matrix),dim=0)
transfer_matrix = torch.nn.Embedding.from_pretrained(transfer_matrix.view(-1,(e_dim+1)*(e_dim+1)),freeze=False)
# print(transfer_matrix.weight.size())
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
word_stat = {}
entity_hit_stat = {}
# MAX_TARGET=32
# MAX_NUM_PAIR=25

SAVE_THELD = .1
logger = logging.getLogger(__name__)
rng = Rd(43)
import re

def key_fn(obj):
    idx = entity_dict[obj[0]]-1
    weight = idx2weight[str(idx)] if str(idx) in idx2weight else 0
    return weight

class OurENRIEDataset(Dataset):
    def __init__(self, args, data_path, max_seq_length, masked_lm_prob,
                 max_predictions_per_seq, tokenizer, node2entity, entity_dict_init, entity_type, type_embedd,type2id,entityOutNegbhor,entityInNegbhor, 
                 label2id, min_len=128):
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
        self.label2id = dict( (w,i) for i,w in enumerate(label2id) )
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
        tl = [ (t[0],t[2:]) for t in data ]
        head,label = zip(*tl)
        # old_head = head
        # print(label)
        s_list = []
        l_list = []
        s_tmp = []
        current_stat = 'O'
        for idx,l in enumerate(label):
            if( current_stat == 'O'):
                if(len(l) != 1):
                    l_list.append(current_stat)
                    s_list.append(s_tmp)
                    s_tmp = [head[idx]]
                    current_stat = l[2:]
                else:
                    s_tmp.append(head[idx])
            else:
                if(l[0]=='B' or len(l)==1):
                    l_list.append(current_stat)
                    s_list.append(s_tmp)
                    s_tmp = [head[idx]]
                    current_stat = l[2:] if len(l)>1 else l
                else:
                    s_tmp.append(head[idx])
        s_list.append(s_tmp)
        l_list.append(label[-1] if len(label[-1])==1 else label[-1][2:])
        assert(len(s_list) == len(l_list))
        new_head = []
        new_label = []
        for i in range(len(s_list)):
            if(len(s_list[i])<=0): continue
            s=''.join(s_list[i])
            t_s = self.tokenizer.tokenize(s)
            l = l_list[i]
            if(len(l)==1):
                l_s = ['O'] * len(t_s)
            else:
                l_s = ['I-'+ l] * len(t_s)
                l_s[0] = 'B-'+l
            new_head.extend(t_s)
            new_label.extend(l_s)
        head = new_head
        label = new_label
        # print(label)
        # assert(False)
        # assert(False)
        try:
            assert(len(head) == len(label))
        except:
            print(len(head))
            print(len(label))
            # print(old_head)
            print(head)
            print(label)
            assert(False)
        # tail = self.tokenizer.tokenize(tail) if tail else tail
        head = head[:(MAX_SEQ_LEN-2)]
        label = label[:(MAX_SEQ_LEN-2)]
        return (head,None,label)
    
    def __tokens2feature__(self,head,tail,label):
        tokens = head
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        label  = ['O'] + list(label) + ['O']
        segment_ids = [0] * len(tokens)
        if(tail): 
            tail_tokens = tail + ['[SEP]']
            segment_ids += [1]*len(tail_tokens)
            tokens += tail_tokens
        tokens, entity_pos,entity_pos2num = create_entity_pos(tokens,self.tokenizer)
        # tokens, entity_pos = convert_sentence_to_tokens(tokens,self.tokenizer)
        
        assert(len(tokens) == len(label))
        assert(len(segment_ids) == len(label))

        example ={
            'tokens': tokens,
            'segment_ids': segment_ids,
            'entity_pos': entity_pos,
            'entity_pos2num': entity_pos2num,
            'label': [ self.label2id[l] for l in label ]
        }
        return example

    def __get_feature__(self, example):
        args = self.args
        max_seq_length = self.max_seq_length

        tokens = example["tokens"]
        segment_ids = example["segment_ids"]
        entity_pos = example['entity_pos']
        label = example['label']
        entity_pos2num = example['entity_pos2num']
        two_hop_to_insert = entity_pos
        args = self.args
        
        kc_entity_se_index_insert_array = np.zeros((MAX_TARGET,2),dtype=np.int)
        kc_entity_two_hop_labels_insert_array = np.full((MAX_TARGET,args.two_hop_entity_num),fill_value=-1,dtype=np.int)
        
        kc_entity_out_or_in_insert_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        kc_entity_two_hop_rel_types_insert_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        kc_entity_infusion_pos_insert_array = np.zeros(max_seq_length,dtype=np.int)
        kc_entity_two_hop_types_insert_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        
        kc_entity_one_hop_ids_insert_array = np.zeros(MAX_TARGET,dtype=np.int) # Note that Label has -1 as padding while ids use 0.
        kc_entity_one_hop_types_insert_array = np.zeros(MAX_TARGET,dtype=np.int)

        for index,key in  enumerate(two_hop_to_insert):
            word = two_hop_to_insert[key]
            start,end = key,key+entity_pos2num[key]
            tmp_set = [] 
            
            if(word in self.entityInNegbhor):
                for rel,e in self.entityInNegbhor[word]:
                    tmp_set.append((e,rel,-1))
            if(word in self.entityOutNegbhor):
                for rel,e in self.entityOutNegbhor[word]:
                    tmp_set.append((e,rel,1))
            # shuffle(tmp_set) # We do not random shuffle and hop the model can overfit on KAC target.

            if(random()<.5): tmp_set = sorted(tmp_set,key=key_fn,reverse=True)
            else: shuffle(tmp_set)

            tmp_set = tmp_set[:args.two_hop_entity_num]
            kc_entity_se_index_insert_array[index] = [start,end-1]
            tmp = list(t[2] for t in tmp_set)
            kc_entity_out_or_in_insert_array[index][:len(tmp)] = tmp
            tmp = list(self.entity_dict[t[0]] for t in tmp_set)
            ### Add
            # for t in tmp_set: self.used_entity_file.write(t[0]+'\n')
            # self.used_entity_file.write(word+'\n')
            ### Add 
            kc_entity_two_hop_labels_insert_array[index][:len(tmp)] = tmp
            tmp = list(rel_dict[t[1]] for t in tmp_set)
            kc_entity_two_hop_rel_types_insert_array[index][:len(tmp)] = tmp
            tmp = list(self.type2id[self.entity_type[t[0]]] for t in tmp_set)
            kc_entity_two_hop_types_insert_array[index][:len(tmp)] = tmp

            kc_entity_one_hop_ids_insert_array[index] = self.entity_dict[word]
            kc_entity_one_hop_types_insert_array[index] = self.type2id[self.entity_type[word]]
            kc_entity_infusion_pos_insert_array[start:end] = index + 1
        
        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        input_array = np.full(max_seq_length, dtype=np.int, fill_value=self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0] )
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.int)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.int)
        segment_array[:len(segment_ids)] = segment_ids

        label_array = np.full(max_seq_length,dtype=np.int,fill_value=-1)
        label_array[:len(label)] = label

        feature = InputFeatures(input_ids=input_array,
                                input_mask=mask_array,
                                segment_ids=segment_array,
                                kc_entity_one_hop_ids=kc_entity_one_hop_ids_insert_array,
                                kc_entity_one_hop_types=kc_entity_one_hop_types_insert_array,
                                kc_entity_se_index=kc_entity_se_index_insert_array,
                                kc_entity_two_hop_labels=kc_entity_two_hop_labels_insert_array,
                                kc_entity_out_or_in=kc_entity_out_or_in_insert_array,
                                kc_entity_two_hop_rel_types=kc_entity_two_hop_rel_types_insert_array,
                                kc_entity_two_hop_types_array=kc_entity_two_hop_types_insert_array,
                                kc_entity_infusion_pos = kc_entity_infusion_pos_insert_array,
                                label = label_array)
        return feature

    def __feature2tensor__(self, feature):
        f = feature
        all_input_ids = torch.tensor(f.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(f.input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(f.segment_ids, dtype=torch.long)

        kc_entity_one_hop_ids = torch.tensor(f.kc_entity_one_hop_ids, dtype=torch.long)
        kc_entity_one_hop_types = torch.tensor(f.kc_entity_one_hop_types, dtype=torch.long)
        
        kc_entity_se_index = torch.tensor(f.kc_entity_se_index, dtype=torch.long)
        kc_entity_two_hop_labels = torch.tensor(f.kc_entity_two_hop_labels, dtype=torch.long)
        kc_entity_out_or_in = torch.tensor(f.kc_entity_out_or_in, dtype=torch.long)
        kc_entity_two_hop_rel_types = torch.tensor(f.kc_entity_two_hop_rel_types, dtype=torch.long)
        kc_entity_two_hop_types = torch.tensor(f.kc_entity_two_hop_types, dtype=torch.long)
        kc_entity_infusion_pos = torch.tensor(f.kc_entity_infusion_pos, dtype=torch.long)
        label = torch.tensor(f.label,dtype=torch.long)
        
        assert(label.size() == all_input_ids.size())
        
        return ((all_input_ids, all_segment_ids,all_input_mask, kc_entity_one_hop_ids,
                 kc_entity_one_hop_types) #+ (all_two_hop_entity_ids, all_two_hop_entity_types)
                 +(kc_entity_se_index, kc_entity_two_hop_labels, kc_entity_out_or_in, 
                 kc_entity_two_hop_rel_types,kc_entity_two_hop_types,kc_entity_infusion_pos,label)
                 )

    def __len__(self):
        return len(self.lines)

    def __read_data__(self):
        fr = open(self.data_path, "r", encoding='utf-8')
        # self.http_remover = re.compile(r'https?://[a-zA-Z0-9.%+()/&:;<=●>?@^_`{|}~]*', re.S)
        lines = fr.readlines()
        example = []
        for line in tqdm(lines, desc='loading train / dev lines'):
            assert(line[-1] == '\n')
            line = line[:-1]
            if(len(line)):
                example.append(line)
            else:
                self.lines.append(example)
                example = []
        fr.close()


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


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
        if rng.random() < 0.05:  # I do not want you delete front because you cause the head always produce [UNK]
            del trunc_tokens[0]
        else:
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

def get_reduce_dict(d,n):
    tmp = list(d.items())
    shuffle(tmp)
    tmp = tmp[:n]
    return dict(tmp)

def create_entity_pos(tokens, tokenizer):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    global entity_hit_stat
    global word_stat
    chars = [token for token in tokens]

    entity_pos = {}
    entity_pos2num = {}
    # entity_ids = []  # This list would like [ [0] , [1,2] , [3,4,5]] to pack wwm word toghter
    # We use it to 保存分词预测和特征词原子词相互预测的位置
    cand_index = 1
    skip_index = set()
    chars_index = []
    assert (chars[0] == '[CLS]')
    # words_hit = []
    
    # if(random()<.5): print(chars)
    
    while (cand_index < len(chars)):
        if (isSkipToken(chars[cand_index])):
            skip_index.add(cand_index)
            # entity_ids.append(-1)
        elif ( ww_tree.startsWith(chars[cand_index])):
            c = ww_tree.get_lengest_match(chars, cand_index)
            if (c == None):
                chars_index.append(cand_index)
            else:
                word = ''.join(chars[cand_index:c + 1])
                if(word not in entity_dict):
                    print('Found UnhashAbleWord',word,'\n',chars[cand_index:c + 1])
                entity_pos[cand_index]=word
                entity_pos2num[cand_index] = c+1 - cand_index
                cand_index += len(word)
                continue
        else:
            chars_index.append(cand_index)
        cand_index += 1

    if(len(entity_pos)>MAX_TARGET):
        entity_pos = get_reduce_dict(entity_pos,MAX_TARGET)
    assert (len(entity_pos)<=MAX_TARGET)
    return chars, entity_pos,entity_pos2num


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

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + 2)
    rng.seed(seed + 4)


def evaluate(args, model, eval_dataloader, device, loss_bag, eval_step,label2id):
    id2label = {}
    label2id = dict( (l,i) for i,l in enumerate(label2id))
    for k,v in label2id.items():
        id2label[v] = k
    metric = SeqEntityScore(id2label, markup='bio')
    torch.cuda.empty_cache()
    best_loss, epoch, tr_loss = loss_bag
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    labels = []
    # assert(False)
    for batch in eval_dataloader:
        # logger.info('Testing step: {} / {}'.format(nb_eval_steps, eval_step))
        batch0 = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            output = model(*batch0)
            loss = output[0]
            logits = output[1]
            pred = torch.argmax(logits,dim=-1)
            pred = pred.view(batch[-1].size())
            # print(pred[0])
            # print(batch[-1][0])

            metric.update(pred_paths=pred.cpu().tolist(), label_paths=batch[-1].cpu().tolist())
            # preds.extend(pred.cpu().tolist())
            # labels.extend(batch[-1].view(-1).cpu().tolist())
        eval_loss += loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    prefix = '{}'.format(eval_step)
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
        # f1_list.append(entity_info[key]['f1'])
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
    entity_type_dict['響'] = ('药品')
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
        # print(tmp_set)
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
    while(len(weights)<287): weights.append([0]*dim)
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
    parser.add_argument("--train_batch_size", type=int, default=8, help="train_batch_size")  # May Need to finetune
    parser.add_argument("--eval_batch_size", type=int, default=32, help="eval_batch_size")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="num_train_epochs")
    parser.add_argument("--learning_rate", type=float, default=1.5e-5, help="learning rate")  # May Need to finetune
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
    parser.add_argument("--model_name_or_path", type=str, default="bert_base", help="model_name_or_path")
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
    parser.add_argument('--task_name', type=str)
    
    args = parser.parse_args()

    label2id = labels_list[args.task_name]
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
    
    for key in list(entity_dict.keys()):
        if (len(key) > 1 and not key.isdigit() and key not in abstract_obj):
            ww_tree.insert(tokenizer.tokenize(key.rstrip()))
        new_key = ''.join(tokenizer.tokenize(key.rstrip()))
        entity_dict[new_key] = entity_dict[key]
        combine_entity_type_dict[new_key] = combine_entity_type_dict[key]

        if(key in entityInNegbhor): entityInNegbhor[new_key] = entityInNegbhor[key]
        if(key in entityOutNegbhor): entityOutNegbhor[new_key] = entityOutNegbhor[key]
    
    num_train_optimization_steps = None

    if args.do_train:
        model, missing_keys = cMeForSoftMaxNER.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            cache_dir=args.cache_dir,
            entity_embedding=embed,
            entity_type_embedding=type_embedd,
            transfer_matrix = transfer_matrix,
            e_dim=e_dim,
            labels_num=len(label2id)
        )
        # used_entity_file = open('used_entity.txt','w+',encoding='utf-8')
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
                                        entityOutNegbhor=entityOutNegbhor,entityInNegbhor=entityInNegbhor,label2id=label2id)
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
                                       entityOutNegbhor=entityOutNegbhor,entityInNegbhor=entityInNegbhor,label2id=label2id)

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

    LR_AMP = 5
    new_optimizer_grouped_parameters = [
        {'params': [p for n, p in new_add_param if not any(nd in n for nd in no_decay)], 'weight_decay': 0.04,'lr':args.learning_rate * LR_AMP },
        {'params': [p for n, p in new_add_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':args.learning_rate * LR_AMP}
    ]
    old_optimizer_grouped_parameters = [
        {'params': [p for n, p in pretrain_parm if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,'lr':args.learning_rate },
        {'params': [p for n, p in pretrain_parm if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':args.learning_rate }
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
                                       num_workers=4,prefetch_factor=4,pin_memory=True)
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
                                      num_workers=4,prefetch_factor=8,pin_memory=True)
        # total_step = len(train_dataloader) 
        # eval_step = int(total_step * args.eval_pre_step)
        # scheduler = get_double_linear_schedule_with_warmup(optimizer, total_step, args.warmup_proportion,
        #                                                    args.finetune_proportion)
        total_step = len(train_dataloader) * args.num_train_epochs
        eval_step = int( len(train_dataloader) * args.eval_pre_step)
        scheduler = get_linear_schedule_with_warmup(optimizer, total_step*args.warmup_proportion, total_step)
        print(eval_step)
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
                loss = model(*batch0)[0]

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
                    
                    if( step < CLIP_UNTIL):
                        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),10000)
                    else:
                        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP_VALUE)
                    if(args.local_rank <=0):
                        writer.add_scalar('Grad',grad.item(),global_step=global_step)
                        writer.add_scalar('LOSS',loss.item(),global_step=global_step)
                        writer.add_scalar('LR',max(scheduler.get_last_lr()),global_step=global_step)
                    # batch_loss = 0
                    batch_masked_lm_loss,batch_next_sentence_loss,batch_ent_loss = 0,0,0
                    if(global_step<CLIP_UNTIL or grad < SKIP_VALUE):
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (((step + 1) % eval_step) == 0  or (step + 1) == len(train_dataloader)):
                    best_loss = evaluate(args, model, eval_dataloader, device, (best_loss, e, tr_loss / loss_step),
                                         total_eval_step,label2id)
                    model.train()
                    tr_loss = 0
                    loss_step = 0
        # Run evaluation when finishing the training process.
        best_loss = evaluate(args, model, eval_dataloader, device, (best_loss, e, 1),
                                         total_eval_step,label2id)
        if(args.local_rank <=0):
            logger.info('Training Done!')
    # used_entity_file.close()


if __name__ == "__main__":
    main()
