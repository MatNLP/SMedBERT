from __future__ import absolute_import, division, print_function

import json

import csv
import logging
import os

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
from sklearn.metrics import matthews_corrcoef, f1_score
import jieba
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME, MAX_TARGET, MAX_NUM_PAIR, MAX_LONG_WORD_USE, \
    MAX_SHORT_WORD_USE, MAX_SEG_USE
from pytorch_pretrained_bert.modeling import BertForHyperPreTraining, ERNIEForPreTraining, \
    BertConfig  
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.Trie import Trie
from torch.optim import AdamW
from transformers.optimization import  get_linear_schedule_with_warmup, get_double_linear_schedule_with_warmup
from pytorch_pretrained_bert.optimization import BertAdam
import argparse
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Process
import gc
import pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



entity_file = open('kgs/ch_entity2id.txt', 'r',
                   encoding='utf-8')  
entity_dict = {}
entity_file.readline()
id2entity = {}

for line in entity_file:
    name, idx = line.rstrip().split('\t')
    entity_dict[name] = int(idx) + 1
    id2entity[idx] = name
entity_file.close()


idx2weight = {}
entity_file = open('global_PageRank.json', 'r',
                   encoding='utf-8')
name2weight = json.load(entity_file)
entity_file.close()
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
mask_count = dict.fromkeys(entity_dict.keys(), MAX_PRE_ENTITY_MASK)

for key in entity_dict.keys():
    if (len(key) > 1 and not key.isdigit()):
        ww_tree.insert(key)
# ebmedding.

entity_dict_index2str = {value: key for key, value in entity_dict.items()}


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


vecs = []
vecs.append([0] * 100)  # CLS
for vec in embedding_list:
    vecs.append(vec)
embedding_list = vecs
embed = torch.FloatTensor(vecs)
embed = torch.nn.Embedding.from_pretrained(embed,freeze=False)


del  js_file, entity_file
MAX_SEQ_LEN = 512
WORD_CUTTING_MAX_PAIR = 50
GUEESS_ATOM_MAX_PAIR = 50
POS_NEG_MAX_PAIR = 10
word_stat = {}
entity_hit_stat = {}


SAVE_THELD = .1
logger = logging.getLogger(__name__)
rng = Rd(43)
import re

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
        self.examples = []
        self.vocab = list(tokenizer.vocab.keys())
        self.entity_type = entity_type
        self.type_embedd = type_embedd
        self.entity_dict_reverse = {value: key for key, value in entity_dict_init.items()}
        self.entityInNegbhor = entityInNegbhor
        self.entityOutNegbhor = entityOutNegbhor
        self.__read_data__()

    def __getitem__(self, index):
        example = self.examples[index]
        line = example.rstrip()
        line = self.http_remover.sub("", line).replace('##', "").strip()
        line = line 
        example = self.__get_example__(line)
        feature = self.__get_feature__(example)
        tensor_tuple = self.__feature2tensor__(feature)
        # tensor_tuple = (torch.cat(tensor_tuple[0], dim=0), tensor_tuple[1])
        return tensor_tuple

    def __get_example__(self, line):
        line = line.rstrip()
        
        tokens_a = self.tokenizer.tokenize(line)
        insert_pos = len(tokens_a) // 2
        offset = 0
        while (tokens_a[insert_pos].startswith('##')):
            if (not tokens_a[insert_pos - offset].startswith('##')):
                insert_pos -= offset
                break
            elif (not tokens_a[insert_pos + offset].startswith('##')):
                insert_pos += offset
                break
            offset += 1
            if (insert_pos - offset < 0 or insert_pos + offset >= len(tokens_a)):
                print('Head Hard Found for sentence: {}'.format(tokens_a))
                break
        head = tokens_a[:insert_pos]
        tail = tokens_a[insert_pos:]
        if (len(tokens_a) > self.max_num_tokens):
            truncate_seq_pair(head, tail, self.max_num_tokens, rng)
        sop_label = 0
        if (random() > .5):
            head, tail = tail, head
            sop_label = 1
        if (head[0].startswith('##')): head[0] = head[0][2:]
        tokens_a = ["[CLS]"] + head + ["[SEP]"] + tail + ["[SEP]"]

        tokens_a, masked_lm_positions, masked_lm_labels, two_hop_to_pred, two_hop_to_insert = create_wwm_lm_predictions(
            tokens_a, self.masked_lm_prob, self.max_predictions_per_seq, self.vocab, self.tokenizer)
        # We use the sop task. the simple implement that.
        tokens = tokens_a
        segment_ids = [0 for _ in range(len(head) + 2)] + [1 for _ in range(len(tail) + 1)]
        assert (len(segment_ids) == len(tokens))

        example = {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels,
            'sop_label': sop_label,
            'two_hop_to_pred': two_hop_to_pred,
            'two_hop_to_insert': two_hop_to_insert
        }
        return example

    def __get_feature__(self, example):
        args = self.args
        max_seq_length = self.max_seq_length
        tokens = example["tokens"]
        segment_ids = example["segment_ids"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_labels = example["masked_lm_labels"]
        # entity_pos = example['entity_pos']
        
        two_hop_to_insert = example['two_hop_to_insert']
        two_hop_to_pred = example['two_hop_to_pred']

        sop_label = example['sop_label']
        args = self.args
        
        kc_entity_se_index_insert_array = np.zeros((MAX_TARGET,2),dtype=np.int)
        kc_entity_two_hop_labels_insert_array = np.full((MAX_TARGET,args.two_hop_entity_num),fill_value=-1,dtype=np.int)
        
        kc_entity_out_or_in_insert_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        kc_entity_two_hop_rel_types_insert_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        kc_entity_infusion_pos_insert_array = np.zeros(max_seq_length,dtype=np.int)
        kc_entity_two_hop_types_insert_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        
        kc_entity_one_hop_ids_insert_array = np.zeros(MAX_TARGET,dtype=np.int) # Note that Label has -1 as padding while ids use 0.
        kc_entity_one_hop_types_insert_array = np.zeros(MAX_TARGET,dtype=np.int)

        insert_params = (kc_entity_se_index_insert_array ,kc_entity_two_hop_labels_insert_array,
        kc_entity_out_or_in_insert_array ,
        kc_entity_two_hop_rel_types_insert_array, 
        kc_entity_infusion_pos_insert_array ,
        kc_entity_two_hop_types_insert_array ,
        kc_entity_one_hop_ids_insert_array ,
        kc_entity_one_hop_types_insert_array)

        for index,key in  enumerate(two_hop_to_insert):
            word = two_hop_to_insert[key]
            start,end = key,key+len(word)
            
            tmp_set = [] 
            if(word in self.entityInNegbhor):
                for rel,e in self.entityInNegbhor[word]:
                    tmp_set.append((e,rel,-1))
            if(word in self.entityOutNegbhor):
                for rel,e in self.entityOutNegbhor[word]:
                    tmp_set.append((e,rel,1))
            
            # shuffle(tmp_set) # We do not random shuffle and hop the model can overfit on KAC target.
            tmp_set = sorted(tmp_set,key=key_fn,reverse=True)
            tmp_set = tmp_set[:args.two_hop_entity_num]

            kc_entity_se_index_insert_array[index] = [start,end-1]
            tmp = list(t[2] for t in tmp_set)
            kc_entity_out_or_in_insert_array[index][:len(tmp)] = tmp
            tmp = list(self.entity_dict[t[0]] for t in tmp_set)
            kc_entity_two_hop_labels_insert_array[index][:len(tmp)] = tmp
            tmp = list(rel_dict[t[1]] for t in tmp_set)
            kc_entity_two_hop_rel_types_insert_array[index][:len(tmp)] = tmp
            tmp = list(self.type2id[self.entity_type[t[0]]] for t in tmp_set)
            kc_entity_two_hop_types_insert_array[index][:len(tmp)] = tmp

            kc_entity_one_hop_ids_insert_array[index] = self.entity_dict[word]
            kc_entity_one_hop_types_insert_array[index] = self.type2id[self.entity_type[word]]
            kc_entity_infusion_pos_insert_array[start:end] = index + 1

        kc_entity_se_index_pred_array = np.zeros((MAX_TARGET,2),dtype=np.int)
        kc_entity_two_hop_labels_pred_array = np.full((MAX_TARGET,args.two_hop_entity_num),fill_value=-1,dtype=np.int)
        kc_entity_out_or_in_pred_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        kc_entity_two_hop_rel_types_pred_array = np.zeros((MAX_TARGET,args.two_hop_entity_num),dtype=np.int)
        for index,key in  enumerate(two_hop_to_pred):
            word = two_hop_to_pred[key]
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
            kc_entity_se_index_pred_array[index] = [start,end-1]
            tmp = list(t[2] for t in tmp_set)
            kc_entity_out_or_in_pred_array[index][:len(tmp)] = tmp
            tmp = list(self.entity_dict[t[0]] for t in tmp_set)
            kc_entity_two_hop_labels_pred_array[index][:len(tmp)] = tmp
            tmp = list(rel_dict[t[1]] for t in tmp_set)
            kc_entity_two_hop_rel_types_pred_array[index][:len(tmp)] = tmp

        pred_params = (kc_entity_se_index_pred_array,kc_entity_two_hop_labels_pred_array,
                        kc_entity_out_or_in_pred_array,kc_entity_two_hop_rel_types_pred_array)


        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        masked_label_ids = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)
        assert (len(masked_label_ids) == len(masked_lm_positions))
        input_array = np.full(max_seq_length, dtype=np.int, fill_value=self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0] )
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.int)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.int)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
        lm_label_array[masked_lm_positions] = masked_label_ids


        feature = InputFeatures(input_ids=input_array,
                                input_mask=mask_array,
                                segment_ids=segment_array,
                                label_id=lm_label_array,
                                sop_label=sop_label,insert_params=insert_params,pred_params=pred_params)
                                
        return feature

    def __feature2tensor__(self, feature):
        f = feature
        all_input_ids = torch.tensor(f.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(f.input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(f.segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(f.label_id, dtype=torch.long)
        all_sop_labels = torch.tensor(f.sop_label, dtype=torch.long).unsqueeze(-1)

        insert_tensors = tuple( torch.tensor(t,dtype=torch.long) for t in f.insert_params)
        pred_tensors = tuple( torch.tensor(t,dtype=torch.long) for t in f.pred_params)
        
        return (all_input_ids, all_segment_ids,all_input_mask, all_label_ids,all_sop_labels) + insert_tensors + pred_tensors

    def __len__(self):
        return len(self.examples)

    def __read_data__(self):
        fr = open(self.data_path, "r", encoding='utf-8')
        self.http_remover = re.compile(r'https?://[a-zA-Z0-9.%+()/&:;<=●>?@^_`{|}~]*', re.S)
        examples = fr.readlines()
        # max_total_example=len(exmaples)
        lines = []
        for line in tqdm(examples, desc='loading train / dev examples'):
            line= line
            line = self.http_remover.sub("", line).replace('##', "").strip()
            count = 0
            for w in line:
                if (is_chinese_char(w)): count += 1
                if (count >= self.min_len // 2): break
            if (len(line) <= self.min_len or count < self.min_len // 2): continue
            lines.append(line + '\n' )
        self.examples = lines
        fr.close()





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

    def __init__(self, input_ids, input_mask, segment_ids, label_id,sop_label,insert_params, pred_params):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sop_label = sop_label
        self.insert_params = insert_params
        self.pred_params = pred_params

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

def create_wwm_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab, tokenizer):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    global entity_hit_stat
    global word_stat
    chars = [token for token in tokens]
    cand_map = dict(zip(range(len(tokens)), range(len(tokens))))

    entity_pos = {}
    # entity_ids = []  # This list would like [ [0] , [1,2] , [3,4,5]] to pack wwm word toghter
    # We use it to 保存分词预测和特征词原子词相互预测的位置
    cand_index = 0
    skip_index = set()
    chars_index = []
    assert (chars[0] == '[CLS]')
    # words_hit = []
    
    while (cand_index < len(chars)):
        if (isSkipToken(chars[cand_index])):
            skip_index.add(cand_index)
            # entity_ids.append(-1)
        elif ( ww_tree.startsWith(chars[cand_index])):
            c = ww_tree.get_lengest_match(chars, cand_index)
            if (c == None):
                chars_index.append(cand_index)
                pass
            else:
                word = ''.join(chars[cand_index:c + 1])
                word_stat[word] = word_stat.get(word,0)+1
                assert (word in entity_dict)
                entity_pos[cand_index]=word
                cand_index += len(word)
                continue
        else:
            chars_index.append(cand_index)
        cand_index += 1
    
    if(len(entity_pos) not in entity_hit_stat):
        entity_hit_stat[len(entity_pos)] = 1
    else:
        entity_hit_stat[len(entity_pos)] += 1

    words_hit= list(entity_pos.items())
    num_to_mask = min(max_predictions_per_seq,
                    max(1, int(round((len(chars)-len(skip_index))* masked_lm_prob))))
    # mask_indices = []
    has_masked = 0
    shuffle(words_hit)

    mask_pos = []
    masked_token_labels = []
    word_select = set()
    for pos,(start_index,word) in enumerate(words_hit):
        if(has_masked >= num_to_mask): break
        if(has_masked + len(word) > num_to_mask): continue
        word_select.add(pos)
        for w in words_hit:
            if(w in mask_count):
                mask_count[w]-=1
                if(mask_count[w]==0):
                    mask_count.pop(w)
                    ww_tree.delete(w)
        index_set = list(range(start_index,start_index+len(word)))
        mask_pos.extend(index_set)
        for index in index_set:
            masked_token = chars[index]
            if(random()<.5):
                masked_token = "[MASK]"
            elif(random()<.8):
                masked_token = choice(vocab)
            chars[index] = masked_token
            masked_token_labels.append(tokens[index])
    
    shuffle(chars_index)
    for index in chars_index:
        if(has_masked >= num_to_mask): break
        has_masked += 1
        masked_token_labels.append(tokens[index])
        mask_pos.append(index)
        masked_token = chars[index]
        if(random()<.5):
            masked_token = "[MASK]"
        elif(random()<.8):
            masked_token = choice(vocab)
        chars[index] = masked_token

    two_hop_to_pred = {}
    two_hop_to_insert = {}
    for i,(k,v) in enumerate(words_hit):
        if(i in word_select):
            two_hop_to_insert[k] = v
        else:
            two_hop_to_pred[k] = v
    

    assert (len(mask_pos) <= MAX_SEQ_LEN)
    assert (len(masked_token_labels) == len(mask_pos))
    
    assert (len(chars) <= MAX_SEQ_LEN)
    assert (len(set(mask_pos)) == len(mask_pos))
    if(len(two_hop_to_insert)>MAX_TARGET):
        two_hop_to_insert = get_reduce_dict(two_hop_to_insert,MAX_TARGET)
    if(len(two_hop_to_pred)>MAX_TARGET):
        two_hop_to_pred = get_reduce_dict(two_hop_to_pred,MAX_TARGET)
    assert (len(two_hop_to_insert)<=MAX_TARGET)

    return chars, mask_pos, masked_token_labels, two_hop_to_pred, two_hop_to_insert


def convert_examples_to_features(args, examples, max_seq_length, tokenizer, do_gc=False):
    features = []
    example_num = len(examples)
    names_list = []
    save_pre_step = max(int(.25 * example_num), 1)

    for f_index in tqdm(range(example_num), desc="Converting Feature"):
        #    for i, example in enumerate(examples):
        # print(f_index)
        example = examples[-1]
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


        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
        assert (len(masked_label_ids) == len(masked_lm_positions))
        input_array = np.zeros(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.int)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.int)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
        lm_label_array[masked_lm_positions] = masked_label_ids

        entity_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
        entity_array[:len(entiy_ids)] = entiy_ids


        feature = InputFeatures(input_ids=input_array,
                                input_mask=mask_array,
                                segment_ids=segment_array,
                                label_id=lm_label_array,
                                entiy_ids=entity_array,
                                entity_mapping=entity_ids_mapping,
                                entity_mapping_mask=entity_ids_mapping_mask,
                                sop_label=sop_label)
        features.append(feature)
        examples.pop()
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
    examples = []
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
    
def reduce_tensor(tensor,ws=2):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= ws
    return rt

def evaluate(args, model, eval_dataloader, device, loss_bag, eval_step):
    torch.cuda.empty_cache()
    best_loss, epoch, tr_loss = loss_bag
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0

    for batch in tqdm(eval_dataloader,desc='Evaluation'):
        # logger.info('Testing step: {} / {}'.format(nb_eval_steps, eval_step))
        batch0 = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            loss = model(*batch0)
            if(args.local_rank>=0):
                loss = reduce_tensor(loss)
        eval_loss += loss.mean().item()
        nb_eval_steps += 1
    
    eval_loss = eval_loss / nb_eval_steps
    if eval_loss < best_loss:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        if(args.local_rank > 0):
            torch.distributed.barrier()
        if(args.local_rank <= 0):
            torch.save(model_to_save.state_dict(), output_model_file)
        if(args.local_rank == 0):
            torch.distributed.barrier()
        best_loss = eval_loss
    if(args.local_rank <=0 ):
        logger.info(
        "============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n" % (epoch, tr_loss, eval_loss))
    
    torch.cuda.empty_cache()
    return min(best_loss, eval_loss)

def entity_info(args):
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
                        default="./data_aug/high_low_corpus.txt",
                        help="pretrain train path to file")
    parser.add_argument("--pretrain_dev_path", type=str,
                        default="./data_aug/high_low_corpus.txt",
                        help="pretrain dev path to file")
    parser.add_argument("--max_seq_length", type=int, default=512, help="max seq length of input sequences")
    parser.add_argument("--do_train", type=bool, default=True, help="If do train")
    parser.add_argument("--do_lower_case", type=bool, default=True, help="If do case lower")
    parser.add_argument("--train_batch_size", type=int, default=1024, help="train_batch_size")  # May Need to finetune
    parser.add_argument("--eval_batch_size", type=int, default=64, help="eval_batch_size")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="num_train_epochs")
    parser.add_argument("--learning_rate", type=float, default=6e-5, help="learning rate")  # May Need to finetune
    parser.add_argument("--warmup_proportion", type=float, default=.05,
                        help="warmup_proportion")  # May Need to finetune
    parser.add_argument("--no_cuda", type=bool, default=False, help="prevent use GPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="If we are using cluster for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64,
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
    parser.add_argument('--eval_pre_step', type=float, default=.126,
                        help="The percent of how many train with one eval run")
    parser.add_argument('--finetune_proportion', type=float, default=.05,
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
        # print(torch.distributed.get_world_size())
        # assert(torch.distributed.get_world_size() == 2)

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
        model, missing_keys = ERNIEForPreTraining.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            cache_dir=args.cache_dir,
            entity_embedding=embed,
            entity_type_embedding=type_embedd,
            transfer_matrix = transfer_matrix,
            e_dim=e_dim
        )
        train_dataset = None
        eval_dataset = None
        if(args.local_rank > 0):
            torch.distributed.barrier()
        if(args.local_rank <=0 ):
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
            if(args.local_rank == 0):
                with open('./run_tmp/training.pt','wb+') as train_f:
                    torch.save(train_dataset,train_f)
                with open('./run_tmp/test.pt','wb+') as test_f:
                    torch.save(eval_dataset,test_f)
                torch.distributed.barrier()
                print('Save Successfully!')
        if(args.local_rank != -1):
            with open('./run_tmp/training.pt','rb') as train_f:
                train_dataset = torch.load(train_f)
            with open('./run_tmp/test.pt','rb') as test_f:
                eval_dataset = torch.load(test_f)
            print('Loading successfully!')

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

    new_optimizer_grouped_parameters = [
        {'params': [p for n, p in new_add_param if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in new_add_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    old_optimizer_grouped_parameters = [
        {'params': [p for n, p in pretrain_parm if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in pretrain_parm if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
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
        optimizer = AdamW(new_optimizer_grouped_parameters,
                          lr=args.learning_rate)

    for g in old_optimizer_grouped_parameters:
        optimizer.add_param_group(g)
    global_step = 0
    best_loss = 100000
    #    eval_pern_steps=570

    if args.do_train:

        total_eval_step = int(len(eval_dataset) / args.eval_batch_size)
        train_features_len = len(train_dataset)
        
        if(args.local_rank != -1):
            train_sampler = DistributedSampler(train_dataset,shuffle=True)
        else:
            train_sampler = RandomSampler(train_dataset)


        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        model.train()

        eval_sampler = SequentialSampler(eval_dataset)
        
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                      num_workers=3,pin_memory=True)
        total_step = len(train_dataloader) * args.num_train_epochs
        eval_step = int(len(train_dataloader) * args.eval_pre_step)
        # scheduler = get_double_linear_schedule_with_warmup(optimizer, total_step, args.warmup_proportion,
        #                                                    args.finetune_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer,args.warmup_proportion*total_step,total_step)
        if(args.local_rank <=0):
            logger.info("Start Train...")
        for e in trange(int(args.num_train_epochs), desc="Epoch"):
            nb_tr_examples = 0
            tr_loss = 0
            loss_step = 0

            for step, batch in enumerate(tqdm(train_dataloader)):

                nb_tr_examples += train_dataloader.batch_size

        if(args.local_rank <=0):
            logger.info('Training Done!')
    global entity_hit_stat
    global word_stat
    print(entity_hit_stat)
    with open('entity_hit_stat.pkl','wb+') as bf:
        pickle.dump(entity_hit_stat,bf)
    print(word_stat)
    with open('word_stat.pkl','wb+') as bf:
        pickle.dump(word_stat,bf)
    

if __name__ == "__main__":
    main()
    # with open('entity_hit_stat.pkl','wb+') as bf:
    #     pickle.dump(entity_hit_stat,bf)
    # with open('word_stat.pkl','wb+') as bf:
    #     pickle.dump(word_stat,bf)
    
    
