from __future__ import absolute_import, division, print_function

import json
#import pretraining_args as args
import csv
import logging
import os
#import random
import sys
from glob import glob
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset,Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from random import random, randrange, randint, shuffle, choice, sample,uniform,randint
from random import Random as Rd
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import jieba
from pytorch_pretrained_bert.file_utils import  WEIGHTS_NAME, CONFIG_NAME, MAX_TARGET,MAX_NUM_PAIR,MAX_LONG_WORD_USE,MAX_SHORT_WORD_USE,MAX_SEG_USE
from pytorch_pretrained_bert.modeling import BertForHyperPreTraining,ERNIEForPreTraining,BertConfig # Note that we use the sop, rather than nsp task.
# from code.knowledge_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.Trie import Trie
from transformers.optimization import AdamW,get_linear_schedule_with_warmup,get_double_linear_schedule_with_warmup
from pytorch_pretrained_bert.optimization import BertAdam
import argparse
import multiprocessing
from multiprocessing import Manager
from multiprocessing import Process
import gc
import pickle

EMPTY_SIZE=1

entity_file=open('/home/dell/PycharmProjects/tmp_czr/KIM/OpenKE-master/ch_entity2id.txt','r',encoding='utf-8') # Note that we must add the first entity as EMPETY.
# entity_dict=json.load(entity_file) # Key: entity_name Id: entity_id
entity_dict={}
entity_file.readline()
for line in entity_file:
    name,idx=line.rstrip().split('\t')
    entity_dict[name]=int(idx)
ww_tree=Trie()
MAX_PRE_ENTITY_MASK=100
mask_count=dict.fromkeys(entity_dict.keys(),MAX_PRE_ENTITY_MASK)

for key in entity_dict.keys():
    if(len(key)>1 and not key.isdigit() ):
        ww_tree.insert(key)
    entity_dict[key]+=1 # For the zero embedding is used as Pad ebmedding.
entity_file.close()
entity_dict_index2str ={value:key for key,value in entity_dict.items()}

# keys = ['ent_embeddings', 'rel_embeddings']
js_file=open('./embedding.vec.json','r',encoding='utf-8') # Note that we must add the first entity as EMPETY.
embedding_list=json.load(js_file)
js_file.close()
embedding_list=embedding_list['ent_embeddings']


def euclidean(p, q):

    e = sum([(p[i] - q[i]) ** 2 for i in range(len(p))])
    return e
vecs = []
vecs.append([0]*100) # CLS
for vec in embedding_list:
    vecs.append(vec)
embed = torch.FloatTensor(vecs)
embed = torch.nn.Embedding.from_pretrained(embed)
del vecs,embedding_list,js_file,entity_file

class ERNIEDataset(Dataset):
    def __init__(self,*data):
        # for d in data:
            # print(d.size())
        self.data=torch.cat(data,dim=-1)

        self.size=len(data[0])
    def __getitem__(self,index):
        assert(len(self.data[index])==2561)
        return self.data[index]
    def __len__(self):
        return self.size

MAX_SEQ_LEN=512


WORD_CUTTING_MAX_PAIR=50
GUEESS_ATOM_MAX_PAIR=50
POS_NEG_MAX_PAIR=10

SAVE_THELD=.1

def swap_exmples(args, tokenizer,masked_lm_prob,max_predictions_per_seq,max_num_tokens,lines, idx, return_dict,
                 node2entity, entity_dict_copy):
    vocab=list(tokenizer.vocab.keys())
    examples=[]
    save_pre=max(int(SAVE_THELD*len(lines)),1)
    num=0
    file_names=[]
    for line in lines:
        # print(line)
        line,isAug=line.rstrip().rsplit('\t',1)
        isAug=bool(int(isAug))
        tokens_a=tokenizer.tokenize(line)
        insert_pos=len(tokens_a)//2
        offset=0
        while(tokens_a[insert_pos].startswith('##')):
            if(not tokens_a[insert_pos-offset].startswith('##')):
                insert_pos-=offset
                break
            elif(not tokens_a[insert_pos+offset].startswith('##')):
                insert_pos+=offset
                break
            offset+=1
            if(insert_pos-offset<0 or insert_pos+offset>=len(tokens_a)):
                print('Head Hard Found for sentence: {}'.format(tokens_a))
                break
        head=tokens_a[:insert_pos]
        tail=tokens_a[insert_pos:]
        if(len(tokens_a)>max_num_tokens):
            truncate_seq_pair(head,tail,max_num_tokens,rng)
        sop_label=0
        if(random()>.5):
            head,tail=tail,head
            sop_label=1
        if(head[0].startswith('##')): head[0]=head[0][2:]
        tokens_a=["[CLS]"] +head+["[SEP]"]+tail+ ["[SEP]"]
        
        tokens_a, masked_lm_positions, masked_lm_labels, entity_ids = create_wwm_lm_predictions(
                tokens_a, masked_lm_prob, max_predictions_per_seq, isAug,vocab,tokenizer)
        
        tokens=tokens_a
        segment_ids = [0 for _ in range(len(head)+2)]+ [ 1 for _ in range(len(tail)+1)]
        assert(len(segment_ids)==len(tokens) )

        # node2entity, entity_dict
        entity_mapping = node2entity
        entity_ids_mapping = []
        entity_ids_mapping_mask = []
        for index_id, entity_id in enumerate(entity_ids):
            # if index_id == 150:
            #     print()
            temp_entity_ids_mapping = []
            temp_entity_ids_mapping_mask = []
            if entity_id == -1:
                for i in range(args.two_hop_entity_num):
                    temp_entity_ids_mapping.append(-1)
                    temp_entity_ids_mapping_mask.append(0)
            else:
                if len(entity_mapping[entity_dict_index2str[entity_id]])  == 0:
                    for i in range(args.two_hop_entity_num):
                        temp_entity_ids_mapping.append(-1)
                        temp_entity_ids_mapping_mask.append(0)
                else:
                    if len(entity_mapping[entity_dict_index2str[entity_id]]) > args.two_hop_entity_num:
                        for two_hop_entity in reversed(node2entity[entity_dict_index2str[entity_id]]):
                            if len(temp_entity_ids_mapping) != args.two_hop_entity_num:
                                temp_entity_ids_mapping.append(entity_dict_copy[two_hop_entity])
                                temp_entity_ids_mapping_mask.append(1)
                            if len(temp_entity_ids_mapping) == args.two_hop_entity_num:
                                break
                    elif len(entity_mapping[entity_dict_index2str[entity_id]]) < args.two_hop_entity_num:
                        for two_hop_entity in node2entity[entity_dict_index2str[entity_id]]:
                            temp_entity_ids_mapping.append(entity_dict_copy[two_hop_entity])
                            temp_entity_ids_mapping_mask.append(1)
                        for i in range(args.two_hop_entity_num - len(entity_mapping[entity_dict_index2str[entity_id]])):
                            temp_entity_ids_mapping.append(-1)
                            temp_entity_ids_mapping_mask.append(0)
                        assert len(temp_entity_ids_mapping) == args.two_hop_entity_num
                    else:
                        for two_hop_entity in node2entity[entity_dict_index2str[entity_id]]:
                            temp_entity_ids_mapping.append(entity_dict_copy[two_hop_entity])
                            temp_entity_ids_mapping_mask.append(1)
            entity_ids_mapping.append(temp_entity_ids_mapping)
            entity_ids_mapping_mask.append(temp_entity_ids_mapping_mask)
       
        example = {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels,
            "entiy_ids": entity_ids,
            "entity_ids_mapping": entity_ids_mapping,
            "entity_ids_mapping_mask": entity_ids_mapping_mask,
            'sop_label':sop_label
        }
        examples.append(example)
        num+=1
        if(num%save_pre==0 or num==len(lines)):
            ran_num=str(rng.randint(0,100000))
            name='run_tmp/{}_{}_{}.pkls'.format(idx,num,ran_num)
            sf=open(name,'wb+')
            pickle.dump(examples,sf)
            sf.close()
            file_names.append(name)
            del name
            examples.clear()
            gc.collect()
    return_dict[idx]=file_names

logger = logging.getLogger(__name__)
rng=Rd(43)
import re
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        cp=ord(cp)
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
    def __init__(self, input_ids, input_mask, segment_ids, label_id, entiy_ids, entity_mapping, entity_mapping_mask,
                 sop_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entiy_ids=entiy_ids
        self.entity_mapping = entity_mapping
        self.entity_mapping_mask = entity_mapping_mask
        self.sop_label = sop_label

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    if rng.random() < 0.05: # I do not want you delete front because you cause the head always produce [UNK]
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq,vocab):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]" or not token.isalnum():
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    
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
    return token == "[CLS]" or token == "[SEP]" or (not token.isalnum() and len(token)==1)

def create_wwm_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq,isAug,vocab,tokenizer):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    
    chars=[token for token in tokens]
    cand_map=dict(zip(range(len(tokens)),range(len(tokens))))

    entity_pos=set()
    entity_ids=[] # This list would like [ [0] , [1,2] , [3,4,5]] to pack wwm word toghter
                   
    cand_index=0
    skip_index=set()
    
    assert(chars[0]=='[CLS]')
    
    while(cand_index<len(chars)):
        if(isSkipToken( chars[cand_index])):
            skip_index.add(len(entity_ids))
            entity_ids.append(-1)            
        elif( rng.random()< .5+(cand_index/len(chars))*.4 and ww_tree.startsWith(chars[cand_index])):
            c=ww_tree.get_lengest_match(chars,cand_index)
            if(c == None ):
                entity_ids.append(-1)
            else:
                word=''.join(chars[cand_index:c+1])
                assert(word in entity_dict)
                mask_count[word]-=1
                entity_pos.add(len(entity_ids))
                entity_ids.append(entity_dict[word])
                if(mask_count[word]==0):
                    mask_count.pop(word)
                    ww_tree.delete(word)
        else: entity_ids.append(-1)
        cand_index+=1

    cand_indices=[  i for i in range(cand_index) if i not in skip_index and i not in entity_pos]
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

    assert(len(mask_indices)<=MAX_SEQ_LEN)
    assert(len(masked_token_labels)==len(mask_indices))
    if(len(entity_ids)!=len(chars)):
        print(len(entity_ids))
        print(len(chars))
        print(entity_ids)
        print(chars)
        raise RuntimeError('Entity_ids and chars should have same length.')

    assert(len(entity_ids)==len(chars))
    assert(len(chars)<=MAX_SEQ_LEN)
    assert(len(set(mask_indices))==len(mask_indices))
    
    return chars, mask_indices, masked_token_labels,entity_ids


def create_examples(args, data_path, max_seq_length, masked_lm_prob, max_predictions_per_seq,tokenizer,node2entity,
                                entity_dict, min_len=128):
    """Creates examples for the training and dev sets."""
    examples = []
    max_num_tokens = max_seq_length - 3
    fr = open(data_path, "r",encoding='utf-8')
    http_remover=re.compile(r'https?://[a-zA-Z0-9.%+()/&:;<=●>?@^_`{|}~]*',re.S)
    max_total_example=len(fr.readlines())
    fr.seek(0)
    lines=[]
    jobs=[]
    idx=0
    join_unti=16
    batch_size=1+(max_total_example//join_unti)
    man=Manager()
    return_dict=man.dict()
    gc.collect()
    for (i, line) in tqdm(enumerate(fr), desc="Creating Example"):
        if i == 158200:
            break
        line,label=line.rsplit('\t',1)
        line=http_remover.sub("",line).replace('##',"").strip()
        count=0
        for w in line:
            if(is_chinese_char(w)): count+=1
            if(count>=min_len//2): break
        if(len(line)<=min_len or count<min_len//2): continue
        lines.append(line+'\t'+label)
        idx+=1
        if(idx%batch_size==0):
          
            job=Process(target=swap_exmples, args=(args,tokenizer,masked_lm_prob,max_predictions_per_seq,
                                                  max_num_tokens,lines,idx,return_dict, node2entity, entity_dict))
            jobs.append(job)
            job.start()
            lines.clear()
            if(idx%(batch_size*join_unti)==0):
                for job in jobs:
                    job.join()
                keys=list(return_dict.keys())
                for key in keys:
                    examples.extend(return_dict[key])
                    return_dict.pop(key)
                return_dict.clear()
    if(len(lines)):
        job=Process(target=swap_exmples,args=(args, tokenizer,masked_lm_prob,max_predictions_per_seq,max_num_tokens,lines,idx,return_dict, node2entity, entity_dict))
        jobs.append(job)
        job.start()
        lines.clear()
    for job in jobs:
        job.join()
    keys=list(return_dict.keys())
    for key in keys:
        examples.extend(return_dict[key])
        return_dict.pop(key)
    return_dict.clear()
    fr.close()
    result=[]
    for name in tqdm(examples,desc='Loading example from tmp file'):
        sf=open(name,'rb')
        tmp=pickle.load(sf)
        sf.close()
        result.extend(tmp)
        del tmp
    return result

def convert_examples_to_features(args, examples, max_seq_length, tokenizer,do_gc=False):
    features = []
    example_num=len(examples)
    names_list=[]
    save_pre_step=max(int( .25 * example_num ) ,1 )
    
    for f_index in tqdm(range(example_num), desc="Converting Feature"):

        example=examples[-1]
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
        sop_label=example['sop_label']

        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
        assert(len(masked_label_ids)==len(masked_lm_positions))
        input_array = np.zeros(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.bool)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.bool)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
        lm_label_array[masked_lm_positions] = masked_label_ids

        entity_array=np.full(max_seq_length, dtype=np.int, fill_value=-1)
        entity_array[:len(entiy_ids)]= entiy_ids
        
        
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
        if( ((f_index+1)%save_pre_step)==0 or (f_index+1) == example_num):
            print("Do Save There")
            name='run_tmp/{}_f.pklf'.format(f_index)
            sf=open(name,'wb+')
            pickle.dump(features,sf)
            sf.close()
            names_list.append(name)
            features.clear()
            del name
    del features
    features=[]
    examples=[]
    for name in tqdm(names_list,desc='Loading features'):
        sf=open(name,'rb')
        f=pickle.load(sf)
        sf.close()
        features.extend(f)
        del f
    return features

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + 2)
    rng.seed(seed + 4)

def evaluate(args,model,eval_dataloader1, eval_dataloader2, device,loss_bag, eval_step):
    torch.cuda.empty_cache()
    best_loss,epoch,tr_loss=loss_bag
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    for batch  in zip(eval_dataloader1, eval_dataloader2):
        logger.info('Testing step: {} / {}'.format(nb_eval_steps, eval_step))
        batch0 = tuple(t.to(device) for t in batch[0])
        each_token_condidate_entity = tuple(t.to(device) for t in batch[1])[0]
        each_token_condidate_entity_mask = tuple(t.to(device) for t in batch[1])[1]

        input_ids, input_mask, segment_ids, masked_lm_labels, input_ent, ent_mask, next_sentence_label, \
        ent_candidate, ent_labels = batch0
        with torch.no_grad():
            loss = model(input_ids, input_mask, segment_ids, masked_lm_labels, input_ent, ent_mask,
                         next_sentence_label, ent_candidate, ent_labels, each_token_condidate_entity,
                         each_token_condidate_entity_mask)
        eval_loss += loss.mean().item()
#                    sop_loss += loss[1].mean()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    if eval_loss < best_loss:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        best_loss = eval_loss
    logger.info("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n"% (epoch, tr_loss, eval_loss))
    torch.cuda.empty_cache()
    return min(best_loss, eval_loss)


def main():
    parser = argparse.ArgumentParser()
    

    parser.add_argument("--pretrain_train_path", type=str, default="./data_aug/aug_dev.txt", help="pretrain train path to file")
    parser.add_argument("--pretrain_dev_path", type=str, default="./data_aug/aug_dev.txt", help="pretrain dev path to file")
    parser.add_argument("--max_seq_length", type=int, default=512,help="max seq length of input sequences")

    parser.add_argument("--do_train", type=bool,default=True, help="If do train")
    parser.add_argument("--do_lower_case", type=bool,default=True, help="If do case lower")
    parser.add_argument("--train_batch_size", type=int, default=64, help="train_batch_size") # May Need to finetune
    parser.add_argument("--eval_batch_size", type=int, default=16, help="eval_batch_size")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="num_train_epochs")
    parser.add_argument("--learning_rate", type=float, default=8e-6, help="learning rate") # May Need to finetune
    parser.add_argument("--warmup_proportion", type=float, default=.05, help="warmup_proportion") # May Need to finetune
    parser.add_argument("--no_cuda", type=bool, default=False, help="prevent use GPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="If we are using cluster for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24, help="gradient_accumulation_steps") # May Need to finetune
    parser.add_argument("--fp16", type=bool, default=False, help="If use apex to train")
    parser.add_argument("--loss_scale", type=int, default=0, help="loss_scale")
    parser.add_argument("--bert_config_json", type=str, default="pytorch_pretrained_bert/bert_config.json", help="bert_config_json")
    parser.add_argument("--vocab_file", type=str, default="pytorch_pretrained_bert/vocab.txt", help="Path to vocab file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="output_dir")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="masked_lm_prob")
    parser.add_argument("--max_predictions_per_seq", type=int, default=72, help="max_predictions_per_seq")
    parser.add_argument("--cache_dir", type=str, default='pytorch_pretrained_bert', help="cache_dir")
    parser.add_argument("--model_name_or_path", type=str, default="pytorch_pretrained_bert", help="model_name_or_path")
    parser.add_argument('--eval_pre_setp',type=float,default=.1,help="The percent of how many train with one eval run")
    parser.add_argument('--finetune_proportion',type=float,default=.5,help="Detemind the proportion of the first training stage")
    parser.add_argument('--two_hop_entity_num', default=5, type=int, help='The threshold value of two hop entities of each entities in knowledge graph')
    parser.add_argument('--entity_type', default='./kg_embed/type_set.pkl', type=str, help='entity type in knowledge graph')
    parser.add_argument('--node2entity', default='./new_n2e.pkl', type=str, help='target node to other entity relationship')
    args=parser.parse_args()

    with open(args.entity_type, 'rb') as fo:
        entity_type_dict = pickle.load(fo, encoding='utf-8')
    with open(args.node2entity, 'rb') as fo:
        node2entity = pickle.load(fo, encoding='utf-8')

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
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        eval_examples = create_examples(args=args,
                                data_path=args.pretrain_dev_path,
                                max_seq_length=args.max_seq_length,
                                masked_lm_prob=args.masked_lm_prob,
                                max_predictions_per_seq=args.max_predictions_per_seq,
                                tokenizer=tokenizer,
                                node2entity=node2entity,
                                entity_dict=entity_dict)
        train_examples = create_examples(args=args,
                                         data_path=args.pretrain_train_path,
                                         max_seq_length=args.max_seq_length,
                                         masked_lm_prob=args.masked_lm_prob,
                                         max_predictions_per_seq=args.max_predictions_per_seq,
                                         tokenizer=tokenizer,
                                         node2entity=node2entity,
                                         entity_dict=entity_dict)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size/ args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
            
        model,missing_keys = ERNIEForPreTraining.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            cache_dir=args.cache_dir
        )
    
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    new_add_param = [ (n,p)  for n,p in param_optimizer if n in missing_keys ]
    pretrain_parm = [ (n,p)  for n,p in param_optimizer if n not in missing_keys ]
    
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
        
    global_step = 0
    best_loss = 100000
#    eval_pern_steps=570

    if args.do_train:
        eval_features = convert_examples_to_features(args,
            eval_examples, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_entity_ids = torch.tensor([f.entiy_ids for f in eval_features], dtype=torch.long)
        all_entity_mapping = torch.tensor([f.entity_mapping for f in eval_features], dtype=torch.long)
        all_entity_mapping_repre = embed(all_entity_mapping+1)
        all_entity_mapping_mask = torch.tensor([f.entity_mapping_mask for f in eval_features], dtype=torch.long)

        all_sop_labels = torch.tensor([f.sop_label for f in eval_features], dtype=torch.long).unsqueeze(-1)
        # eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_entity_ids,
        #                         all_sop_labels, all_entity_mapping)
        eval_data1 = ERNIEDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_entity_ids,
                                all_sop_labels)
        eval_data2 = TensorDataset(all_entity_mapping_repre, all_entity_mapping_mask)
        total_eval_step = int(all_input_ids.size(0) / args.eval_batch_size)
        del eval_features
        # gc.collect()
        train_features = convert_examples_to_features(args,
            train_examples, args.max_seq_length, tokenizer,do_gc=True)
        train_features_len = len(train_features)
        del train_examples
        gc.collect()
        print('Going to tranfer features to tensor')
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        for f in train_features: f.input_ids=None
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        for f in train_features: f.input_mask=None
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        for f in train_features: f.segment_ids=None
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        for f in train_features: f.label_id=None
        all_entity_ids = torch.tensor([f.entiy_ids for f in train_features], dtype=torch.long)
        for f in train_features: f.entiy_ids=None

        all_entity_mapping = torch.tensor([f.entity_mapping for f in train_features], dtype=torch.long)
        for f in train_features: f.entity_mapping = None
        all_entity_mapping_repre = embed(all_entity_mapping+1)

        all_entity_mapping_mask = torch.tensor([f.entity_mapping_mask for f in train_features], dtype=torch.long)
        for f in train_features: f.entity_mapping_mask = None

        all_sop_labels = torch.tensor([f.sop_label for f in train_features], dtype=torch.long).unsqueeze(-1)
        for f in train_features: f.sop_label=None

        print('Going to tranfer tensors to TensorDataset')
        del train_features
        gc.collect()

        train_data1 = ERNIEDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
            all_entity_ids, all_sop_labels)
        train_data2 = TensorDataset(all_entity_mapping_repre, all_entity_mapping_mask)
        
        if args.local_rank == -1:
            train_sampler1 = SequentialSampler(train_data1)
            train_sampler2 = SequentialSampler(train_data2)

        else:
            train_sampler1 = DistributedSampler(train_data1)
            train_sampler2 = SequentialSampler(train_data2)

        def collect_fn(x):
            x = torch.cat(tuple(xx.unsqueeze(0) for xx in x) , dim=0)
            entity_idx = x[:, 4*args.max_seq_length:5*args.max_seq_length]
            uniq_idx = np.unique(entity_idx.numpy())
            if(len(uniq_idx)<=0): raise RuntimeError('Find Zero candidate in the sentence')
            ent_candidate = embed(torch.LongTensor(uniq_idx+1))
            ent_candidate = ent_candidate.repeat([n_gpu, 1])
            d = {}
            dd = []
            for i, idx in enumerate(uniq_idx):
                d[idx] = i
                dd.append(idx)
            ent_size = len(uniq_idx)-1
            def map(x):
                if x == -1:
                    return -1
                else:
                    rnd = uniform(0, 1)
                    if rnd < 0.05:
                        return dd[randint(1, ent_size)]
                    elif rnd < 0.2:
                        return -1
                    else:
                        return x
            ent_labels = entity_idx.clone()
            d[-1] = -1
            ent_labels = ent_labels.apply_(lambda x: d[x])
            entity_idx.apply_(map)
            ent_emb = embed(entity_idx+1)
            mask = entity_idx.clone()
            mask.apply_(lambda x: 0 if x == -1 else 1)
            mask[:,0] = 1
            assert(len(ent_labels))

            return x[:,:args.max_seq_length], x[:,args.max_seq_length:2*args.max_seq_length], \
                   x[:,2*args.max_seq_length:3*args.max_seq_length], \
                   x[:,3*args.max_seq_length:4*args.max_seq_length], \
                   ent_emb, mask, x[:,5*args.max_seq_length:6*args.max_seq_length], \
                   ent_candidate, ent_labels

        train_dataloader1 = DataLoader(train_data1, sampler=train_sampler1, batch_size=args.train_batch_size,
                                      collate_fn=collect_fn, shuffle=False)
        train_dataloader2 = DataLoader(train_data2, sampler=train_sampler2, batch_size=args.train_batch_size, shuffle=False)
        model.train()
        # Run prediction for full data
        eval_sampler1 = SequentialSampler(eval_data1)
        eval_sampler2 = SequentialSampler(eval_data2)
        eval_dataloader1 = DataLoader(eval_data1, sampler=eval_sampler1, batch_size=args.eval_batch_size,
                                      collate_fn=collect_fn, shuffle=False)
        eval_dataloader2 = DataLoader(eval_data2, sampler=eval_sampler2, batch_size=args.eval_batch_size, shuffle=False)

        total_step=len(train_dataloader1)*args.num_train_epochs
        eval_step=int(total_step*args.eval_pre_setp)
        scheduler = get_double_linear_schedule_with_warmup(optimizer,total_step,args.warmup_proportion,args.finetune_proportion)

        logger.info("Start Train...")
        for e in trange(int(args.num_train_epochs), desc="Epoch"):
            nb_tr_examples = 0
            for step, batch in enumerate(zip(train_dataloader1, train_dataloader2)):
                batch0 = tuple(t.to(device) for t in batch[0])
                each_token_condidate_entity = tuple(t.to(device) for t in batch[1])[0]
                each_token_condidate_entity_mask = tuple(t.to(device) for t in batch[1])[1]

                input_ids, input_mask, segment_ids, masked_lm_labels, input_ent, ent_mask, next_sentence_label, \
                ent_candidate, ent_labels = batch0
                loss = model(input_ids, input_mask, segment_ids, masked_lm_labels, input_ent, ent_mask,
                             next_sentence_label, ent_candidate, ent_labels, each_token_condidate_entity,
                             each_token_condidate_entity_mask)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                nb_tr_examples += input_ids.size(0)
                # nb_tr_steps += 1
                logger.info("epoch: {}   Step: {} / {}   loss: {}".format(e, step,
                                                                int(train_features_len/args.train_batch_size), loss.item()))
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    del loss
                    scheduler.step() 
                    optimizer.zero_grad()
                    global_step += 1
                    if( global_step == int(total_step*args.finetune_proportion)):
                        optimizer.add_param_group(old_optimizer_grouped_parameters)

                if ((step+1) % 100) == 0:
                    best_loss=evaluate(args,model,eval_dataloader1, eval_dataloader2, device, (best_loss,e,loss / step),
                                       total_eval_step)
        logger.info('Training Done!')


if __name__ == "__main__":
    main()

