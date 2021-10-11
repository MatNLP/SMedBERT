# SMedBERT
Source code and data for [SMedBERT: A Knowledge-Enhanced Pre-trained Language Model withStructured Semantics for Medical Text Mining](https://github.com/algoflow19/SMedBERT/blob/main/SMedBERT.pdf)

## News: Data Problems
Our code is almost ready for you. Due to the importance of commercial KG, we have to get permission from [DXY](https://portal.dxy.cn/) whether the private owned datasets and KGs are allowed to be public or not. For now, we only release our code for you to train SMedBERT model with your Chinese medical data. If they agree to release the data, we will release 5% of the entities in KG and its embedding trained by KGE algorithm for you to play around, and for now you have to use your own KG. Feel free to open issue if you have any question. 

## Reqirements
- Python 3.6
- PyTorch 1.6
- transformers==2.2.1
- numpy
- tqdm
- scikit-learn
- jieba

## Datasets:
- [CMedQANER](https://github.com/alibaba-research/ChineseBLUE)
- [CHIP2020](http://cips-chip.org.cn/2020/eval2)
- [IR](https://github.com/alibaba-research/ChineseBLUE)
- ...

We release datasets with "open" licenses in the hub and for other datasets you may have to acquire them by your own.

## Use your own KG (New).
Since the Authorization in DXY is very slow, and keeping people always waiting is embarrassing, I summarize the pre-training process as follow so you can use your own KG to play the model!
### KG format.
Our framework only need to use entities( with their corresponding alias and types) and relations bewteen them in your KG. In fact, the alias are only used to linking the spans in the input sentence to the enities in KG, and so when a custom entit-linker is available for your KG, the alias are not necessary.
### Step 1: Train TransR embedding with your data.
The entity, relation and transfer matrix weights are nessary to use our framework, as you see in [there](https://github.com/MatNLP/SMedBERT/blob/15808263a03930eef173b485b5330abfc575509c/run_pretraining_stream.py#L81-L87). I recommend to use [DGL-KE](https://github.com/awslabs/dgl-ke) to train the embedding since it is fast and scale to very large KG. 
### Step 2: Train the entities rank weights.
As we mention in the paper, for a entity in KG, it may has too many neighbours and we have to decide use which of them. We perform PageRank on the KG and the value for each entity(node) is used as weight as shown in [there](https://github.com/MatNLP/SMedBERT/blob/92141b7f4d2ec39cb56d28eebc3d13f84ebd9b56/run_pretraining_stream.py#L58-L60). You need to arrange it into the json foramt.
### Step 3: Prepare the entity2neighbours dict for quick search.
As we often need to use the neighbours of a linked entity, we decide to build a dict beforehand to avoid unnecessary computing. We need two files, 'ent2outRel.pkl' and 'ent2inRel.pkl' respectively for out and in directions relations. The format should be **ent_name -> \[(rel_name,ent_name), ..., (rel_name,ent_name)\]**.
### Step 4: Prepare the entity2type dict.
As we propose the hyper-attention that makes use of entity types knowledge, we need a dict to provide our model with such information. The format should be **ent_name -> type_val**,
the **type_val** could be type name or type id.
### Step 5: Prepare the name2id dict.
As shown in [there](https://github.com/MatNLP/SMedBERT/blob/92141b7f4d2ec39cb56d28eebc3d13f84ebd9b56/run_pretraining_stream.py#L46-L69), name2id files are needed to provide the mapping bewteen entities and their corrponding resouces. The format is obvious as you see.
### Step 6: Run Pre-training!!!
```
python -m torch.distributed.launch --nproc_per_node=4 run_pretraining_stream.py
```
Note that since the there are very large files need to be loaded into memory, the program may appear to freeze at first.


## Usage
- Download the Pre-trained Model (BaiduPan link: https://pan.baidu.com/s/1T0L6uv3JzY6dT3mcX_mghQ passwd: ea6f), and put it into the main folder.  
- Download the KG embedding: (BaiduPan link：https://pan.baidu.com/s/19V-M70TdndPCR50r5Z2OYQ  passwd：0000), and put it into the /kgs folder.  
- Example of how to run pre-training process.
```
CUDA_VISIBLE_DEVICES=0 ./run_pretraining_stream.py
```
- Example of how to run NER on CMedQANER dataset.
```
CUDA_VISIBLE_DEVICES=0 ./run_ner_cmedqa.sh
```
Note we force to only use single GPU for now.


## Citation
```
@inproceedings{zhang-etal-2021-smedbert,
    title = "{SM}ed{BERT}: A Knowledge-Enhanced Pre-trained Language Model with Structured Semantics for Medical Text Mining",
    author = "Zhang, Taolin  and
      Cai, Zerui  and
      Wang, Chengyu  and
      Qiu, Minghui  and
      Yang, Bite  and
      He, Xiaofeng",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.457",
    doi = "10.18653/v1/2021.acl-long.457",
    pages = "5882--5893"
}
```
