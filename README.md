# SMedBERT
[SMedBERT: A Knowledge-Enhanced Pre-trained Language Model withStructured Semantics for Medical Text Mining](https://github.com/algoflow19/SMedBERT/blob/main/SMedBERT.pdf)


## Hi everyone!
Our code is almost ready for your, while we have to checkout with DXY if the current KG and private owned datasets are allowed to be public. For now, we only release 5% of the entities in KG for you to play around. Feel free to open issue if you have any question. 

## Datasets:
- [CMedQANER](https://github.com/alibaba-research/ChineseBLUE)
- [CHIP2020](http://cips-chip.org.cn/2020/eval2)
- [IR](https://github.com/alibaba-research/ChineseBLUE)
- ...

We have include datasets with "open" licenses in the hub and for other datasets you may have to acquire them by your own.

## Usage
Download the Pre-trained Model (BaiduPan 链接: https://pan.baidu.com/s/1T0L6uv3JzY6dT3mcX_mghQ 提取码: ea6f), and put it into the main folder.  
Download the KG embedding: (BaiduPan 链接：https://pan.baidu.com/s/19V-M70TdndPCR50r5Z2OYQ  提取码：0000), and put it into the /kgs folder.  
Below are examples of how to run NER on CMedQANER dataset.
```
CUDA_VISIBLE_DEVICES=0 ./run_ner_cmedqa.sh
```
Note we force to only use single GPU for now.


## Code dependency
- Python 3.6
- PyTorch 1.6
- transformers==2.2.1

## Cite
```
@article{zhangsmedbert,
  title={SMedBERT: A Knowledge-Enhanced Pre-trained Language Model with Structured Semantics for Medical Text Mining},
  author={Zhang, Taolin and Cai, Zerui and Wang, Chengyu and Qiu, Minghui and Yang, Bite and He, Xiaofeng}
}
```
