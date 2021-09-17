python run_ner.py --pretrain_train_path ./datasets/dxy_clean_ner/train.txt   \
    --pretrain_dev_path  datasets/dxy_clean_ner/test.txt \
    --task_name dxy_clean_ner --learning_rate 6e-5 --num_train_epochs 2 --eval_pre_step .1
