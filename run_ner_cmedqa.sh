python run_ner.py --pretrain_train_path ./datasets/cmedqaner/train_new.txt   \
    --pretrain_dev_path  datasets/cmedqaner/test_new.txt \
    --task_name cmedqaner --train_batch_size 16\
    --num_train_epochs 8 --learning_rate 8e-5 --eval_pre_step .35 --model_name_or_path  bert_base
