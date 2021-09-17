python -u run_seq_cls.py --train_batch_size 8  \
    --num_train_epochs 3 \
    --pretrain_train_path ./datasets/cMedQQ/train.txt   \
    --pretrain_dev_path  ./datasets/cMedQQ/dev.txt \
    --eval_batch_size 32 --train_batch_size 16 --learning_rate 1.5e-5