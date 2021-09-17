python -Ou run_IR.py --train_batch_size 8  \
    --num_train_epochs 3 \
    --pretrain_train_path ./datasets/IR/reduce_train.txt   \
    --pretrain_dev_path  ./datasets/IR/reduce_dev.txt \
    --eval_batch_size 100