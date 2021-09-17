python run_re.py --pretrain_train_path ./datasets/dxyrc/dxyrc_train.pt   \
    --pretrain_dev_path  datasets/dxyrc/dxyrc_dev.pt \
    --num_train_epochs 3 --learning_rate 2e-5 --label_num 43 --eval_pre_step 0.26
python run_re.py --pretrain_train_path ./datasets/dxyrc/dxyrc_train.pt   \
    --pretrain_dev_path  datasets/dxyrc/dxyrc_dev.pt \
    --num_train_epochs 4 --learning_rate 2e-5 --label_num 43 --eval_pre_step 0.26

