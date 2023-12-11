

## span 分类器 + 对比学习



```bash

CUDA_VISIBLE_DEVICES="0" nohup python -u src/span_ner/main.py --random_state 1111 --data_dir datasets/CMeEE-v2/ --check_dir experiments/outputs/bert_CMeEE-v2_0 --pretrained_model_dir resources/chinese_bert_wwm_ext --warmup_steps 400 --batch_size 16 --negative_rate 0.9 --epoch_num 50 --learning_rate 2e-5 > train_0.log &




```


