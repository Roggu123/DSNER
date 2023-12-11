

## 模型介绍

我们的模型具有如下特点：
(1) label embedding 采用label embedding建模每类实体的综合语义信息
(2) 对比学习：unified contrastive learning，一种新颖的有监督场景的对比学习损失函数，有效的对齐语义空间
(3) 对不同的错误类型采用不同的对比损失权重(grouped)



## 代码运行

在GPU环境下，运行如下命令
```bash

CUDA_VISIBLE_DEVICES="0" nohup python -u src/span_ner/main.py --random_state 1111 --data_dir datasets/CMeEE-v2/ --check_dir experiments/outputs/bert_CMeEE-v2_0 --pretrained_model_dir resources/chinese_bert_wwm_ext --warmup_steps 400 --batch_size 16 --negative_rate 0.9 --epoch_num 50 --learning_rate 2e-5 > train_0.log &


```

对其他数据集，我们也采用类似命令即可进行训练和预测




