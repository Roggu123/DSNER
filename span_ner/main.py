import argparse
import os
import json

import torch

import sys
sys.path.insert(0, "./")

from transformers import AdamW, get_linear_schedule_with_warmup

from src.span_ner.utils import UnitAlphabet, LabelAlphabet
from src.span_ner.model import PhraseClassifier
from src.span_ner.misc import fix_random_seed
from src.span_ner.utils import corpus_to_iterator, Procedure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_dir", type=str, required=True)
    parser.add_argument("--check_dir", type=str, required=True)
    # parser.add_argument("--script_path", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=20220301)
    parser.add_argument("--epoch_num", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    parser.add_argument("--negative_rate", type=float, default=0.7)    # 5, 1.5， 0.5， 0.3, 0.1
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # add ----------------------------------------------------
    parser.add_argument("--cl_weight", type=float, default=0.5)   # 对比学习损失函数的权重

    ##########################################################

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=True), end="\n\n")

    fix_random_seed(args.random_state)
    lexical_vocab = UnitAlphabet(os.path.join(args.pretrained_model_dir, "vocab.txt"))
    label_vocab = LabelAlphabet()
    print("label_vocab: ", label_vocab)

    train_loader = corpus_to_iterator(
        os.path.join(args.data_dir, "train.json"),
        args.batch_size, True, label_vocab)
    dev_loader = corpus_to_iterator(os.path.join(args.data_dir, "dev.json"), args.batch_size, False)
    test_loader = corpus_to_iterator(os.path.join(args.data_dir, "test.json"), args.batch_size, False)

    print("label_vocab: ", label_vocab)
    json.dump(
        label_vocab._item_to_idx,
        open(os.path.join(args.data_dir, "label2idx.json"), "w", encoding="utf-8"),
        ensure_ascii=False
    )

    model = PhraseClassifier(lexical_vocab,
                             label_vocab,
                             args.hidden_dim,
                             args.dropout_rate,
                             args.negative_rate,
                             args.pretrained_model_dir,
                             args=args,
                             )

    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    all_parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_param = [{'params': [p for n, p in all_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                     {'params': [p for n, p in all_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    total_steps = int(len(train_loader) * (args.epoch_num + 1))
    optimizer = AdamW(
        grouped_param,
        lr=args.learning_rate,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    if not os.path.exists(args.check_dir):
        os.makedirs(args.check_dir)
    best_dev = 0.0
    # script_path = args.script_path
    # checkpoint_path = os.path.join(args.check_dir, "model.pt")

    for epoch_i in range(0, args.epoch_num + 1):
        # dev_f1, dev_time = Procedure.test(model, dev_loader)
        # print("(Epoch {:3d}) f1 score on dev set is {:.5f} using {:.3f} secs".format(epoch_i, dev_f1, dev_time))

        # test_f1, test_time, test_sents, test_preds, test_gts = Procedure.test(model, test_loader)
        # print("{{Epoch {:3d}}} f1 score on test set is {:.5f} using {:.3f} secs".format(epoch_i, test_f1, test_time))

        loss, train_time = Procedure.train(model, train_loader, optimizer, scheduler)
        print("[Epoch {:3d}] loss on train set is {:.5f} using {:.3f} secs".format(epoch_i, loss, train_time))

        # dev_f1, dev_time, _, _, _ = Procedure.test(model, dev_loader)
        # print("(Epoch {:3d}) f1 score on dev set is {:.5f} using {:.3f} secs".format(epoch_i, dev_f1, dev_time))

        test_f1, test_time, test_sents, test_preds, test_gts = Procedure.test(
            model, test_loader
        )
        print("{{Epoch {:3d}}} f1 score on test set is {:.5f} using {:.3f} secs".format(epoch_i, test_f1, test_time))

        if test_f1 > best_dev:
            best_dev = test_f1

            print("\n<Epoch {:3d}> save best model with score: {:.5f} in terms of test set".format(epoch_i, test_f1))

            # 保存模型
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save, os.path.join(args.check_dir, "pytorch_model.bin"))
            # tokenizer.save_pretrained(outputs_dir)
            torch.save(args, os.path.join(args.check_dir, "training_args.bin"))
            print("\nSaving model checkpoint to %s" % (args.check_dir))
            torch.save(optimizer.state_dict(), os.path.join(args.check_dir, "optimizer.pt"))

            # 保存预测结果
            f_out = open(os.path.join(args.check_dir, "test_predictions.jsonl"), "w", encoding="utf-8")
            for i, words in enumerate(test_sents):
                tags_1 = test_gts[i]
                tags_2 = test_preds[i]

                tokens = []
                pred_tags = []
                gt_tags = []
                for j, word in enumerate(words):
                    tag_1 = tags_1[j]
                    tag_2 = tags_2[j]

                    tokens.append(word)
                    gt_tags.append(tag_1)
                    pred_tags.append(tag_2)

                f_out.write(
                    json.dumps(
                        {
                            "tokens": tokens,
                            "pred_tags": pred_tags,
                            "gt_tags": gt_tags,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

        print(end="\n\n")
