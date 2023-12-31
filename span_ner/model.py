import random

import numpy as np

import torch
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
from torch import nn

from transformers import BertPreTrainedModel, BertModel

from src.span_ner.misc import flat_list
from src.span_ner.misc import iterative_support, conflict_judge

from src.span_ner.utils import UnitAlphabet, LabelAlphabet


class BERT(nn.Module):

    def __init__(self, source_path):
        super(BERT, self).__init__()
        self._repr_model = BertModel.from_pretrained(source_path)

    @property
    def dimension(self):
        return 768

    def forward(self, var_h, attn_mask, starts):
        all_hidden = self._repr_model(var_h,
                                      attention_mask=attn_mask,)[0]

        batch_size, _, hidden_dim = all_hidden.size()
        _, unit_num = starts.size()
        positions = starts.unsqueeze(-1).expand(batch_size, unit_num, hidden_dim)
        return torch.gather(all_hidden, dim=-2, index=positions)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()

        self._activator = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=False),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim, output_dim, bias=False))
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, var_h):
        return self._activator(self._dropout(var_h))



class PhraseClassifier(nn.Module):

    def __init__(self,
                 lexical_vocab: UnitAlphabet,
                 label_vocab: LabelAlphabet,
                 hidden_dim: int,
                 dropout_rate: float,
                 neg_rate: float,
                 bert_path: str,
                 args=None
                 ):
        super(PhraseClassifier, self).__init__()

        self._lexical_vocab = lexical_vocab
        self._label_vocab = label_vocab
        self._neg_rate = neg_rate

        self._encoder = BERT(bert_path)

        self._classifier = MLP(
            self._encoder.dimension * 4,   # * 3， * 2
            hidden_dim,
            len(label_vocab),
            dropout_rate,

        )
        self._criterion = nn.NLLLoss()

        self.args = args

    def forward(self, var_h, **kwargs):

        con_repr = self._encoder(
            var_h,
            kwargs["mask_mat"],
            kwargs["starts"]
        )

        batch_size, token_num, hidden_dim = con_repr.shape

        ext_row = con_repr.unsqueeze(2).expand(batch_size, token_num, token_num, hidden_dim)
        ext_column = con_repr.unsqueeze(1).expand_as(ext_row)

        # span 特征
        table = torch.cat(
            [ext_row, ext_column, ext_row - ext_column, ext_row * ext_column],
            dim=-1
        )
        # 用于对比试验
        # table = torch.cat(
        #     [ext_row, ext_column],
        #     dim=-1
        # )
        # table = torch.cat(
        #     [ext_row, ext_column, ext_row - ext_column,],
        #     dim=-1
        # )
        # table = torch.cat(
        #     [ext_row, ext_column, ext_row * ext_column, ],
        #     dim=-1
        # )

        return self._classifier(table), table

    def _pre_process_input(self, utterances):
        lengths = [len(s) for s in utterances]
        max_len = max(lengths)
        pieces = iterative_support(self._lexical_vocab.tokenize, utterances)
        units, positions = [], []

        for tokens in pieces:
            units.append(flat_list(tokens))
            cum_list = np.cumsum([len(p) for p in tokens]).tolist()
            positions.append([0] + cum_list[:-1])

        sizes = [len(u) for u in units]
        max_size = max(sizes)
        cls_sign = self._lexical_vocab.CLS_SIGN
        sep_sign = self._lexical_vocab.SEP_SIGN
        pad_sign = self._lexical_vocab.PAD_SIGN
        pad_unit = [[cls_sign] + s + [sep_sign] + [pad_sign] * (max_size - len(s)) for s in units]
        starts = [[ln + 1 for ln in u] + [max_size + 1] * (max_len - len(u)) for u in positions]

        var_unit = torch.LongTensor([self._lexical_vocab.index(u) for u in pad_unit])
        attn_mask = torch.LongTensor([[1] * (lg + 2) + [0] * (max_size - lg) for lg in sizes])
        var_start = torch.LongTensor(starts)

        if torch.cuda.is_available():
            var_unit = var_unit.cuda()
            attn_mask = attn_mask.cuda()
            var_start = var_start.cuda()
        return var_unit, attn_mask, var_start, lengths

    def _pre_process_output(self, entities, lengths):
        positions, labels = [], []
        batch_size = len(entities)

        for utt_i in range(0, batch_size):
            for segment in entities[utt_i]:
                positions.append((utt_i, segment[0], segment[1]))
                labels.append(segment[2])

        for utt_i in range(0, batch_size):
            reject_set = [(e[0], e[1]) for e in entities[utt_i]]   # 实际标注的实体
            s_len = lengths[utt_i]  # 句子长度
            neg_num = int(s_len * self._neg_rate) + 1

            candies = flat_list([[(i, j) for j in range(i, s_len) if (i, j) not in reject_set] for i in range(s_len)])
            if len(candies) > 0:
                sample_num = min(neg_num, len(candies))
                assert sample_num > 0

                np.random.shuffle(candies)
                for i, j in candies[:sample_num]:
                    positions.append((utt_i, i, j))
                    labels.append("O")

        var_lbl = torch.LongTensor(iterative_support(self._label_vocab.index, labels))
        if torch.cuda.is_available():
            var_lbl = var_lbl.cuda()
        return positions, var_lbl

    def estimate(self, sentences, segments):

        var_sent, attn_mask, start_mat, lengths = self._pre_process_input(sentences)

        score_t, table_t = self(
            var_sent,
            mask_mat=attn_mask,
            starts=start_mat
        )
        # print("lengths: ", lengths)
        # print("score_t: ", score_t.shape)
        # print("table_t: ", table_t.shape)

        positions, targets = self._pre_process_output(segments, lengths)
        # print("segments: ", segments)
        # print("positions: ", positions)

        flat_s = torch.cat([score_t[[i], j, k] for i, j, k in positions], dim=0)
        loss_ = self._criterion(torch.log_softmax(flat_s, dim=-1), targets, )

        if random.uniform(0, 1) < 0.01:
            print("loss_: ", loss_)

        # print("targets: ", targets.shape)

        repre_s = torch.cat([table_t[[i], j, k] for i, j, k in positions], dim=0)
        # print("repre_s: ", repre_s.shape)

        # 基于对比学习的损失计算
        if self.args.cl_weight > 0:
            # loss_fct_contrast = NTXentLoss(
            loss_fct_contrast = SupConLoss(
                distance=DotProductSimilarity(),
            )
            embeddings = torch.cat(
                [
                    repre_s,
                    self.classifiers[i].output_layer_2.weight
                ]
            )
            labels = targets

            contra_loss_ = loss_fct_contrast(
                embeddings,
                labels
            )
            # print("contra_loss_: ", contra_loss_)

            loss_ = loss_ + contra_loss_ * self.args.cl_weight

        return loss_

    def inference(self, sentences):
        var_sent, attn_mask, starts, lengths = self._pre_process_input(sentences)
        log_items, _ = self(var_sent, mask_mat=attn_mask, starts=starts)

        score_t = torch.log_softmax(log_items, dim=-1)
        val_table, idx_table = torch.max(score_t, dim=-1)

        listing_it = idx_table.cpu().numpy().tolist()
        listing_vt = val_table.cpu().numpy().tolist()
        label_table = iterative_support(self._label_vocab.get, listing_it)
        # print(label_table)

        candidates = []
        for l_mat, v_mat, sent_l in zip(label_table, listing_vt, lengths):
            candidates.append([])
            for i in range(0, sent_l):
                for j in range(i, sent_l):
                    if l_mat[i][j] != "O":
                        candidates[-1].append((i, j, l_mat[i][j], v_mat[i][j]))

        entities = []
        for segments in candidates:
            ordered_seg = sorted(segments, key=lambda e: -e[-1])
            filter_list = []

            # 采用没有冲突的span作为预测
            for elem in ordered_seg:
                flag = False
                current = (elem[0], elem[1])
                for prior in filter_list:
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append((elem[0], elem[1], elem[2]))
            entities.append(sorted(filter_list, key=lambda e: e[0]))
        return entities


