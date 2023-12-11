import json

from seqeval.metrics import classification_report

from src.span_ner.ner_format_utils import conll2doccano_json

if __name__ == "__main__":

    file_1 = "experiments/outputs/bert_CMeEE-v2_drop_0.9_0/test_predictions.jsonl"
    file_2 = "experiments/outputs/bert_CMeEE-v2_drop_0.9_select_0.9_0/test_predictions.jsonl"

    list_pred_samples_1 = open(file_1, "r", encoding="utf-8").readlines()
    # list_pred_samples_1 = [json.loads(w.strip()) for w in list_pred_samples_1]
    list_pred_samples_2 = open(file_2, "r", encoding="utf-8").readlines()
    # list_pred_samples_2 = [json.loads(w.strip()) for w in list_pred_samples_2]

    for idx, (pred_1, pred_2) in enumerate(zip(list_pred_samples_1, list_pred_samples_2)):
        pred_1 = json.loads(pred_1)
        pred_2 = json.loads(pred_2)

        pred_1_tokens = pred_1["tokens"]
        pred_1_pred_tags = pred_1["pred_tags"]
        pred_1_gt_tags = pred_1["gt_tags"]

        pred_2_tokens = pred_2["tokens"]
        pred_2_pred_tags = pred_2["pred_tags"]
        pred_2_gt_tags = pred_2["gt_tags"]

        assert pred_1_tokens == pred_2_tokens
        assert pred_1_gt_tags == pred_2_gt_tags

        pred_1_spans = conll2doccano_json(pred_1_tokens, pred_1_pred_tags)
        pred_2_spans = conll2doccano_json(pred_1_tokens, pred_2_pred_tags)
        gt_spans = conll2doccano_json(pred_1_tokens, pred_2_gt_tags)

        scores_1 = classification_report([pred_1_gt_tags], [pred_1_pred_tags], output_dict=True)
        scores_2 = classification_report([pred_1_gt_tags], [pred_2_pred_tags], output_dict=True)
        # print("scores_1: ", scores_1)
        # print("scores_2: ", scores_2)

        pred_1_f1 = scores_1["weighted avg"]["f1-score"]
        pred_2_f1 = scores_2["weighted avg"]["f1-score"]

        if pred_1_f1 - pred_2_f1 > 0.3:
            print("idx: ", idx)
            print("tokens: ", "".join(pred_1_tokens))
            print("pred_1_f1: ", pred_1_f1)
            print("pred_2_f1: ", pred_2_f1)
            print("pred_1_spans: ", pred_1_spans)
            print("pred_2_spans: ", pred_2_spans)
            print("gt_spans: ", gt_spans)
