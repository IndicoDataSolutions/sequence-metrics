import copy
import functools
import re
import typing as t
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable

import numpy as np
import spacy
import tabulate
from sklearn.metrics import confusion_matrix

NLP = None

LabeledSpan = t.TypedDict(
    "LabeledSpan",
    {
        "start": int,
        "end": int,
        "label": str,
        # Doc-id allows us to compute bundle-level metrics. Use doc-id to
        # compare spans across documents. Can be any type other than NoneType
        # as long as we can compare equality with ==.
        "doc_id": t.Optional[t.Any],
        "text": str,
        # You can have any other fields, and they will be persisted to the quadrant outputs.
    },
)

SpanType = t.Literal["token", "overlap", "exact", "superset", "value"]
AverageType = t.Literal["micro", "macro", "weighted"]


def get_spacy():
    global NLP
    if NLP is None:
        NLP = spacy.load(
            "en_core_web_sm", disable=["parser", "tagger", "ner", "textcat"]
        )
        NLP.max_length = (
            800000000  # approximately one volume of the encyclopedia britannica.
        )
    return NLP


def _get_unique_classes(
    true: t.Sequence[LabeledSpan], predicted: t.Sequence[LabeledSpan]
) -> list[str]:
    true_and_pred = list(true) + list(predicted)
    return list({seq["label"] for seqs in true_and_pred for seq in seqs})


def _convert_to_token_list(
    annotations: t.Sequence[LabeledSpan], unique_classes: t.Optional[list[str]] = None
) -> list:
    nlp = get_spacy()
    tokens = []
    annotations = copy.deepcopy(annotations)

    for annotation in annotations:
        if unique_classes and annotation.get("label") not in unique_classes:
            continue
        start_idx = annotation.get("start")
        tokens.extend(
            [
                {
                    "start": start_idx + token.idx,
                    "end": start_idx + token.idx + len(token.text),
                    "text": token.text,
                    "label": annotation.get("label"),
                    "doc_id": annotation.get("doc_id"),
                }
                for token in nlp(annotation.get("text"))
            ]
        )

    return tokens


def sequence_labeling_token_confusion(
    text: t.Sequence[str],
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
) -> str:
    nlp = get_spacy()
    none_class = "<None>"
    unique_classes = _get_unique_classes(true, predicted)
    unique_classes.append(none_class)

    true_per_token_all = []
    pred_per_token_all = []

    for i, (text_i, true_list, pred_list) in enumerate(zip(text, true, predicted)):
        tokens = nlp(text_i)
        true_per_token = []
        pred_per_token = []
        for token in tokens:
            token_start_end = {"start": token.idx, "end": token.idx + len(token.text)}
            for true_i in true_list:
                if sequences_overlap(token_start_end, true_i):
                    true_per_token.append(true_i["label"])
                    break
            else:
                true_per_token.append(none_class)

            for pred_i in pred_list:
                if sequences_overlap(token_start_end, pred_i):
                    pred_per_token.append(pred_i["label"])
                    break
            else:
                pred_per_token.append(none_class)
        true_per_token_all.extend(true_per_token)
        pred_per_token_all.extend(pred_per_token)
    cm = confusion_matrix(
        y_true=true_per_token_all, y_pred=pred_per_token_all, labels=unique_classes
    )
    return tabulate.tabulate(
        [["Predicted\nTrue", *unique_classes]]
        + [[l, *r] for l, r in zip(unique_classes, cm)]
    )


def sequence_labeling_token_quadrants(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
) -> dict[str, t.Optional[dict[str, list[dict[str, LabeledSpan]]]]]:
    """
    Return FP, FN, and TP quadrants
    """

    unique_classes = _get_unique_classes(true, predicted)
    classes_to_skip = set(
        l["label"]
        for label in true + predicted
        for l in label
        if "start" not in l or "end" not in l
    )
    unique_classes = [c for c in unique_classes if c not in classes_to_skip]

    d = {
        cls_: {"false_positives": [], "false_negatives": [], "true_positives": []}
        for cls_ in unique_classes
    }
    for cls_ in classes_to_skip:
        d[cls_] = None

    for true_list, pred_list in zip(true, predicted):
        true_tokens = _convert_to_token_list(true_list, unique_classes=unique_classes)
        pred_tokens = _convert_to_token_list(pred_list, unique_classes=unique_classes)

        # correct + false negatives
        for true_token in true_tokens:
            for pred_token in pred_tokens:
                if (
                    pred_token["start"] == true_token["start"]
                    and pred_token["end"] == true_token["end"]
                    and pred_token["doc_id"] == true_token["doc_id"]
                ):
                    if pred_token["label"] == true_token["label"]:
                        d[true_token["label"]]["true_positives"].append(
                            {"true": true_token, "pred": pred_token}
                        )
                    else:
                        d[true_token["label"]]["false_negatives"].append(
                            {"true": true_token, "pred": None}
                        )
                        d[pred_token["label"]]["false_positives"].append(
                            {"true": None, "pred": pred_token}
                        )

                    break
            else:
                d[true_token["label"]]["false_negatives"].append(
                    {"true": true_token, "pred": None}
                )

        # false positives
        for pred_token in pred_tokens:
            for true_token in true_tokens:
                if (
                    pred_token["start"] == true_token["start"]
                    and pred_token["end"] == true_token["end"]
                    and pred_token["doc_id"] == true_token["doc_id"]
                ):
                    break
            else:
                d[pred_token["label"]]["false_positives"].append(
                    {"true": None, "pred": pred_token}
                )

    return d


def calc_recall(TP, FN):
    try:
        return TP / float(FN + TP)
    except ZeroDivisionError:
        return 0.0


def calc_precision(TP, FP):
    try:
        return TP / float(FP + TP)
    except ZeroDivisionError:
        return 0.0


def calc_f1(recall, precision):
    try:
        return 2 * (recall * precision) / (recall + precision)
    except ZeroDivisionError:
        return 0.0


def seq_recall(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
    span_type: SpanType | Callable = "token",
) -> dict[str, float | None]:
    quadrants_fn = get_seq_quadrants_fn(span_type)
    class_quadrants = quadrants_fn(true, predicted)
    results = {}
    for cls_, quadrants in class_quadrants.items():
        if quadrants is None:
            # Class is skipped due to missing start or end
            results[cls_] = None
            continue
        FN = len(quadrants["false_negatives"])
        TP = len(quadrants["true_positives"])
        results[cls_] = calc_recall(TP, FN)
    return results


def seq_precision(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
    span_type: SpanType | Callable = "token",
) -> dict[str, float | None]:
    quadrants_fn = get_seq_quadrants_fn(span_type)
    class_quadrants = quadrants_fn(true, predicted)
    results = {}
    for cls_, quadrants in class_quadrants.items():
        if quadrants is None:
            # Class is skipped due to missing start or end
            results[cls_] = None
            continue
        FP = len(quadrants["false_positives"])
        TP = len(quadrants["true_positives"])
        results[cls_] = calc_precision(TP, FP)
    return results


def micro_f1(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
    span_type: SpanType | Callable = "token",
) -> float | None:
    quadrants_fn = get_seq_quadrants_fn(span_type)
    class_quadrants = quadrants_fn(true, predicted)
    TP, FP, FN = 0, 0, 0
    for quadrants in class_quadrants.values():
        if quadrants is None:
            # Class is skipped due to missing start or end
            # We cannot calculate a micro_f1
            return None
        FN += len(quadrants["false_negatives"])
        TP += len(quadrants["true_positives"])
        FP += len(quadrants["false_positives"])
    recall = calc_recall(TP, FN)
    precision = calc_precision(TP, FP)
    return calc_f1(recall, precision)


def per_class_f1(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
    span_type: SpanType | Callable = "token",
) -> dict[str, float | None]:
    """
    F1-scores per class
    """
    quadrants_fn = get_seq_quadrants_fn(span_type)
    class_quadrants = quadrants_fn(true, predicted)
    results = OrderedDict()
    for cls_, quadrants in class_quadrants.items():
        if quadrants is None:
            # Class is skipped due to missing start or end
            results[cls_] = None
            continue
        results[cls_] = {}
        FP = len(quadrants["false_positives"])
        FN = len(quadrants["false_negatives"])
        TP = len(quadrants["true_positives"])
        recall = calc_recall(TP, FN)
        precision = calc_precision(TP, FP)
        results[cls_]["support"] = FN + TP
        results[cls_]["f1-score"] = calc_f1(recall, precision)
    return results


def sequence_f1(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
    span_type: SpanType | Callable = "token",
    average: AverageType | None = None,
) -> float | dict[str, float | None]:
    """
    If average = None, return per-class F1 scores, otherwise
    return the requested model-level score.
    """
    if average == "micro":
        return micro_f1(true, predicted, span_type)

    f1s_by_class = per_class_f1(true, predicted, span_type)
    if average is None:
        return f1s_by_class

    if any(v is None for v in f1s_by_class.values()):
        # Some classes are skipped due to missing start or end
        return None
    f1s = [value.get("f1-score") for key, value in f1s_by_class.items()]
    supports = [value.get("support") for key, value in f1s_by_class.items()]

    if average == "weighted":
        if sum(supports) == 0:
            return 0.0
        return np.average(np.array(f1s), weights=np.array(supports))
    if average == "macro":
        return np.average(f1s)
    raise ValueError(f"Unknown average: {average}")


def strip_whitespace(y: LabeledSpan) -> LabeledSpan:
    label_text = y["text"]
    lstripped = label_text.lstrip()
    new_start = y["start"] + (len(label_text) - len(lstripped))
    stripped = label_text.strip()
    return {
        "text": label_text.strip(),
        "start": new_start,
        "end": new_start + len(stripped),
        **{k: v for k, v in y.items() if k not in ["text", "start", "end"]},
    }


def _norm_text(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).lower()


def fuzzy_compare(x: LabeledSpan, y: LabeledSpan) -> bool:
    return _norm_text(x["text"]) == _norm_text(y["text"])


def sequence_labeling_token_precision(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
) -> dict[str, float | None]:
    """
    Token level precision
    """
    return seq_precision(true, predicted, span_type="token")


def sequence_labeling_token_recall(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
) -> dict[str, float | None]:
    """
    Token level recall
    """
    return seq_recall(true, predicted, span_type="token")


def sequence_labeling_micro_token_f1(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
) -> float | None:
    """
    Token level F1
    """
    return micro_f1(true, predicted, span_type="token")


def check_doc_id(
    metric_fn: Callable[[LabeledSpan, LabeledSpan], bool]
) -> Callable[[LabeledSpan, LabeledSpan], bool]:
    """
    A decorator that adds doc_id check to the comparison function.
    """

    @functools.wraps(metric_fn)
    def wrapper(true: LabeledSpan, predicted: LabeledSpan):
        true_doc_id = true.get("doc_id", None)
        predicted_doc_id = predicted.get("doc_id", None)
        if true_doc_id is not None or predicted_doc_id is not None:
            if true_doc_id is None:
                raise ValueError("Label value is missing doc_id")
            if predicted_doc_id is None:
                raise ValueError("Prediction value is missing doc_id")
            if true_doc_id != predicted_doc_id:
                return False
        return metric_fn(true, predicted)

    return wrapper


@check_doc_id
def sequences_overlap(x: LabeledSpan, y: LabeledSpan) -> bool:
    return x["start"] < y["end"] and y["start"] < x["end"]


@check_doc_id
def sequence_exact_match(true_seq: LabeledSpan, pred_seq: LabeledSpan) -> bool:
    """
    Boolean return value indicates whether or not seqs are exact match
    """
    true_seq = strip_whitespace(true_seq)
    pred_seq = strip_whitespace(pred_seq)
    return pred_seq["start"] == true_seq["start"] and pred_seq["end"] == true_seq["end"]


@check_doc_id
def sequence_superset(true_seq: LabeledSpan, pred_seq: LabeledSpan) -> bool:
    """
    Boolean return value indicates whether or predicted seq is a superset of target
    """
    true_seq = strip_whitespace(true_seq)
    pred_seq = strip_whitespace(pred_seq)
    return pred_seq["start"] <= true_seq["start"] and pred_seq["end"] >= true_seq["end"]


def single_class_single_example_quadrants(
    true: t.Sequence[LabeledSpan],
    predicted: t.Sequence[LabeledSpan],
    equality_fn: Callable[[LabeledSpan, LabeledSpan], bool],
) -> dict[str, list[dict[str, LabeledSpan]]]:
    """
    Return FP, FN, and TP quadrants for a single class
    """
    # Some of the equality_fn checks are redundant, so it's helpful if the equality_fn is cached
    quadrants = {"false_positives": [], "false_negatives": [], "true_positives": []}
    try:
        for true_annotation in true:
            for pred_annotation in predicted:
                if equality_fn(true_annotation, pred_annotation):
                    quadrants["true_positives"].append(
                        {"true": true_annotation, "pred": pred_annotation}
                    )
                    break
            else:
                quadrants["false_negatives"].append(
                    {"true": true_annotation, "pred": None}
                )

        for pred_annotation in predicted:
            for true_annotation in true:
                if equality_fn(true_annotation, pred_annotation):
                    break
            else:
                quadrants["false_positives"].append(
                    {"true": None, "pred": pred_annotation}
                )
    except KeyError:
        # Missing start or end
        return {"skip_class": True}
    return quadrants


def sequence_labeling_quadrants(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
    equality_fn: Callable[[LabeledSpan, LabeledSpan], bool],
    n_threads: int = 5,
) -> dict[str, list[dict[str, LabeledSpan]]]:
    """
    Return FP, FN, and TP quadrants
    """
    unique_classes = _get_unique_classes(true, predicted)

    d = {}
    future_to_cls = {}
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for cls_ in unique_classes:
            for true_annotations, predicted_annotations in zip(true, predicted):
                # Per example
                true_cls_annotations = [
                    annotation
                    for annotation in true_annotations
                    if annotation["label"] == cls_
                ]
                predicted_cls_annotations = [
                    annotation
                    for annotation in predicted_annotations
                    if annotation["label"] == cls_
                ]

                ex_quadrants_future = pool.submit(
                    single_class_single_example_quadrants,
                    true_cls_annotations,
                    predicted_cls_annotations,
                    equality_fn,
                )
                future_to_cls[ex_quadrants_future] = cls_

    for future, cls_ in future_to_cls.items():
        ex_quadrants = future.result()
        if ex_quadrants.get("skip_class", False) or cls_ in d and d[cls_] is None:
            # Class is skipped due to key error on equality function
            d[cls_] = None
            continue
        if cls_ not in d:
            d[cls_] = {
                "false_positives": [],
                "false_negatives": [],
                "true_positives": [],
            }
        for key, value in ex_quadrants.items():
            d[cls_][key].extend(value)

    return d


EQUALITY_FN_MAP: dict[SpanType, Callable[[LabeledSpan, LabeledSpan], bool]] = {
    "overlap": sequences_overlap,
    "exact": sequence_exact_match,
    "superset": sequence_superset,
    "value": fuzzy_compare,
}


SPAN_TYPE_FN_MAPPING = {
    "token": sequence_labeling_token_quadrants,
    "overlap": partial(sequence_labeling_quadrants, equality_fn=sequences_overlap),
    "exact": partial(sequence_labeling_quadrants, equality_fn=sequence_exact_match),
    "superset": partial(sequence_labeling_quadrants, equality_fn=sequence_superset),
    "value": partial(sequence_labeling_quadrants, equality_fn=fuzzy_compare),
}


def get_seq_quadrants_fn(span_type: SpanType | Callable = "token"):
    if isinstance(span_type, str):
        return SPAN_TYPE_FN_MAPPING[span_type]
    elif callable(span_type):
        # Interpret span_type as an equality function
        return partial(sequence_labeling_quadrants, equality_fn=span_type)

    raise ValueError(
        f"Invalid span_type: {span_type}.  Must either be a string or a callable."
    )


def sequence_labeling_overlap_precision(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
) -> dict[str, float | None]:
    """
    Sequence overlap precision
    """
    return seq_precision(true, predicted, span_type="overlap")


def sequence_labeling_overlap_recall(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
) -> dict[str, float | None]:
    """
    Sequence overlap recall
    """
    return seq_recall(true, predicted, span_type="overlap")


def sequence_labeling_overlap_micro_f1(
    true: t.Sequence[t.Sequence[LabeledSpan]],
    predicted: t.Sequence[t.Sequence[LabeledSpan]],
) -> float | None:
    """
    Sequence overlap micro F1
    """
    return micro_f1(true, predicted, span_type="overlap")


def annotation_report(
    y_true,
    y_pred: t.Sequence[t.Sequence[LabeledSpan]],
    labels: t.Optional[t.Sequence[str]] = None,
    target_names: t.Optional[t.Sequence[str]] = None,
    digits: int = 2,
    width: int = 20,
):
    # Adaptation of https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/metrics/classification.py#L1363
    token_precision = sequence_labeling_token_precision(y_true, y_pred)
    token_recall = sequence_labeling_token_recall(y_true, y_pred)
    overlap_precision = sequence_labeling_overlap_precision(y_true, y_pred)
    overlap_recall = sequence_labeling_overlap_recall(y_true, y_pred)

    count_dict = defaultdict(int)
    for annotation_seq in y_true:
        for annotation in annotation_seq:
            count_dict[annotation["label"]] += 1

    seqs = [
        token_precision,
        token_recall,
        overlap_precision,
        overlap_recall,
        dict(count_dict),
    ]
    labels = set(token_precision.keys()) | set(token_recall.keys())
    target_names = ["%s" % l for l in labels]
    counts = [count_dict.get(target_name, 0) for target_name in target_names]

    last_line_heading = "Weighted Summary"
    headers = [
        "token_precision",
        "token_recall",
        "overlap_precision",
        "overlap_recall",
        "support",
    ]
    head_fmt = "{:>{width}s} " + " {:>{width}}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>{width}.{digits}f}" * 4 + " {:>{width}}" "\n"
    seqs = [[seq.get(target_name, 0.0) for target_name in target_names] for seq in seqs]
    rows = zip(target_names, *seqs)
    for row in rows:
        report += row_fmt.format(*row, width=width, digits=digits)

    report += "\n"
    averages = [np.average(seq, weights=counts) for seq in seqs[:-1]] + [
        np.sum(seqs[-1])
    ]
    report += row_fmt.format(last_line_heading, *averages, width=width, digits=digits)
    return report


def get_spantype_metrics(
    span_type: SpanType,
    preds: t.Sequence[t.Sequence[LabeledSpan]],
    labels: t.Sequence[t.Sequence[LabeledSpan]],
    field_names: t.Sequence[str],
) -> dict[str, dict]:
    quadrants = get_seq_quadrants_fn(span_type)(labels, preds)
    precisions = seq_precision(labels, preds, span_type)
    recalls = seq_recall(labels, preds, span_type)
    per_class_f1s = sequence_f1(labels, preds, span_type)
    return {
        class_: (
            dict(
                f1=(per_class_f1s[class_] or {}).get("f1-score", None),
                recall=recalls[class_],
                precision=precisions[class_],
                false_positives=len(quadrants[class_]["false_positives"])
                if quadrants[class_] is not None
                else None,
                false_negatives=len(quadrants[class_]["false_negatives"])
                if quadrants[class_] is not None
                else None,
                true_positives=len(quadrants[class_]["true_positives"])
                if quadrants[class_] is not None
                else None,
            )
            if class_ in quadrants
            else dict(
                f1=0.0,
                recall=0.0,
                precision=0.0,
                false_positives=0,
                false_negatives=0,
                true_positives=0,
            )
        )
        for class_ in field_names
    }


def weighted_mean(value: t.Sequence[float], weights: t.Sequence[float]) -> float:
    if sum(weights) == 0.0:
        return 0.0
    return sum(v * w for v, w in zip(value, weights)) / sum(weights)


def mean(value: t.Sequence[float]) -> float:
    if sum(value) == 0:
        return 0.0
    return sum(value) / len(value)


def summary_metrics(
    metrics: dict[SpanType, dict[str, dict]]
) -> dict[SpanType, dict[str, float | None]]:
    summary = {}
    for span_type, span_metrics in metrics.items():
        span_type_summary = {}
        f1 = []
        precision = []
        recall = []
        weight = []
        TP = 0
        FP = 0
        FN = 0
        if any(cls_metrics["f1"] is None for cls_metrics in span_metrics.values()):
            summary[span_type] = None
            continue
        for cls_metrics in span_metrics.values():
            f1.append(cls_metrics["f1"])
            precision.append(cls_metrics["precision"])
            recall.append(cls_metrics["recall"])
            TP += cls_metrics["true_positives"]
            FP += cls_metrics["false_positives"]
            FN += cls_metrics["false_negatives"]
            weight.append(
                cls_metrics["true_positives"] + cls_metrics["false_negatives"]
            )
        span_type_summary["macro_f1"] = mean(f1)
        span_type_summary["macro_precision"] = mean(precision)
        span_type_summary["macro_recall"] = mean(recall)

        span_type_summary["micro_precision"] = calc_precision(TP, FP)
        span_type_summary["micro_recall"] = calc_recall(TP, FN)
        span_type_summary["micro_f1"] = calc_f1(
            span_type_summary["micro_recall"], span_type_summary["micro_precision"]
        )

        span_type_summary["weighted_f1"] = weighted_mean(f1, weight)
        span_type_summary["weighted_precision"] = weighted_mean(precision, weight)
        span_type_summary["weighted_recall"] = weighted_mean(recall, weight)
        summary[span_type] = span_type_summary

    return summary


def get_all_metrics(
    preds: t.Sequence[t.Sequence[LabeledSpan]],
    labels: t.Sequence[t.Sequence[LabeledSpan]],
    field_names: t.Optional[t.Sequence[str]] = None,
) -> dict[str, dict]:
    if field_names is None:
        field_names = sorted(set(l["label"] for li in (labels + preds) for l in li))
    detailed_metrics: dict[SpanType, dict[str, dict]] = {}
    for span_type in ["token", "overlap", "exact", "superset", "value"]:
        detailed_metrics[span_type] = get_spantype_metrics(
            span_type=span_type, preds=preds, labels=labels, field_names=field_names
        )
    return {
        "summary_metrics": summary_metrics(detailed_metrics),
        "class_metrics": detailed_metrics,
    }
