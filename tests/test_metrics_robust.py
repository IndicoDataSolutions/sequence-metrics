import pytest

from sequence_metrics.metrics import (
    get_all_metrics,
    get_seq_quadrants_fn,
    seq_precision,
    seq_recall,
    sequence_f1,
    sequences_overlap,
)
from sequence_metrics.testing import extend_label, verify_all_metrics_structure


def exact_equality(a, b):
    return a["text"] == b["text"]


def all_combos(true, pred):
    classes = list(set([l["label"] for labels in true + pred for l in labels]))
    classes_with_missing_keys = set(
        [
            l["label"]
            for label in true + pred
            for l in label
            if "start" not in l or "end" not in l
        ]
    )

    for span_type, skips_missing in [
        ("token", True),
        ("overlap", True),
        ("exact", True),
        ("superset", True),
        ("value", False),
        (exact_equality, False),
    ]:
        for average in ["micro", "macro", "weighted"]:
            f1 = sequence_f1(true, pred, span_type=span_type, average=average)
            if len(classes_with_missing_keys) > 0 and skips_missing:
                assert f1 is None
            else:
                assert isinstance(f1, float)
        counts = get_seq_quadrants_fn(span_type)(true, pred)
        assert set(counts.keys()) == set(classes)
        f1_by_class = sequence_f1(true, pred, span_type=span_type)
        precision_by_class = seq_precision(true, pred, span_type=span_type)
        recall_by_class = seq_recall(true, pred, span_type=span_type)
        for cls_ in classes:
            f1 = f1_by_class[cls_]
            prec = precision_by_class[cls_]
            rec = recall_by_class[cls_]
            cls_counts = counts[cls_]

            if cls_ in classes_with_missing_keys and skips_missing:
                assert f1 is None
                assert prec is None
                assert rec is None
                assert cls_counts is None
            else:
                assert isinstance(f1, dict)
                assert isinstance(f1["f1-score"], float)
                assert isinstance(f1["support"], int)
                assert isinstance(prec, float)
                assert isinstance(rec, float)
                assert isinstance(cls_counts, dict)
                assert len(cls_counts.keys()) == 3
                for metric in ["false_positives", "true_positives", "false_negatives"]:
                    assert isinstance(
                        cls_counts[metric], list
                    )  # Inexplicably, this is a list


def test_empty_preds():
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."

    y_true = extend_label(
        text,
        [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        10,
    )

    y_pred = extend_label(
        text,
        [],
        10,
    )

    all_metrics = get_all_metrics(preds=y_pred, labels=y_true)
    verify_all_metrics_structure(all_metrics=all_metrics, classes=["entity", "date"])
    all_combos(y_true, y_pred)


def test_empty_labels():
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."

    y_true = extend_label(
        text,
        [],
        10,
    )

    y_pred = extend_label(
        text,
        [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        10,
    )

    all_metrics = get_all_metrics(preds=y_pred, labels=y_true)
    verify_all_metrics_structure(all_metrics=all_metrics, classes=["entity", "date"])
    all_combos(y_true, y_pred)


def test_all_empty():
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."

    empty = extend_label(
        text,
        [],
        10,
    )

    all_metrics = get_all_metrics(preds=empty, labels=empty)
    verify_all_metrics_structure(all_metrics=all_metrics, classes=[])
    all_combos(empty, empty)


def test_missing_start_end():
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."

    y_true = extend_label(
        text,
        [
            {"label": "entity", "text": "Pepsi Company"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        10,
    )

    all_metrics = get_all_metrics(preds=y_true, labels=y_true)
    verify_all_metrics_structure(
        all_metrics=all_metrics, classes=["entity", "date"], none_classes=["entity"]
    )
    all_combos(y_true, y_true)
