import pytest

from sequence_metrics.metrics import (
    get_all_metrics,
    get_seq_count_fn,
    seq_precision,
    seq_recall,
    sequence_f1,
    sequences_overlap,
)


def insert_text(docs, labels):
    if len(docs) != len(labels):
        raise ValueError("Number of documents must be equal to the number of labels")
    for doc, label in zip(docs, labels):
        for l in label:
            l["text"] = doc[l["start"] : l["end"]]
    return labels


def extend_label(text, label, amt):
    return insert_text([text for _ in range(amt)], [label for _ in range(amt)])


def remove_label(recs, label):
    return [[pred for pred in rec if not pred.get("label") == label] for rec in recs]


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (
            {"start": 0, "end": 1},
            {"start": 2, "end": 3},
            False,
        ),  # Non-overlapping
        (
            {"start": 0, "end": 1},
            {"start": 1, "end": 2},
            False,
        ),  # Flush against each other, True expected rather than false for frontend
        (
            {"start": 0, "end": 2},
            {"start": 1, "end": 3},
            True,
        ),  # Overlapping
        (
            {"start": 0, "end": 2},
            {"start": 1, "end": 2},
            True,
        ),  # Contained but flush against end
        (
            {"start": 0, "end": 3},
            {"start": 1, "end": 2},
            True,
        ),  # Full contained
        (
            {"start": 0, "end": 1},
            {"start": 0, "end": 2},
            True,
        ),  # Starts same, first label contained by second
        (
            {"start": 0, "end": 1},
            {"start": 0, "end": 1},
            True,
        ),  # Identical start / end
        ({"start": 0, "end": 0}, {"start": 0, "end": 0}, False),
    ],
)
def test_overlap(a, b, expected):
    assert sequences_overlap(a, b) == sequences_overlap(b, a) == expected


def check_metrics(Y, Y_pred, expected, span_type=None):
    counts = get_seq_count_fn(span_type)(Y, Y_pred)
    precisions = seq_precision(Y, Y_pred, span_type=span_type)
    recalls = seq_recall(Y, Y_pred, span_type=span_type)
    micro_f1_score = sequence_f1(Y, Y_pred, span_type=span_type, average="micro")
    per_class_f1s = sequence_f1(Y, Y_pred, span_type=span_type)
    weighted_f1 = sequence_f1(Y, Y_pred, span_type=span_type, average="weighted")
    macro_f1 = sequence_f1(Y, Y_pred, span_type=span_type, average="macro")
    for cls_ in counts:
        for metric in counts[cls_]:
            assert len(counts[cls_][metric]) == expected[cls_][metric]
        assert recalls[cls_] == pytest.approx(expected[cls_]["recall"], abs=1e-3)
        assert per_class_f1s[cls_]["f1-score"] == pytest.approx(
            expected[cls_]["f1-score"], abs=1e-3
        )
        assert precisions[cls_] == pytest.approx(expected[cls_]["precision"], abs=1e-3)

    assert micro_f1_score == pytest.approx(expected["micro_f1"], abs=1e-3)
    assert weighted_f1 == pytest.approx(expected["weighted_f1"], abs=1e-3)
    assert macro_f1 == pytest.approx(expected["macro_f1"], abs=1e-3)

    all_metrics = get_all_metrics(preds=Y_pred, labels=Y)
    cls_metrics = all_metrics["class_metrics"][span_type]
    for cls_, metrics in cls_metrics.items():
        assert metrics["false_positives"] == expected[cls_]["false_positives"]
        assert metrics["false_negatives"] == expected[cls_]["false_negatives"]
        assert metrics["true_positives"] == expected[cls_]["true_positives"]

        assert metrics["recall"] == pytest.approx(expected[cls_]["recall"], abs=1e-3)
        assert metrics["f1"] == pytest.approx(expected[cls_]["f1-score"], abs=1e-3)
        assert precisions[cls_] == pytest.approx(metrics["precision"], abs=1e-3)
    summary_metrics = all_metrics["summary_metrics"][span_type]
    assert summary_metrics["micro_f1"] == pytest.approx(expected["micro_f1"], abs=1e-3)
    assert summary_metrics["weighted_f1"] == pytest.approx(
        expected["weighted_f1"], abs=1e-3
    )
    assert summary_metrics["macro_f1"] == pytest.approx(expected["macro_f1"], abs=1e-3)
    assert summary_metrics["micro_recall"] == pytest.approx(
        expected["micro_recall"], abs=1e-3
    )
    assert summary_metrics["weighted_recall"] == pytest.approx(
        expected["weighted_recall"], abs=1e-3
    )
    assert summary_metrics["macro_recall"] == pytest.approx(
        expected["macro_recall"], abs=1e-3
    )
    assert summary_metrics["micro_precision"] == pytest.approx(
        expected["micro_precision"], abs=1e-3
    )
    assert summary_metrics["weighted_precision"] == pytest.approx(
        expected["weighted_precision"], abs=1e-3
    )
    assert summary_metrics["macro_precision"] == pytest.approx(
        expected["macro_precision"], abs=1e-3
    )


@pytest.mark.parametrize("span_type", ["overlap", "exact", "superset", "value"])
def test_incorrect(span_type):
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."
    expected = {
        "entity": {
            "false_positives": 10 if span_type == "token" else 10,
            "false_negatives": 20 if span_type == "token" else 10,
            "true_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
        },
        "date": {
            "false_positives": 10 if span_type == "token" else 10,
            "false_negatives": 40 if span_type == "token" else 10,
            "true_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
        },
        "micro_f1": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "micro_precision": 0.0,
        "macro_precision": 0.0,
        "weighted_precision": 0.0,
        "micro_recall": 0.0,
        "macro_recall": 0.0,
        "weighted_recall": 0.0,
    }
    y_true = extend_label(
        text,
        [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        10,
    )
    y_false_pos = extend_label(
        text,
        [
            {"start": 21, "end": 28, "label": "entity"},
            {"start": 62, "end": 65, "label": "date"},
        ],
        10,
    )
    check_metrics(y_true, y_false_pos, expected, span_type=span_type)


def test_token_correct():
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."
    expected = {
        "entity": {
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 20,
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
        },
        "date": {
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 40,
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
        },
        "micro_f1": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "micro_precision": 1.0,
        "macro_precision": 1.0,
        "weighted_precision": 1.0,
        "micro_recall": 1.0,
        "macro_recall": 1.0,
        "weighted_recall": 1.0,
    }
    y_true = extend_label(
        text,
        [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        10,
    )

    check_metrics(y_true, y_true, expected, span_type="token")


def test_token_mixed():
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."

    Y_mixed = extend_label(
        text,
        [
            {"start": 21, "end": 28, "label": "entity"},
            {"start": 62, "end": 65, "label": "date"},
        ],
        5,
    ) + extend_label(
        text,
        [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        5,
    )
    y_true = extend_label(
        text,
        [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        10,
    )
    expected = {
        "entity": {
            "false_positives": 5,
            "false_negatives": 10,
            "true_positives": 10,
            "precision": 0.66666,
            "recall": 0.5,
            "f1-score": 0.571,
        },
        "date": {
            "false_positives": 5,
            "false_negatives": 20,
            "true_positives": 20,
            "precision": 0.8,
            "recall": 0.5,
            "f1-score": 0.6153,
        },
        "micro_f1": 0.6,
        "macro_f1": 0.593,
        "weighted_f1": 0.601,
        "micro_precision": 0.75,
        "macro_precision": 0.7333,
        "weighted_precision": 0.7555,
        "micro_recall": 0.5,
        "macro_recall": 0.5,
        "weighted_recall": 0.5,
    }
    check_metrics(
        y_true,
        Y_mixed,
        expected,
        span_type="token",
    )


def test_token_mixed_2():
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."
    y_mixed = (
        extend_label(
            text,
            [
                {"start": 21, "end": 28, "label": "entity"},
                {"start": 62, "end": 65, "label": "date"},
            ],
            5,
        )
        + extend_label(
            text,
            [
                {"start": 7, "end": 20, "label": "entity"},
                {"start": 41, "end": 54, "label": "date"},
            ],
            2,
        )
        + extend_label(text, [{"start": 7, "end": 20, "label": "entity"}], 3)
    )
    y_true = extend_label(
        text,
        [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        10,
    )
    expected = {
        "entity": {
            "false_positives": 5,
            "false_negatives": 10,
            "true_positives": 10,
            "precision": 0.66666,
            "recall": 0.5,
            "f1-score": 0.571,
        },
        "date": {
            "false_positives": 5,
            "false_negatives": 32,
            "true_positives": 8,
            "precision": 0.615,
            "recall": 0.2,
            "f1-score": 0.302,
        },
        "micro_f1": 0.409,
        "macro_f1": 0.437,
        "weighted_f1": 0.392,
        "micro_precision": 0.642,
        "macro_precision": 0.641,
        "weighted_precision": 0.632,
        "micro_recall": 0.3,
        "macro_recall": 0.35,
        "weighted_recall": 0.3,
    }
    check_metrics(
        y_true,
        y_mixed,
        expected,
        span_type="token",
    )


@pytest.mark.parametrize(
    "overlapping",
    [
        [
            {"start": 5, "end": 16, "label": "entity"},
            {"start": 34, "end": 50, "label": "date"},
        ],
        [
            {"start": 6, "end": 10, "label": "entity"},
            {"start": 15, "end": 23, "label": "entity"},
            {"start": 34, "end": 50, "label": "date"},
        ],
        [
            {"start": 6, "end": 21, "label": "entity"},
            {"start": 38, "end": 60, "label": "date"},
        ],
    ],
)
def test_seq_mixed_overlap(overlapping):
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."
    expected = {
        "entity": {
            "false_positives": 4,
            "false_negatives": 4,
            "true_positives": 6,
            "precision": 0.6,
            "recall": 0.6,
            "f1-score": 0.6,
        },
        "date": {
            "false_positives": 4,
            "false_negatives": 4,
            "true_positives": 6,
            "precision": 0.6,
            "recall": 0.6,
            "f1-score": 0.6,
        },
        "micro_f1": 0.6,
        "macro_f1": 0.6,
        "weighted_f1": 0.6,
        "micro_precision": 0.6,
        "macro_precision": 0.6,
        "weighted_precision": 0.6,
        "micro_recall": 0.6,
        "macro_recall": 0.6,
        "weighted_recall": 0.6,
    }
    y_true = extend_label(
        text,
        [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        10,
    )
    y_mixed = extend_label(
        text,
        [
            {"start": 21, "end": 28, "label": "entity"},
            {"start": 62, "end": 65, "label": "date"},
        ],
        4,
    ) + extend_label(text, overlapping, 6)
    check_metrics(y_true, y_mixed, expected=expected, span_type="overlap")


@pytest.mark.parametrize(
    "span_type,true_positive_non_exact",
    [
        (
            "overlap",
            [
                {"start": 5, "end": 16, "label": "entity"},
                {"start": 34, "end": 50, "label": "date"},
            ],
        ),
        (
            "exact",
            [
                {"start": 7, "end": 20, "label": "entity"},
                {"start": 41, "end": 54, "label": "date"},
            ],
        ),
        (
            "superset",
            [
                {"start": 6, "end": 21, "label": "entity"},
                {"start": 38, "end": 60, "label": "date"},
            ],
        ),
        (
            "value",
            [
                {"start": 7, "end": 20, "label": "entity"},
                {"start": 41, "end": 54, "label": "date"},
            ],
        ),
        (
            "value",
            [
                {"start": 6, "end": 21, "label": "entity"},
                {"start": 40, "end": 54, "label": "date"},
            ],
        ),
    ],
)
def test_mixed_overlap(span_type, true_positive_non_exact):
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."
    y_true = [
        {"start": 7, "end": 20, "label": "entity"},
        {"start": 41, "end": 54, "label": "date"},
    ]
    y_false_pos = [
        {"start": 21, "end": 28, "label": "entity"},
        {"start": 62, "end": 65, "label": "date"},
    ]
    y_false_neg = [
        {"start": 7, "end": 20, "label": "entity"},
    ]

    mixed_false_negs = (
        extend_label(text, y_false_pos, 4)
        + extend_label(text, true_positive_non_exact, 3)
        + extend_label(text, y_false_neg, 3)
    )
    expected = {
        "entity": {
            "false_positives": 4,
            "false_negatives": 4,
            "true_positives": 6,
            "precision": 0.6,
            "recall": 0.6,
            "f1-score": 0.6,
        },
        "date": {
            "false_positives": 4,
            "false_negatives": 7,
            "true_positives": 3,
            "precision": 0.42857,
            "recall": 0.3,
            "f1-score": 0.353,
        },
        "micro_f1": 0.4864,
        "macro_f1": 0.476,
        "weighted_f1": 0.476,
        "micro_precision": 0.52941,
        "macro_precision": 0.51428,
        "weighted_precision": 0.51428,
        "micro_recall": 0.45,
        "macro_recall": 0.45,
        "weighted_recall": 0.45,
    }
    check_metrics(
        extend_label(text, y_true, 10),
        mixed_false_negs,
        expected=expected,
        span_type=span_type,
    )


def test_overlapping_2_class():
    x = "a and b"
    y_true = [{"start": 0, "end": 7, "text": x, "label": "class1"}]
    y_pred = [
        {"start": 0, "end": 1, "text": "a", "label": "class2"},
        {"start": 6, "end": 7, "text": "b", "label": "class1"},
    ]
    expected = {
        "class1": {
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
        },
        "class2": {
            "false_positives": 1,
            "false_negatives": 0,
            "true_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
        },
        "micro_f1": 0.66666,  # Calculated as the harmonic mean of Recall = 1, Precision = 0.5
        "macro_f1": 0.5,
        "weighted_f1": 1.0,  # because there is no support for class2
        "micro_precision": 0.5,
        "macro_precision": 0.5,
        "weighted_precision": 1.0,
        "micro_recall": 1.0,
        "macro_recall": 0.5,
        "weighted_recall": 1.0,
    }
    check_metrics(
        [y_true],
        [y_pred],
        expected=expected,
        span_type="overlap",
    )


def test_overlapping_2_class_swapped():
    x = "a and b"
    y_true = [{"start": 0, "end": 7, "text": x, "label": "class1"}]
    y_pred = [
        {"start": 0, "end": 1, "text": "a", "label": "class1"},
        {"start": 6, "end": 7, "text": "b", "label": "class2"},
    ]
    expected = {
        "class1": {
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
        },
        "class2": {
            "false_positives": 1,
            "false_negatives": 0,
            "true_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
        },
        "micro_f1": 0.66666,  # Calculated as the harmonic mean of Recall = 1, Precision = 0.5
        "macro_f1": 0.5,
        "weighted_f1": 1.0,  # because there is no support for class2
        "micro_precision": 0.5,
        "macro_precision": 0.5,
        "weighted_precision": 1.0,
        "micro_recall": 1.0,
        "macro_recall": 0.5,
        "weighted_recall": 1.0,
    }
    check_metrics(
        [y_true],
        [y_pred],
        expected=expected,
        span_type="overlap",
    )


def test_overlapping_1_class():
    x = "a and b"
    y_true = [{"start": 0, "end": 7, "text": x, "label": "class1"}]
    y_pred = [
        {"start": 0, "end": 1, "text": "a", "label": "class1"},
        {"start": 6, "end": 7, "text": "b", "label": "class1"},
    ]
    expected = {
        "class1": {
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
        },
        "micro_f1": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "micro_precision": 1.0,
        "macro_precision": 1.0,
        "weighted_precision": 1.0,
        "micro_recall": 1.0,
        "macro_recall": 1.0,
        "weighted_recall": 1.0,
    }
    check_metrics(
        [y_true],
        [y_pred],
        expected=expected,
        span_type="overlap",
    )


def test_2_class():
    x = "a and b"
    y_true = [
        {"start": 0, "end": 1, "text": "a", "label": "class1"},
        {"start": 6, "end": 7, "text": "b", "label": "class1"},
    ]
    y_pred = [
        {"start": 0, "end": 7, "text": x, "label": "class1"},
    ]
    expected = {
        "class1": {
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 2,
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
        },
        "micro_f1": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "micro_precision": 1.0,
        "macro_precision": 1.0,
        "weighted_precision": 1.0,
        "micro_recall": 1.0,
        "macro_recall": 1.0,
        "weighted_recall": 1.0,
    }
    for span_type in ["overlap", "superset"]:
        check_metrics(
            [y_true],
            [y_pred],
            expected=expected,
            span_type=span_type,
        )


@pytest.mark.parametrize(
    "span_type", ["overlap", "exact", "superset", "value", "token"]
)
def test_whitespace(span_type):
    x = "a and b"
    y_true = [{"start": 0, "end": 7, "text": x, "label": "class1"}]
    y_pred = [
        {"start": 0, "end": 8, "text": x + " ", "label": "class1"},
    ]
    expected = {
        "class1": {
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 3 if span_type == "token" else 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
        },
        "micro_f1": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "micro_precision": 1.0,
        "macro_precision": 1.0,
        "weighted_precision": 1.0,
        "micro_recall": 1.0,
        "macro_recall": 1.0,
        "weighted_recall": 1.0,
    }
    check_metrics(
        [y_true],
        [y_pred],
        expected=expected,
        span_type=span_type,
    )


def test_class_filtering_get_all_metrics():
    y_true = [{"start": 0, "end": 7, "text": "a and b", "label": "class1"}]
    y_pred = [
        {"start": 0, "end": 7, "text": "a and b", "label": "class1"},
        {"start": 8, "end": 17, "text": "b", "label": "class2"},
    ]
    all_metrics = get_all_metrics(
        preds=[y_pred], labels=[y_true], field_names=["class1"]
    )
    for span_type, metrics in all_metrics["class_metrics"].items():
        assert len(metrics.keys()) == 1
        summary_metrics = all_metrics["summary_metrics"][span_type]
        assert all(
            [sm == 1.0 for sm in summary_metrics.values()]
        ), summary_metrics  # All 1 because we have only one class.


@pytest.mark.parametrize(
    "pred",
    [
        {
            "start": 10,
            "end": 16,
            "text": "friday",
            "label": "label1",
        },
        {
            "start": 10,
            "end": 17,
            "text": "friday ",
            "label": "label1",
        },
        {
            "start": 10,
            "end": 17,
            "text": "Fri-day",
            "label": "label1",
        },
    ],
)
def test_value_metrics(pred):
    y_true = [
        {
            "start": 5,
            "end": 11,
            "text": "Friday",
            "label": "label1",
        }
    ]
    y_pred = [pred]
    expected = {
        "label1": {
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1-score": 1.0,
        },
        "micro_f1": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "micro_precision": 1.0,
        "macro_precision": 1.0,
        "weighted_precision": 1.0,
        "micro_recall": 1.0,
        "macro_recall": 1.0,
        "weighted_recall": 1.0,
    }
    check_metrics(
        [y_true],
        [y_pred],
        expected=expected,
        span_type="value",
    )


def test_value_metrics_multi():
    y_true = [
        {
            "start": 5,
            "end": 11,
            "text": "Friday",
            "label": "label1",
        },
        {
            "start": 16,
            "end": 21,
            "text": "Friday",
            "label": "label1",
        },
        {
            "start": 35,
            "end": 41,
            "text": "wednesday",
            "label": "label1",
        },
    ]

    y_pred = [
        {
            "start": 5,
            "end": 11,
            "text": "Friday",
            "label": "label1",
        },
        {
            "start": 16,
            "end": 21,
            "text": "friday",
            "label": "label1",
        },
        {
            "start": 26,
            "end": 32,
            "text": "friday.",
            "label": "label1",
        },
        {
            "start": 26,
            "end": 32,
            "text": "thursday",
            "label": "label1",
        },
    ]
    expected = {
        "label1": {
            "false_positives": 1,
            "false_negatives": 1,
            "true_positives": 1,
            "precision": 0.5,
            "recall": 0.5,
            "f1-score": 0.5,
        },
        "micro_f1": 0.5,
        "macro_f1": 0.5,
        "weighted_f1": 0.5,
        "micro_precision": 0.5,
        "macro_precision": 0.5,
        "weighted_precision": 0.5,
        "micro_recall": 0.5,
        "macro_recall": 0.5,
        "weighted_recall": 0.5,
    }
    check_metrics(
        [y_true],
        [y_pred],
        expected=expected,
        span_type="value",
    )


def verify_all_metrics_structure(all_metrics, classes):
    span_types = ["token", "overlap", "exact", "superset", "value"]
    assert len(all_metrics.keys()) == 2
    summary_metrics = all_metrics["summary_metrics"]
    assert len(summary_metrics.keys()) == len(span_types)
    for span_type in span_types:
        assert len(summary_metrics[span_type].keys()) == 9
        for metric in [
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "weighted_f1",
            "weighted_precision",
            "weighted_recall",
        ]:
            assert isinstance(summary_metrics[span_type][metric], float)
    class_metrics = all_metrics["class_metrics"]
    assert len(class_metrics) == len(span_types)
    for span_type in span_types:
        assert len(class_metrics[span_type]) == len(classes)
        for cls_, metrics in class_metrics[span_type].items():
            assert cls_ in classes
            assert len(metrics.keys()) == 6
            for metric in ["f1", "precision", "recall"]:
                assert isinstance(metrics[metric], float)
            for metric in ["false_positives", "true_positives", "false_negatives"]:
                assert isinstance(metrics[metric], int)


def test_get_all_metrics():
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."

    y_mixed = extend_label(
        text,
        [
            {"start": 21, "end": 28, "label": "entity"},
            {"start": 62, "end": 65, "label": "date"},
        ],
        5,
    ) + extend_label(
        text,
        [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        5,
    )
    y_true = extend_label(
        text,
        [
            {"start": 7, "end": 20, "label": "entity"},
            {"start": 41, "end": 54, "label": "date"},
        ],
        10,
    )

    all_metrics = get_all_metrics(preds=y_mixed, labels=y_true)
    verify_all_metrics_structure(all_metrics=all_metrics, classes=["entity", "date"])


def test_get_all_metrics_missing_class():
    y_true = [[{"start": 0, "end": 7, "text": "a and b", "label": "entity"}]]
    span_types = ["token", "overlap", "exact", "superset", "value"]

    all_metrics_2_classes = get_all_metrics(
        preds=y_true, labels=y_true, field_names=["entity", "date"]
    )
    verify_all_metrics_structure(
        all_metrics=all_metrics_2_classes, classes=["entity", "date"]
    )

    all_metrics_1_classes = get_all_metrics(
        preds=y_true, labels=y_true, field_names=["entity"]
    )
    verify_all_metrics_structure(all_metrics=all_metrics_1_classes, classes=["entity"])

    for m in ["f1", "precision", "recall"]:
        # Macro should differ - but micro and weighted should be exactly the same.
        for span_type in span_types:
            assert (
                all_metrics_2_classes["summary_metrics"][span_type][f"macro_{m}"]
                == 0.5
                * all_metrics_1_classes["summary_metrics"][span_type][f"macro_{m}"]
            )
            assert (
                all_metrics_2_classes["summary_metrics"][span_type][f"micro_{m}"]
                == all_metrics_1_classes["summary_metrics"][span_type][f"micro_{m}"]
            )
            assert (
                all_metrics_2_classes["summary_metrics"][span_type][f"weighted_{m}"]
                == all_metrics_1_classes["summary_metrics"][span_type][f"weighted_{m}"]
            )


@pytest.mark.parametrize("classes", [[], ["date"]])
def test_all_metrics_no_class_match(classes):
    y_true = [[{"start": 0, "end": 7, "text": "a and b", "label": "entity"}]]
    all_metrics_0_classes = get_all_metrics(
        preds=y_true, labels=y_true, field_names=classes
    )
    verify_all_metrics_structure(all_metrics=all_metrics_0_classes, classes=classes)


@pytest.mark.parametrize("classes", [[], ["date"]])
def test_empty_preds_metrics(classes):
    all_metrics = get_all_metrics(preds=[], labels=[], field_names=classes)
    verify_all_metrics_structure(all_metrics=all_metrics, classes=classes)


@pytest.mark.parametrize("classes", [[], ["date"]])
def test_empty_preds_metrics(classes):
    all_metrics = get_all_metrics(preds=[[]], labels=[[]], field_names=classes)
    verify_all_metrics_structure(all_metrics=all_metrics, classes=classes)
