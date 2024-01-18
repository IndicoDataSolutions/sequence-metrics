import pytest
from sequence_metrics.metrics import (
    seq_recall,
    seq_precision,
    get_seq_count_fn,
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
            assert precisions[cls_] == pytest.approx(
                expected[cls_]["precision"], abs=1e-3
            )

    assert micro_f1_score == pytest.approx(expected["micro-f1"], abs=1e-3)
    assert weighted_f1 == pytest.approx(expected["weighted-f1"], abs=1e-3)
    assert macro_f1 == pytest.approx(expected["macro-f1"], abs=1e-3)


def test_token_incorrect():
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."
    expected = {
        "entity": {
            "false_positives": 10,
            "false_negatives": 20,
            "true_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
        },
        "date": {
            "false_positives": 10,
            "false_negatives": 40,
            "true_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
        },
        "micro-f1": 0.0,
        "macro-f1": 0.0,
        "weighted-f1": 0.0,
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
    check_metrics(y_true, y_false_pos, expected, span_type="token")


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
        "micro-f1": 1.0,
        "macro-f1": 1.0,
        "weighted-f1": 1.0,
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
        "micro-f1": 0.6,
        "macro-f1": 0.593,
        "weighted-f1": 0.601,
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
        "micro-f1": 0.409,
        "macro-f1": 0.437,
        "weighted-f1": 0.392,
    }
    check_metrics(
        y_true,
        y_mixed,
        expected,
        span_type="token",
    )


def test_seq_incorrect():
    text = "Alert: Pepsi Company stocks are up today April 5, 2010 and no one profited."
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
    seq_expected_incorrect = {
        "entity": {
            "false_positives": 10,
            "false_negatives": 10,
            "true_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
        },
        "date": {
            "false_positives": 10,
            "false_negatives": 10,
            "true_positives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
        },
        "micro-f1": 0.0,
        "macro-f1": 0.0,
        "weighted-f1": 0.0,
    }

    # Overlap
    check_metrics(
        y_true,
        y_false_pos,
        seq_expected_incorrect,
        span_type="overlap",
    )

    # Exact
    check_metrics(
        y_true,
        y_false_pos,
        seq_expected_incorrect,
        span_type="exact",
    )

    # Superset
    check_metrics(
        y_true,
        y_false_pos,
        seq_expected_incorrect,
        span_type="superset",
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
def test_seq_mixed(overlapping):
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
        "micro-f1": 0.6,
        "macro-f1": 0.6,
        "weighted-f1": 0.6,
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
        "micro-f1": 0.4864,
        "macro-f1": 0.476,
        "weighted-f1": 0.476,
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
        "micro-f1": 0.66666,  # Calculated as the harmonic mean of Recall = 1, Precision = 0.5
        "macro-f1": 0.5,
        "weighted-f1": 1.0,  # because there is no support for class2
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
        "micro-f1": 0.66666,  # Calculated as the harmonic mean of Recall = 1, Precision = 0.5
        "macro-f1": 0.5,
        "weighted-f1": 1.0,  # because there is no support for class2
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
        "micro-f1": 1.0,
        "macro-f1": 1.0,
        "weighted-f1": 1.0,
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
        "micro-f1": 1.0,
        "macro-f1": 1.0,
        "weighted-f1": 1.0,
    }
    for span_type in ["overlap", "superset"]:
        check_metrics(
            [y_true],
            [y_pred],
            expected=expected,
            span_type=span_type,
        )


def test_whitespace():
    x = "a and b"
    y_true = [{"start": 0, "end": 7, "text": x, "label": "class1"}]
    y_pred = [
        {"start": 0, "end": 8, "text": x + " ", "label": "class1"},
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
        "micro-f1": 1.0,
        "macro-f1": 1.0,
        "weighted-f1": 1.0,
    }
    for span_type in ["superset", "overlap", "exact"]:
        check_metrics(
            [y_true],
            [y_pred],
            expected=expected,
            span_type=span_type,
        )
