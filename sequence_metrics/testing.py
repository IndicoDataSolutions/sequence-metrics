def verify_all_metrics_structure(all_metrics, classes, none_classes=None):
    span_types = ["token", "overlap", "exact", "superset", "value"]
    assert len(all_metrics.keys()) == 2
    summary_metrics = all_metrics["summary_metrics"]
    assert len(summary_metrics.keys()) == len(span_types)
    for span_type in span_types:
        if none_classes and span_type != "value":
            assert summary_metrics[span_type] is None
        else:
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
                if none_classes and cls_ in none_classes and span_type != "value":
                    assert (
                        metrics[metric] is None
                    ), f"{cls_} {metric}, {span_type} {metrics[metric]} should be None"
                else:
                    assert isinstance(metrics[metric], float)
            for metric in ["false_positives", "true_positives", "false_negatives"]:
                if none_classes and cls_ in none_classes and span_type != "value":
                    assert metrics[metric] is None
                else:
                    assert isinstance(metrics[metric], int)


def insert_text(docs, labels):
    if len(docs) != len(labels):
        raise ValueError("Number of documents must be equal to the number of labels")
    for doc, label in zip(docs, labels):
        for l in label:
            if "text" not in l:
                l["text"] = doc[l["start"] : l["end"]]
    return labels


def extend_label(text, label, amt):
    return insert_text([text for _ in range(amt)], [label for _ in range(amt)])


def remove_label(recs, label):
    return [[pred for pred in rec if not pred.get("label") == label] for rec in recs]
