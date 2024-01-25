import typing as t

import pandas as pd

import wandb
from sequence_metrics.metrics import (
    EQUALITY_FN_MAP,
    _get_unique_classes,
    get_all_metrics,
    get_seq_count_fn,
)


class LabelSpan(t.TypedDict):
    label: str
    start: int | None
    end: int | None
    text: str | None
    metadata: dict[str, t.Any]


class PredSpan(LabelSpan):
    confidence: dict[str, float] | None


Label = list[LabelSpan]
Pred = list[PredSpan]

PRED_TYPE_MAP = {
    "false_positives": "FP",
    "true_positives": "TP",
    "false_negatives": "FN",
}


def sequence_labeling_counts(true, predicted, equality_fn):
    """
    Return FP, FN, and TP counts
    """


def map_span(pred, tpe):
    output = {}
    for key, value in pred.items():
        if key == "metadata":
            for metadata_key, metadata_value in value.items():
                output[f"{tpe}_{metadata_key}"] = metadata_value
        elif key in {"start", "end", "text"}:
            output[f"{tpe}_{key}"] = value
        else:
            output[key] = value
    return output


def span_type_metrics_to_dataframe(span_type_metrics):
    records = []
    for cls, class_metrics in span_type_metrics.items():
        records.append({"class": cls, **class_metrics})
    return pd.DataFrame.from_records(records)


def flatten_summary_metrics(summary_metrics):
    return {
        f"{span_type}_{key}": value
        for span_type, span_type_metrics in summary_metrics.items()
        for key, value in span_type_metrics.items()
    }


def log_to_wandb(
    *,
    test_x: list[str],
    test_y: list[Label],
    pred_y: list[Pred],
    per_sample_metadata: list[dict[str, t.Any]],
    metadata: dict[str, t.Any],
):
    assert len(test_x) == len(test_y)
    assert len(test_x) == len(pred_y)
    assert len(test_x) == len(per_sample_metadata)
    # unique_classes = _get_unique_classes(test_y, pred_y)
    equality_types = ["overlap", "exact", "superset", "value"]

    flat_spans = {equality_type: [] for equality_type in equality_types}
    documents = []

    for i, (text, true_annotations, predicted_annotations) in enumerate(
        zip(test_x, test_y, pred_y)
    ):
        # add doc idx to make verification easier
        for annotations in [true_annotations, predicted_annotations]:
            for annotation in annotations:
                annotation["doc_idx"] = i
        documents.append({"text": text, "doc_idx": i})

        for equality_type in equality_types:
            equality_fn = EQUALITY_FN_MAP[equality_type]
            for true_annotation in true_annotations:
                for pred_annotation in predicted_annotations:
                    if equality_fn(true_annotation, pred_annotation):
                        if pred_annotation["label"] == true_annotation["label"]:
                            flat_spans[equality_type].append(
                                {
                                    **map_span(pred_annotation, "pred"),
                                    **map_span(true_annotation, "label"),
                                    "pred_type": "TP",
                                }
                            )
                            break
                else:
                    flat_spans[equality_type].append(
                        {
                            **map_span(true_annotation, "label"),
                            "pred_type": "FN",
                        }
                    )

            for pred_annotation in predicted_annotations:
                for true_annotation in true_annotations:
                    if (
                        equality_fn(true_annotation, pred_annotation)
                        and true_annotation["label"] == pred_annotation["label"]
                    ):
                        break
                else:
                    flat_spans[equality_type].append(
                        {
                            **map_span(pred_annotation, "pred"),
                            "pred_type": "FP",
                        }
                    )

    tables = {k: pd.DataFrame.from_records(rows) for k, rows in flat_spans.items()}

    for table_name, df_table in tables.items():
        wandb.log({table_name: wandb.Table(dataframe=df_table)})

    documents_df = pd.DataFrame.from_records(documents)
    wandb.log({"documents": wandb.Table(dataframe=documents_df)})
    all_metrics = get_all_metrics(pred_y, test_y)
    class_metrics_tables = {
        f"{span_type}_metrics": wandb.Table(
            dataframe=span_type_metrics_to_dataframe(span_type_metrics)
        )
        for span_type, span_type_metrics in all_metrics["class_metrics"].items()
    }
    wandb.log(class_metrics_tables)
    wandb.log(flatten_summary_metrics(all_metrics["summary_metrics"]))
    wandb.log(metadata)

    return {
        "example_tables": tables,
        "all_metrics": all_metrics,
        "documents_table": documents_df,
        "metadata": metadata,
    }
