import pytest

import wandb
from sequence_metrics.wandb_tools import log_to_wandb


def test_metrics():
    wandb.init(project="test_metrics-2", entity="indico")
    log_to_wandb(
        test_x=["a and b"],
        test_y=[
            [
                {
                    "start": 0,
                    "end": 1,
                    "text": "a",
                    "label": "label_1",
                    "metadata": {"some": "metadata"},
                }
            ]
        ],
        pred_y=[
            [
                {
                    "start": 6,
                    "end": 7,
                    "text": "b",
                    "label": "label_1",
                    "metadata": {"some": "other metadata"},
                }
            ]
        ],
        per_sample_metadata=[{"foo": "bar"}],
        metadata={"num_epochs": 5},
    )
