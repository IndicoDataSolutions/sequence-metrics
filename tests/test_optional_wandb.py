import subprocess
import sys


def test_metrics_without_wandb():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from importlib.abc import MetaPathFinder

class NoWandb(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "wandb":
            raise ModuleNotFoundError("No module named 'wandb'", name="wandb")

sys.meta_path.insert(0, NoWandb())
import sequence_metrics.metrics
try:
    import sequence_metrics.wandb_tools
except ImportError as exc:
    assert "sequence-metrics[wandb]" in str(exc), str(exc)
else:
    raise AssertionError("wandb_tools imported without wandb")
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
