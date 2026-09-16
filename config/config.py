"""Validated experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    data_file: Path = Path("Dataset/reentrancy_1671.txt")
    output_root: Path = Path("EXPERIMENT")
    max_len: int = 500
    vector_dim: int = 300
    hidden_units: int = 300
    dense_units: int = 300
    dropout: float = 0.2
    learning_rate: float = 0.002
    batch_size: int = 32
    epochs: int = 40
    patience: int = 5
    folds: int = 5
    validation_folds: int = 5
    w_ratio: float = 0.6
    min_recall: float = 0.95
    threshold_strategy: str = "recall"
    early_stop_monitor: str = "val_pr_auc"
    seed: int = 42
    verbose: int = 2

    def __post_init__(self) -> None:
        if self.max_len < 3:
            raise ValueError("max_len must leave room for W, [SEP], and C")
        if self.vector_dim < 1 or self.hidden_units < 1 or self.dense_units < 1:
            raise ValueError("model dimensions must be positive")
        if self.batch_size < 1 or self.epochs < 1 or self.patience < 0:
            raise ValueError("batch_size/epochs must be positive and patience non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.5 <= self.w_ratio <= 0.8:
            raise ValueError("w_ratio must be in the documented range [0.5, 0.8]")
        if self.w_budget < 1 or self.c_budget < 1:
            raise ValueError("max_len and w_ratio must allocate tokens to both W and C")
        if self.folds < 2 or self.validation_folds < 2:
            raise ValueError("fold counts must be at least 2")
        if not 0.0 < self.min_recall <= 1.0:
            raise ValueError("min_recall must be in (0, 1]")
        if self.threshold_strategy not in ("recall", "eer"):
            raise ValueError("threshold_strategy must be 'recall' or 'eer'")
        if self.early_stop_monitor not in ("val_loss", "val_accuracy", "val_pr_auc"):
            raise ValueError(
                "early_stop_monitor must be one of val_loss, val_accuracy, val_pr_auc"
            )
        if self.verbose not in (0, 1, 2):
            raise ValueError("verbose must be 0, 1, or 2")

    @property
    def w_budget(self) -> int:
        return int((self.max_len - 1) * self.w_ratio)

    @property
    def c_budget(self) -> int:
        return self.max_len - 1 - self.w_budget

    def new_run_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return self.output_root / f"variant-c-{stamp}"

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["data_file"] = str(self.data_file)
        values["output_root"] = str(self.output_root)
        values["w_budget"] = self.w_budget
        values["c_budget"] = self.c_budget
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        fields = dict(values)
        fields.pop("w_budget", None)
        fields.pop("c_budget", None)
        fields["data_file"] = Path(fields["data_file"])
        fields["output_root"] = Path(fields["output_root"])
        return cls(**fields)

    def to_cli_args(self) -> list[str]:
        return [
            "--data-file", str(self.data_file),
            "--output-root", str(self.output_root),
            "--max-len", str(self.max_len),
            "--vector-dim", str(self.vector_dim),
            "--hidden-units", str(self.hidden_units),
            "--dense-units", str(self.dense_units),
            "--dropout", str(self.dropout),
            "--learning-rate", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--epochs", str(self.epochs),
            "--patience", str(self.patience),
            "--folds", str(self.folds),
            "--validation-folds", str(self.validation_folds),
            "--w-ratio", str(self.w_ratio),
            "--min-recall", str(self.min_recall),
            "--threshold-strategy", self.threshold_strategy,
            "--early-stop-monitor", self.early_stop_monitor,
            "--seed", str(self.seed),
            "--verbose", str(self.verbose),
        ]
