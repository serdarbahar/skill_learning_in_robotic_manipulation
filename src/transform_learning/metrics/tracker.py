from typing import Dict, List

import torch


class MetricsTracker:
    def __init__(self):
        self.stats: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_error": [],
            "val_error": [],
            "test_error": [],
        }

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return float(value)

    def log(self, split: str, loss=None, error=None):
        if loss is not None:
            self.stats[f"{split}_loss"].append(self._to_scalar(loss))
        if error is not None:
            self.stats[f"{split}_error"].append(self._to_scalar(error))
