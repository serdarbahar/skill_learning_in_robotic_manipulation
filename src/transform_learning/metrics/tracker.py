from typing import Dict, List

import torch

class MetricsTracker:
    def __init__(self):
        self.stats: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_success": [],
            "val_success": [],
            "test_success": [],
        }

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return float(value)

    def log(self, split: str, loss=None, success=None):
        if loss is not None:
            self.stats[f"{split}_loss"].append(self._to_scalar(loss))
        if success is not None:
            self.stats[f"{split}_success"].append(self._to_scalar(success))

class EmbeddingsTracker:
    def __init__(self):

        self.stats: Dict[str, List[torch.Tensor]] = {
            "vertices_embeddings": [],
            "train_embeddings": [],
            "val_embeddings": [],
            "test_embeddings": [],
        }

    def log_vertices_embeddings(self, embeddings: torch.Tensor):
        self.clear_vertices_embeddings()  # Clear previous vertices embeddings to save memory
        self.stats["vertices_embeddings"].append(embeddings.detach().cpu())
    
    def log_train_embeddings(self, embeddings: torch.Tensor):
        self.stats["train_embeddings"].append(embeddings.detach().cpu())  

    def log_val_embeddings(self, embeddings: torch.Tensor):
        self.stats["val_embeddings"].append(embeddings.detach().cpu())

    def log_test_embeddings(self, embeddings: torch.Tensor):
        self.stats["test_embeddings"].append(embeddings.detach().cpu())

    def clear_vertices_embeddings(self):
        self.stats["vertices_embeddings"] = []
    
    def clear_train_embeddings(self):
        self.stats["train_embeddings"] = []

    def clear_val_embeddings(self):
        self.stats["val_embeddings"] = []

    def clear_test_embeddings(self):
        self.stats["test_embeddings"] = []
        
    def get_vertices_embeddings(self) -> torch.Tensor:
        if not self.stats["vertices_embeddings"]:
            raise ValueError("No vertices embeddings logged yet.")
        return torch.cat(self.stats["vertices_embeddings"], dim=0)
    
    def get_train_embeddings(self) -> torch.Tensor:
        if not self.stats["train_embeddings"]:
            raise ValueError("No training embeddings logged yet.")
        return torch.cat(self.stats["train_embeddings"], dim=0)
    
    def get_val_embeddings(self) -> torch.Tensor:
        if not self.stats["val_embeddings"]:
            raise ValueError("No validation embeddings logged yet.")
        return torch.cat(self.stats["val_embeddings"], dim=0)

    def get_test_embeddings(self) -> torch.Tensor:
        if not self.stats["test_embeddings"]:
            raise ValueError("No test embeddings logged yet.")
        return torch.cat(self.stats["test_embeddings"], dim=0)
    