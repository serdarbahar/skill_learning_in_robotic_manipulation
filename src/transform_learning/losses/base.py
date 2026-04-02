import torch
from abc import ABC, abstractmethod

class TransformLoss(ABC):
    @abstractmethod
    def __call__(self, **kwargs) -> torch.Tensor:
        ...