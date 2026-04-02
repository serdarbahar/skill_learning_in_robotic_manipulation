from .base import TransformLoss

class CompositeLoss(TransformLoss):
    def __init__(self, losses: list[tuple[float, TransformLoss]]):
        self.losses = losses

    def __call__(self, **kwargs):
        return sum(w * loss(**kwargs) for w, loss in self.losses)