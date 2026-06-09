import math
from typing import Optional

import torch
from transform_learning.losses.base import TransformLoss


class SIGRegLoss(TransformLoss):
    """Sketched Isotropic Gaussian Regularization (SIGReg, from LeJEPA).

    Pushes the embedding distribution toward an isotropic standard Gaussian,
    which prevents representational collapse: N(0, I) has unit variance in every
    direction, so the covariance is full rank and no dimension degenerates.

    By the Cramer-Wold device, a distribution is multivariate N(0, I) iff every
    1D projection is N(0, 1). SIGReg "sketches" this by sampling random unit
    directions, projecting the embeddings onto each, and penalizing departure
    from N(0, 1) with the Epps-Pulley goodness-of-fit statistic -- the weighted
    L2 distance between the empirical characteristic function and that of
    N(0, 1), which has a differentiable closed form. Fresh directions are drawn
    each call (the Monte-Carlo "sketch"), unless a seed is fixed.

    Unlike volume preservation it does not force an isometry, and unlike VICReg
    it constrains the full distribution shape rather than just variance and
    pairwise covariance.
    """

    def __init__(self, n_directions: int = 64, seed: Optional[int] = None):
        self.n_directions = n_directions
        self.seed = seed

    def __call__(self, outputs, **kwargs):
        return sigreg_loss(outputs, n_directions=self.n_directions, seed=self.seed)


def _epps_pulley_normal(proj: torch.Tensor) -> torch.Tensor:
    """Closed-form Epps-Pulley statistic against N(0, 1) for each column.

    ``proj`` is (B, K) of projected samples; returns a (K,) tensor of
    non-negative statistics (0 iff the projection matches N(0, 1)). Uses the
    Gaussian weight w(t) = exp(-t^2 / 2), giving

        T = sqrt(2pi)/B^2 * sum_{j,l} exp(-(x_j - x_l)^2 / 2)
            - 2 sqrt(pi)/B * sum_j exp(-x_j^2 / 4)
            + sqrt(2pi / 3).
    """
    batch_size = proj.shape[0]

    # pairwise term: sum_{j,l} exp(-(x_j - x_l)^2 / 2)
    pairwise_diff = proj.unsqueeze(0) - proj.unsqueeze(1)        # (B, B, K)
    pairwise_term = torch.exp(-0.5 * pairwise_diff ** 2).sum(dim=(0, 1))  # (K,)

    # cross term against the N(0, 1) characteristic function
    cross_term = torch.exp(-0.25 * proj ** 2).sum(dim=0)         # (K,)

    c_pair = math.sqrt(2.0 * math.pi)
    c_cross = math.sqrt(math.pi)
    c_const = math.sqrt(2.0 * math.pi / 3.0)

    return (
        c_pair / (batch_size ** 2) * pairwise_term
        - 2.0 * c_cross / batch_size * cross_term
        + c_const
    )


def sigreg_loss(
    outputs: torch.Tensor,
    n_directions: int = 64,
    seed: Optional[int] = None,
) -> torch.Tensor:
    _, dim = outputs.shape

    generator = None
    if seed is not None:
        generator = torch.Generator(device=outputs.device).manual_seed(seed)

    directions = torch.randn(
        n_directions, dim, generator=generator, device=outputs.device
    )
    directions = directions / directions.norm(dim=1, keepdim=True)   # (K, D)

    projections = outputs @ directions.T                             # (B, K)
    stats = _epps_pulley_normal(projections)                         # (K,)
    return stats.mean()
