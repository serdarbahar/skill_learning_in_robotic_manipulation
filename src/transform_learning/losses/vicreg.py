import torch
import torch.nn.functional as F
from transform_learning.losses.base import TransformLoss


class VICRegLoss(TransformLoss):
    """Variance-Covariance regularization (VICReg without the invariance term).

    Prevents representational collapse without forcing an isometry the way the
    old volume-preservation loss did:
      - variance term hinges each output dimension's std above ``gamma`` so no
        dimension is allowed to collapse to a constant.
      - covariance term decorrelates the output dimensions so information is
        spread across them rather than piling onto one axis.

    The invariance term of standard VICReg is omitted: there are no paired
    views here, only a single embedding per sample.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
        eps: float = 1e-4,
    ):
        self.gamma = gamma
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.eps = eps

    def __call__(self, outputs, **kwargs):
        return vicreg_loss(
            outputs,
            gamma=self.gamma,
            var_weight=self.var_weight,
            cov_weight=self.cov_weight,
            eps=self.eps,
        )


def vicreg_loss(
    outputs: torch.Tensor,
    gamma: float = 1.0,
    var_weight: float = 1.0,
    cov_weight: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    batch_size, dim = outputs.shape

    # Variance term: keep each dimension's std at or above gamma.
    std = torch.sqrt(outputs.var(dim=0) + eps)            # (D,)
    var_loss = F.relu(gamma - std).mean()

    # Covariance term: drive off-diagonal covariances to zero.
    centered = outputs - outputs.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (batch_size - 1)      # (D, D)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    cov_loss = (off_diag ** 2).sum() / dim

    return var_weight * var_loss + cov_weight * cov_loss
