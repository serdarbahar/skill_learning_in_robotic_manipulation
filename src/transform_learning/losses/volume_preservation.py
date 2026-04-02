import torch
from transform_learning.losses.base import TransformLoss

class VolumePreservationLoss(TransformLoss):

    def __init__(self): pass

    def __call__(self, outputs,
                        vertices_embeddings,
                        inputs,
                        vertices, **kwargs):
        return volume_preservation_loss(outputs, vertices_embeddings, inputs, vertices)

def volume_preservation_loss(
    outputs: torch.Tensor,
    vertices_embeddings: torch.Tensor,
    inputs: torch.Tensor,
    vertices: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize changes in local volume between input and output space.

    Computes the ratio of pairwise distance distributions before and after
    the transformation. If the mapping preserves volume, pairwise distances
    scale uniformly and the log-ratio variance is zero.
    """
    
    # pairwise distances in input space: each sample to each vertex
    input_dists = torch.cdist(inputs, vertices)          # (B, M)
    output_dists = torch.cdist(outputs, vertices_embeddings)  # (B, M)

    log_ratios = torch.log(output_dists + eps) - torch.log(input_dists + eps)
    loss = log_ratios.var(dim=1).mean()

    return loss