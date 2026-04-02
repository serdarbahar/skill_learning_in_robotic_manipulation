import torch
import torch.nn.functional as F


def vertex_reconstruction_loss(
    outputs: torch.Tensor,
    vertices_embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    outside_margin: float = 0.1,
) -> torch.Tensor:
    """Differentiable convex-hull proxy loss.

    For label 1 samples (inside/near hull), minimize reconstruction distance.
    For label 0 samples (outside), enforce a margin on reconstruction distance.
    """

    vertices_embeddings = vertices_embeddings.detach()

    sq_distances = torch.sum(
        (outputs.unsqueeze(1) - vertices_embeddings.unsqueeze(0)) ** 2,
        dim=-1,
    )
    weights = F.softmax(-sq_distances / temperature, dim=1)
    reconstruction = weights @ vertices_embeddings
    reconstruction_error = torch.sum((outputs - reconstruction) ** 2, dim=1)

    labels = labels.float()
    inside_loss = labels * reconstruction_error
    outside_loss = (1.0 - labels) * F.relu(outside_margin - reconstruction_error)

    return (inside_loss + outside_loss).mean()
