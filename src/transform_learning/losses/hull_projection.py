import torch
import torch.nn.functional as F
from transform_learning.losses.base import TransformLoss


class HullProjectionLoss(TransformLoss):
    """Convex-hull membership loss via a differentiable projection distance.

    For each query we solve, over the probability simplex,

        w* = argmin_w || query - w @ vertices ||^2 ,

    i.e. the projection of the query onto the convex hull of the vertex
    embeddings. The resulting distance is ~0 when the query is inside the hull
    and grows with how far outside it lies -- a proper (one-sided) distance-to-
    hull surrogate, unlike the old softmax-blend reconstruction error.

    The argmin is solved with a softmax-parameterized inner loop with no
    retained gradient. The distance is then recomputed in-graph with w* held
    fixed: by Danskin's theorem this is the correct gradient of the min-value
    function, since the derivative w.r.t. w vanishes at the optimum.

        label 1 (inside)  : minimize distance  -> query pulled into the hull
        label 0 (outside) : hinge at `margin`  -> query pushed out of the hull
    """

    def __init__(
        self,
        margin: float = 1.0,
        n_inner_steps: int = 50,
        inner_lr: float = 0.1,
    ):
        self.margin = margin
        self.n_inner_steps = n_inner_steps
        self.inner_lr = inner_lr

    def __call__(self, outputs, vertices_embeddings, labels, **kwargs):
        return hull_projection_loss(
            outputs,
            vertices_embeddings,
            labels,
            margin=self.margin,
            n_inner_steps=self.n_inner_steps,
            inner_lr=self.inner_lr,
        )


def _project_to_hull(
    queries: torch.Tensor,
    vertices: torch.Tensor,
    n_inner_steps: int,
    inner_lr: float,
) -> torch.Tensor:
    """Solve the batch of independent hull projections.

    Returns the optimal simplex weights w* of shape (B, M), detached. Each
    query's weights are independent, so the whole batch is optimized at once.
    """
    queries = queries.detach()
    vertices = vertices.detach().to(queries.device)
    batch_size, num_vertices = queries.shape[0], vertices.shape[0]

    # Force grad tracking even when called inside a torch.no_grad() block
    # (e.g. validation/eval), since this inner solve needs its own graph.
    with torch.enable_grad():
        logits = torch.zeros(
            batch_size, num_vertices, device=queries.device, requires_grad=True
        )
        optimizer = torch.optim.Adam([logits], lr=inner_lr)
        for _ in range(n_inner_steps):
            optimizer.zero_grad()
            weights = F.softmax(logits, dim=1)            # (B, M)
            reconstruction = weights @ vertices            # (B, d)
            loss = ((queries - reconstruction) ** 2).sum(dim=1).mean()
            loss.backward()
            optimizer.step()

        return F.softmax(logits, dim=1).detach()


def hull_projection_loss(
    outputs: torch.Tensor,
    vertices_embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    n_inner_steps: int = 50,
    inner_lr: float = 0.1,
) -> torch.Tensor:
    w_star = _project_to_hull(
        outputs, vertices_embeddings, n_inner_steps, inner_lr
    )  # (B, M), detached

    # Recompute the projection in-graph with w* fixed (Danskin's theorem).
    # Gradients flow to outputs (and to vertices_embeddings if it carries grad).
    projection = w_star @ vertices_embeddings.to(outputs.device)   # (B, d)
    distance = torch.sum((outputs - projection) ** 2, dim=1)        # (B,)

    labels = labels.float()
    inside_loss = labels * distance

    # calculate adaptive margin based on the distance of the query to the hull
    adaptive_margin = margin - distance.detach().sqrt()  # (B,)

    outside_loss = (1.0 - labels) * F.relu(adaptive_margin - distance)

    return (inside_loss + outside_loss).mean()
