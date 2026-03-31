import torch
import torch.nn.functional as F


def unconstrained_optimization(query: torch.Tensor, vertices: torch.Tensor, n_inner_steps: int = 50, lr: float = 0.1) -> float:
    
    M = vertices.shape[0]
    logits = torch.zeros(M, requires_grad=True)

    optimizer = torch.optim.Adam([logits], lr=lr)

    for _ in range(n_inner_steps):
        optimizer.zero_grad()
        weights = F.softmax(logits, dim=0)              # (M,)
        reconstruction = weights @ vertices              # (d,)
        loss = torch.sum((query - reconstruction) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        weights = F.softmax(logits, dim=0)
        reconstruction = weights @ vertices
        distance = torch.sum((query - reconstruction) ** 2)

    return distance.item()


def kernel_reconstruction(query: torch.Tensor, vertices: torch.Tensor, temperature: float = 1.0) -> float:

    sq_distances = torch.sum((vertices - query.unsqueeze(0)) ** 2, dim=1)  # (M,)
    weights = F.softmax(-sq_distances / temperature, dim=0)                 # (M,)
    reconstruction = weights @ vertices                                     # (d,)
    error = torch.sum((query - reconstruction) ** 2)

    return error.item()