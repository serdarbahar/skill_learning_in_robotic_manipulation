import torch
from transform_learning.losses.volume_preservation import volume_preservation_loss

#inputs, vertices:1d
#outputs, vertices_embeddings: 2d
eps = 1e-6

inputs = torch.tensor([[-2.0], [-1.5], [-1.0], [-0.5]])  # (4, 1)

vertices = torch.tensor([[-1.0], [0.0], [1.0]])  # (3, 1)

outputs = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # (4, 2)

vertices_embeddings = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # (3, 2)


# pairwise distances in input space: each sample to each vertex
input_dists = torch.cdist(inputs, vertices)          # (B, M)
output_dists = torch.cdist(outputs, vertices_embeddings)  # (B, M)
log_ratios = torch.log(output_dists + eps) - torch.log(input_dists + eps)
loss = log_ratios.var(dim=1).mean()

print(f"Input Dists: {input_dists}\n")
print(f"Output Dists: {output_dists}\n")
print(f"Log Ratios: {log_ratios}\n")
print(f"Volume Preservation Loss: {loss.item():.6f}\n")
