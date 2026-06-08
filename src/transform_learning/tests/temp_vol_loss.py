import torch
from transform_learning.losses.volume_preservation import volume_preservation_loss

#inputs, vertices:1d
#outputs, vertices_embeddings: 2d

def example_transform(inp: torch.Tensor):

    res = torch.zeros((inp.shape[0], 2))
    res[:, 0] = 0.0 * inp[:, 0] - 5
    res[:, 1] = 0.0 * inp[:, 0] - 5
    return res

eps = 1e-6
inputs = torch.tensor([[-2.0], [-1.5], [-1.0], [-0.5]])  # (4, 1)
vertices = torch.tensor([[-1.0], [0.0], [1.0]])  # (3, 1)

outputs = example_transform(inputs)
vertices_embeddings = example_transform(vertices)

print(f"outputs: {outputs}")
print(f"vertices_embeddings: {vertices_embeddings}")
print("\n----\n")

# pairwise distances in input space: each sample to each vertex
input_dists = torch.cdist(inputs, vertices)          # (B, M)
output_dists = torch.cdist(outputs, vertices_embeddings)  # (B, M)
log_ratios = torch.log(output_dists + eps) - torch.log(input_dists + eps)

loss = log_ratios.var(dim=1).mean() + 10*log_ratios.mean()**2

print(f"Input Dists: {input_dists}\n")
print(f"Output Dists: {output_dists}\n")
print(f"Log Ratios: {log_ratios}\n")
print(f"Mean Log Ratio: {log_ratios.mean().item():.6f}\n")
print(f"Volume Preservation Loss: {loss.item():.6f}\n")
