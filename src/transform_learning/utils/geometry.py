import numpy as np
import torch
from scipy.spatial import Delaunay


def check_in_hull(query: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Return 1 if each query point is inside the convex hull of vertices, else 0.

    Uses Delaunay simplex lookup for dimensions >= 2, and interval checks for 1D.
    """

    #print(query[-1])  # Print first 5 query points for debugging
    #print(vertices)

    query_np = query.detach().cpu().numpy()
    vertices_np = vertices.detach().cpu().numpy()

    if query_np.ndim != 2 or vertices_np.ndim != 2:
        raise ValueError("query and vertices must both be 2D tensors")
    if query_np.shape[1] != vertices_np.shape[1]:
        raise ValueError("query and vertices must have the same feature dimension")

    feature_dim = vertices_np.shape[1]

    if feature_dim == 1:
        lower = np.min(vertices_np[:, 0])
        upper = np.max(vertices_np[:, 0])
        inside = (query_np[:, 0] >= lower) & (query_np[:, 0] <= upper)
        return torch.from_numpy(inside.astype(np.int64))

    if vertices_np.shape[0] < feature_dim + 1:
        print("Warning: Not enough vertices to form a convex hull in this dimension.")
        return torch.zeros(query_np.shape[0], dtype=torch.long)
    
    triangulation = Delaunay(vertices_np)
    simplex_idx = triangulation.find_simplex(query_np)
    inside = simplex_idx >= 0

    return torch.from_numpy(inside.astype(np.int64))
