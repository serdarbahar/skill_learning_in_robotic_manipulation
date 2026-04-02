# metrics/custom_metrics.py

import torch
from transform_learning.utils.geometry import check_in_hull

def hull_success_rate(embeddings, vertices_embeddings, labels):
    in_hull = check_in_hull(embeddings, vertices_embeddings)
    return torch.sum(in_hull == labels) / len(labels)