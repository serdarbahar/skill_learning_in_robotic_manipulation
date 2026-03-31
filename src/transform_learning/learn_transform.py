import numpy as np
import torch
from utils.mlp import MLP
from torch.utils.data import DataLoader, random_split
from transform_learning.dataset import CustomPointDataset

class LearnTransform:

    def __init__(self, points, device=torch.device('cpu')):

        assert points.isinstance(points, torch.Tensor), "Data points must be a torch.Tensor"
        assert points.ndim == 2, "Data points must be a 2D tensor of shape (num_samples, num_features)"
        assert device in [torch.device('cpu'), torch.device('cuda')], "Device must be either 'cpu' or 'cuda'"

        self.points = (self.points - self.points.min(dim=0)[0]) / (self.points.max(dim=0)[0] - self.points.min(dim=0)[0]) * 2 - 1
        self.init_dim = points.shape[1]
        self.device = device
        self.train_val_test_split = [0.7, 0.15, 0.15]

    def generate_dataset(self, num_samples: int, eps: float, n: float, 
                         sampling_dist: list, seed: int):
        
        self.dataset = CustomPointDataset(num_samples, eps, n, sampling_dist, seed)
        self.train_size = int(self.train_val_test_split[0] * num_samples)
        self.val_size = int(self.train_val_test_split[1] * num_samples)
        self.test_size = num_samples - self.train_size - self.val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [self.train_size, self.val_size, self.test_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
    def train(self, num_epochs, learning_rate, batch_size, hidden_dim, out_dim, seed): 

        self.model = MLP(self.init_dim, hidden_dim, out_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def check_convex_hull(self, test_point, points): pass
    def evaluate(self, num_samples): pass
    def visualize(self): pass
        
