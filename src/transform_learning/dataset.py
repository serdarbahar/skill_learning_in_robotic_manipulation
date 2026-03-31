import torch
from torch.utils.data import Dataset

class CustomPointDataset(Dataset):
    def __init__(self, num_samples: int, eps: float, n: float, sampling_dist: list, seed: int):
        assert len(sampling_dist) == 3, "Sampling distribution must be a list of three values."
        
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.eps = eps
        self.n = n
        
        self.data = torch.zeros((num_samples, 1))
        self.labels = torch.zeros(num_samples, dtype=torch.long)
        
        dist_dist = torch.distributions.Categorical(torch.tensor(sampling_dist))

        for i in range(num_samples):
            dist_choice = dist_dist.sample().item()
            
            if dist_choice == 0:
                self.labels[i] = 0
                if torch.rand(1).item() > 0.5:
                    self.data[i] = torch.rand(1) * (eps * n - eps) + (1 + eps)
                else:
                    self.data[i] = torch.rand(1) * (eps * n - eps) + (-1 - eps * n)
                    
            elif dist_choice == 1:
                self.labels[i] = 1
                if torch.rand(1).item() > 0.5:
                    self.data[i] = torch.rand(1) * eps + 1
                else:
                    self.data[i] = torch.rand(1) * eps - (1 + eps)
            
            else:
                self.labels[i] = 1
                self.data[i] = torch.rand(1) * 2 - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]