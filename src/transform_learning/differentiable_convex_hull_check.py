import torch
from transform_learning.dataset import CustomPointDataset

class DifferentiableConvexHull:

    def __init__(self, num_dim=12, num_samples=1000):
        self.dataset = None
        self.projected_dataset = None
        self.labels = None
        self.num_dim = num_dim
        self.num_samples = num_samples
        self.unconstrained_opt_scores = None
        self.kernel_scores = None
        self.results_text = ""
        self.txt_file_path = "src/transform_learning/convex_hull_results.txt"
    
    def load_dataset(self):
        # Load or generate your dataset here. For example:
        self.dataset = CustomPointDataset(num_samples=self.num_samples, eps=0.1, n=5, sampling_dist=[0.4, 0.4, 0.2], seed=42)
    
    def random_project_dataset(self):

        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        data, labels = zip(*[self.dataset[i] for i in range(len(self.dataset))])
        data = torch.stack(data)
        labels = torch.stack(labels)
        self.labels = labels
        
        # Random projection matrix
        projection_matrix = torch.randn(data.shape[1], self.num_dim)
        self.projected_dataset = data @ projection_matrix

    def unconstrained_optimization(self, test_point) -> float:
        return 0.0

    def kernel_based_prediction(self, test_point) -> float:
        return 0.0
    
    def evaluate(self):

        unc_scores = []
        kernel_scores = []
        
        for projected_point, (original_point, label) in zip(self.projected_dataset, self.dataset):
            
            unc_score = self.unconstrained_optimization(projected_point)
            kernel_score = self.kernel_based_prediction(projected_point)
            unc_scores.append(unc_score)
            kernel_scores.append(kernel_score)
        
        self.unconstrained_opt_scores = torch.tensor(unc_scores)
        self.kernel_scores = torch.tensor(kernel_scores)
    
    def stats(self):

        if self.unconstrained_opt_scores is None or self.kernel_scores is None:
            raise ValueError("Scores not computed. Call evaluate() first.")
        
        self.results_text += "Unconstrained Optimization Scores:"
        self.results_text += f"Mean: {self.unconstrained_opt_scores.mean().item():.4f}, Std: {self.unconstrained_opt_scores.std().item():.4f}\n"
        
        self.results_text += "Kernel-Based Prediction Scores:"
        self.results_text += f"Mean: {self.kernel_scores.mean().item():.4f}, Std: {self.kernel_scores.std().item():.4f}\n"

        # labels are binary, compute the mean score for each class
        for label in torch.unique(self.labels):
            label_mask = self.labels == label
            self.results_text += f"Class {label.item()} - Unconstrained Optimization Mean: {self.unconstrained_opt_scores[label_mask].mean().item():.4f}\n"
            self.results_text += f"Class {label.item()} - Unconstrained Optimization Std: {self.unconstrained_opt_scores[label_mask].std().item():.4f}\n"
            self.results_text += f"Class {label.item()} - Kernel-Based Prediction Mean: {self.kernel_scores[label_mask].mean().item():.4f}\n"
            self.results_text += f"Class {label.item()} - Kernel-Based Prediction Std: {self.kernel_scores[label_mask].std().item():.4f}\n"
        
    def run_experiment(self):
        self.load_dataset()
        self.random_project_dataset()
        self.evaluate()
        self.stats()
        with open(self.txt_file_path, "w") as f:
            f.write(self.results_text)
        f.close()

if __name__ == "__main__":
    experiment = DifferentiableConvexHull(num_dim=12, num_samples=1000)
    experiment.run_experiment()





        
