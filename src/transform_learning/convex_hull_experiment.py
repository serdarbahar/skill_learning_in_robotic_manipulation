import torch
from transform_learning.dataset import CustomPointDataset
from transform_learning.differentiable_convex_hull import unconstrained_optimization, kernel_reconstruction 
import time

class DifferentiableConvexHull:

    def __init__(self, num_dim=12, num_samples=1000, num_test_points=100):
        self.dataset = None
        self.projected_dataset = None
        self.labels = None
        self.num_dim = num_dim
        self.num_samples = num_samples
        self.num_test_points = num_test_points

        self.unconstrained_opt_scores = None
        self.kernel_scores = None
        self.time_unconstrained = None
        self.time_kernel = None

        self.results_text = ""
        self.txt_file_path = "src/transform_learning/convex_hull_results.txt"
    
    def load_dataset(self):
        # Load or generate your dataset here. For example:
        self.dataset = CustomPointDataset(num_samples=self.num_samples, eps=0.1, n=5, sampling_dist=[0.4, 0.4, 0.2], seed=42)

        self.test_points = torch.rand((self.num_test_points-2, 1)) * 2 - 1  # random test points in the range [-1, 1]
        self.test_points = torch.cat((self.test_points, torch.tensor([[1.0], [-1.0]])))  # add boundary points
    
    def random_project_dataset(self):
        
        data, labels = zip(*[self.dataset[i] for i in range(len(self.dataset))])
        data = torch.stack(data)
        labels = torch.stack(labels)
        self.labels = labels

        
        # Random projection matrix
        projection_matrix = torch.randn(data.shape[1], self.num_dim)

        self.projected_dataset = data @ projection_matrix
        self.projected_test_points = self.test_points @ projection_matrix

    def _unconstrained_optimization(self, query_embedding) -> float:
        return 0.0
        #return unconstrained_optimization(query_embedding, self.projected_test_points)

    def _kernel_reconstruction(self, query_embedding) -> float:
        return 0.0
        #return kernel_reconstruction(query_embedding, self.projected_test_points)
    
    def evaluate(self):

        unc_scores = []
        kernel_scores = []
        time_unconstrained = []
        time_kernel = []
                
        for projected_point, (original_point, label) in zip(self.projected_dataset, self.dataset):
    
            init_time = time.time() 
            unc_score = self._unconstrained_optimization(projected_point)
            time_unconstrained.append(time.time() - init_time)

            init_time = time.time()
            kernel_score = self._kernel_reconstruction(projected_point)
            time_kernel.append(time.time() - init_time)

            unc_scores.append(unc_score)
            kernel_scores.append(kernel_score)
        
        self.unconstrained_opt_scores = torch.tensor(unc_scores)
        self.kernel_scores = torch.tensor(kernel_scores)
        self.time_unconstrained = torch.tensor(time_unconstrained)
        self.time_kernel = torch.tensor(time_kernel)
    
    def stats(self):
        
        self.results_text += "Unconstrained Optimization Scores:"
        self.results_text += f"Mean: {self.unconstrained_opt_scores.mean().item():.4f}, Std: {self.unconstrained_opt_scores.std().item():.4f}\n"
        self.results_text += f"Time (Unconstrained): Mean: {self.time_unconstrained.mean().item():.4f}, Std: {self.time_unconstrained.std().item():.4f}\n"

        self.results_text += "Kernel-Based Prediction Scores:"
        self.results_text += f"Mean: {self.kernel_scores.mean().item():.4f}, Std: {self.kernel_scores.std().item():.4f}\n"
        self.results_text += f"Time (Kernel): Mean: {self.time_kernel.mean().item():.4f}, Std: {self.time_kernel.std().item():.4f}\n"

        # labels are binary, compute the mean score for each class
        for label in torch.unique(self.labels):
            label_mask = self.labels == label
            self.results_text += f"Class {label.item()} - Unconstrained Optimization Mean: {self.unconstrained_opt_scores[label_mask].mean().item():.4f}, Std: {self.unconstrained_opt_scores[label_mask].std().item():.4f}\n"
            self.results_text += f"Class {label.item()} - Kernel-Based Prediction Mean: {self.kernel_scores[label_mask].mean().item():.4f}, Std: {self.kernel_scores[label_mask].std().item():.4f}\n"
        
    def run_experiment(self):
        self.load_dataset()
        self.random_project_dataset()
        self.evaluate()
        self.stats()
        with open(self.txt_file_path, "a") as f:
            f.write(f"Experiment with num_dim={self.num_dim}, num_samples={self.num_samples}, num_test_points={self.num_test_points}\n\n")
            f.write(self.results_text)
            f.write("="*50 + "\n")
        f.close()

if __name__ == "__main__":

    num_dims = [2, 4]
    num_samples = [1000, 5000]
    num_test_points = [2, 5]

    for dim in num_dims:
        for samples in num_samples:
            for test_points in num_test_points:
                print(f"Running experiment with dim={dim}, samples={samples}, test_points={test_points}")
                experiment = DifferentiableConvexHull(num_dim=dim, num_samples=samples, num_test_points=test_points)
                experiment.run_experiment()





        
