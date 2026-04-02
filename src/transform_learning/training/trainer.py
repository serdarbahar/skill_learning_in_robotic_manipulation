from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from transform_learning.data import CustomPointDataset
from transform_learning.losses.base import TransformLoss
from transform_learning.metrics import MetricsTracker, EmbeddingsTracker
from transform_learning.metrics import MetricsVisualizer, EmbeddingsVisualizer
from utils.mlp import MLP
from transform_learning.metrics.custom_metrics import hull_success_rate



class TransformTrainer:
    def __init__(self, device=torch.device("cpu")):

        assert device in [torch.device("cpu"), torch.device("cuda")], (
            "Device must be either 'cpu' or 'cuda'"
        )
        self.device = device

        self.metrics_tracker = MetricsTracker()
        self.embeddings_tracker = EmbeddingsTracker()
        self.metrics_visualizer = MetricsVisualizer()
        self.embeddings_visualizer = EmbeddingsVisualizer()

    def generate_dataset(
        self,
        num_samples: int,
        eps: float,
        n: float,
        sampling_dist: list,
        batch_size: int,
        train_val_test_split: list = [0.7, 0.15, 0.15],
        seed: Optional[int] = None,
    ):
        self.train_val_test_split = train_val_test_split
        self.dataset = CustomPointDataset(num_samples, eps, n, sampling_dist, seed)
        self.train_size = int(self.train_val_test_split[0] * num_samples)
        self.val_size = int(self.train_val_test_split[1] * num_samples)
        self.test_size = num_samples - self.train_size - self.val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [self.train_size, self.val_size, self.test_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

    def train(
        self,
        num_epochs,
        vertices: torch.Tensor,
        loss_fn: TransformLoss,
        learning_rate,
        weight_decay,
        hidden_dim,
        num_hidden_dim_layers,
        out_dim,
        activation_fn,
        seed=None,
    ):
        assert vertices.ndim == 2, "Vertices should be a 2D tensor of shape (num_vertices, vertex_dim)"

        if seed is not None:
            torch.manual_seed(seed)

        self.vertices = vertices
        self.init_dim = self.vertices.shape[1]
        self.loss_fn = loss_fn

        self.model = MLP(self.init_dim, hidden_dim, out_dim, num_hidden_dim_layers, activation_fn).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for _ in range(num_epochs):

            
            #self.embeddings_tracker.clear_vertices_embeddings()
            self.embeddings_tracker.clear_train_embeddings()

            self.model.train()
            epoch_loss = 0.0
            for batch in self.train_loader:
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data)

                with torch.no_grad():
                    vertices_embeddings = self.model(self.vertices.to(self.device))
                    self.embeddings_tracker.log_vertices_embeddings(vertices_embeddings)
        
                loss = self.loss_fn(
                    outputs=outputs, vertices_embeddings=self.embeddings_tracker.get_vertices_embeddings(), labels=labels,
                    inputs=data, vertices=self.vertices.to(self.device)
                )

                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                self.embeddings_tracker.log_train_embeddings(outputs)

            self.model.eval()
    
            labels = CustomPointDataset.get_labels_for_subset(self.train_dataset)
            success = hull_success_rate(
                self.embeddings_tracker.get_train_embeddings(),
                self.embeddings_tracker.get_vertices_embeddings(),
                labels
            )
            self.metrics_tracker.log("train", loss=epoch_loss / len(self.train_loader), success=success)

            self.validate()

    def validate(self):
        self.embeddings_tracker.clear_val_embeddings()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(
                    outputs=outputs, vertices_embeddings=self.embeddings_tracker.get_vertices_embeddings(), labels=labels,
                    inputs=data, vertices=self.vertices.to(self.device)
                )
                val_loss += loss.item()
                self.embeddings_tracker.log_val_embeddings(outputs)

        labels = CustomPointDataset.get_labels_for_subset(self.val_dataset)
        success = hull_success_rate(
            self.embeddings_tracker.get_val_embeddings(),
            self.embeddings_tracker.get_vertices_embeddings(),
            labels
        )
        self.metrics_tracker.log("val", loss=val_loss / len(self.val_loader), success=success)

    def evaluate(self):

        self.embeddings_tracker.clear_test_embeddings()

        test_loss = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                self.embeddings_tracker.log_test_embeddings(outputs)
                loss = self.loss_fn(
                    outputs=outputs, vertices_embeddings=self.embeddings_tracker.get_vertices_embeddings(), labels=labels,
                    inputs=data, vertices=self.vertices.to(self.device)
                )
                test_loss += loss.item()

        labels = CustomPointDataset.get_labels_for_subset(self.test_dataset)
        success = hull_success_rate(
            self.embeddings_tracker.get_test_embeddings(),
            self.embeddings_tracker.get_vertices_embeddings(),
            labels
        )

        self.metrics_tracker.log("test", loss=(test_loss / len(self.test_loader)), success=success)

    def visualize(
        self, metrics=None, save_dir=None, title="Training Metrics"
    ) -> None:

        self.metrics_visualizer(
            stats=self.metrics_tracker.stats,
            metrics=metrics,
            save_dir=save_dir,
            title=title,
        )

        self.embeddings_visualizer(
            model=self.model,
            dataset=self.dataset,
            vertices=self.vertices,
            save_dir=save_dir,
        )