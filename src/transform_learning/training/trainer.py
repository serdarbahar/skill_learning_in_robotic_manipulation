import torch
from torch.utils.data import DataLoader, random_split

from transform_learning.data import CustomPointDataset
from transform_learning.losses import vertex_reconstruction_loss
from transform_learning.metrics import MetricsTracker
from transform_learning.visualization import MetricsVisualizer
from transform_learning.utils.geometry import check_in_hull
from utils.mlp import MLP

class TransformTrainer:
    def __init__(self, vertices, device=torch.device("cpu")):

        assert type(vertices) is torch.Tensor, "Vertices must be a torch.Tensor"
        assert vertices.ndim == 2, (
            "Vertices must be a 2D tensor of shape (num_samples, num_features)"
        )
        assert device in [torch.device("cpu"), torch.device("cuda")], (
            "Device must be either 'cpu' or 'cuda'"
        )

        self.vertices = vertices
        self.init_dim = self.vertices.shape[1]
        self.device = device

        self.metrics_tracker = MetricsTracker()
        self.metrics_visualizer = MetricsVisualizer()
        self.stats = self.metrics_tracker.stats
        self.vertices_embeddings_current_epoch = []
        self.trainset_embeddings_current_epoch = []
        self.valset_embeddings_current_epoch = []
        self.testset_embeddings = []

    def generate_dataset(
        self,
        num_samples: int,
        eps: float,
        n: float,
        sampling_dist: list,
        seed: int,
        batch_size: int,
        train_val_test_split: list = [0.7, 0.15, 0.15],
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
        learning_rate,
        hidden_dim,
        num_hidden_dim_layers,
        out_dim,
        seed=None,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        self.model = MLP(self.init_dim, hidden_dim, out_dim, num_hidden_dim_layers, torch.nn.Tanh).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for _ in range(num_epochs):
            self.trainset_embeddings_current_epoch = []

            self.vertices_embeddings_current_epoch = self.model(
                self.vertices.to(self.device)
            )

            self.model.train()
            epoch_loss = 0.0
            for batch in self.train_loader:
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(data)
                loss = vertex_reconstruction_loss(
                    outputs, self.vertices_embeddings_current_epoch, labels
                )
                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                self.trainset_embeddings_current_epoch.append(outputs)

            self.model.eval()
            epoch_loss /= len(self.train_loader)
            self.metrics_tracker.log("train", loss=epoch_loss, error=self.training_error())

            self.validate()

    def validate(self):
        self.valset_embeddings_current_epoch = []
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                data, labels = batch
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = vertex_reconstruction_loss(
                    outputs, self.vertices_embeddings_current_epoch, labels
                )
                val_loss += loss.item()
                self.valset_embeddings_current_epoch.append(outputs)

        val_loss /= len(self.val_loader)
        self.metrics_tracker.log("val", loss=val_loss, error=self.validation_error())

    def evaluate(self):
        assert len(self.testset_embeddings) == 0, "Test embeddings are not empty"

        test_loss = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                data, labels = batch

                print(data)
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                self.testset_embeddings.append(outputs.cpu())
                loss = vertex_reconstruction_loss(
                    outputs, self.vertices_embeddings_current_epoch, labels
                )
                test_loss += loss.item()

        test_loss /= len(self.test_loader)
        self.metrics_tracker.log("test", loss=test_loss, error=self.test_error())

    def visualize(
        self, metrics=None, show=False, save_path=None, title="Training Metrics"
    ):

        return self.metrics_visualizer(
            self.stats,
            metrics=metrics,
            show=show,
            save_path=save_path,
            title=title,
        )

    def training_error(self):
        self.trainset_embeddings_current_epoch = torch.cat(self.trainset_embeddings_current_epoch, dim=0)
        in_hull_check = check_in_hull(self.trainset_embeddings_current_epoch,
                                            self.vertices_embeddings_current_epoch)
        
        return torch.sum(in_hull_check == self.train_loader.dataset.dataset.labels[self.train_loader.dataset.indices]) \
            / len(self.train_loader.dataset)

    def validation_error(self):
        self.valset_embeddings_current_epoch = torch.cat(self.valset_embeddings_current_epoch, dim=0)
        in_hull_check = check_in_hull(self.valset_embeddings_current_epoch, self.vertices_embeddings_current_epoch)
        
        return torch.sum(in_hull_check == self.val_loader.dataset.dataset.labels[self.val_loader.dataset.indices]) \
            / len(self.val_loader.dataset)

    def test_error(self):
        self.testset_embeddings = torch.cat(self.testset_embeddings, dim=0)
        in_hull_check = check_in_hull(self.testset_embeddings, self.vertices_embeddings_current_epoch)

        print("Test set in-hull check:\n", in_hull_check)

        return torch.sum(in_hull_check == self.test_loader.dataset.dataset.labels[self.test_loader.dataset.indices]) \
            / len(self.test_loader.dataset)