import argparse

import matplotlib.pyplot as plt
import torch

from transform_learning.losses.vertex_reconstruction import vertex_reconstruction_loss


class VertexLossVisualizer:
    """Visualize the vertex reconstruction loss over a 2D grid."""

    def __init__(self,):

        self.scale = 1.0

        self.vertex_points = (torch.tensor([
                    
                    [-1.0 * self.scale, -1.0 * self.scale],
                    [-1.0 * self.scale, 1.0 * self.scale],
                    [1.0 * self.scale, -1.0 * self.scale],
                    [1.0 * self.scale, 1.0 * self.scale],

                ], dtype=torch.float32,))
        
        self.temperature = 1.0
        self.outside_margin = 0.1

    def _compute_loss_on_grid(self, label: int, x_min: float, x_max: float, y_min: float, y_max: float, resolution: int,
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x_vals = torch.linspace(x_min, x_max, resolution)
        y_vals = torch.linspace(y_min, y_max, resolution)
        xx, yy = torch.meshgrid(x_vals, y_vals, indexing="xy")

        points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        losses = []
        with torch.no_grad():
            for point in points:
                point_loss = vertex_reconstruction_loss(
                    outputs=point.unsqueeze(0),
                    vertices_embeddings=self.vertex_points,
                    labels=torch.tensor([label], dtype=torch.float32),
                    temperature=self.temperature,
                    outside_margin=self.outside_margin,
                )
                losses.append(point_loss.item())

        loss_grid = torch.tensor(losses, dtype=torch.float32).reshape(resolution, resolution)
        return xx, yy, loss_grid

    def plot(self, x_min: float = -5.0, 
                    x_max: float = 5.0, 
                    y_min: float = -5.0, 
                    y_max: float = 5.0, resolution: int = 150, point_size: int = 12,) -> None:
        
        x_min *= self.scale
        x_max *= self.scale
        y_min *= self.scale
        y_max *= self.scale
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        labels_and_titles = [
            (1, "Loss Landscape for label=1 (inside)"),
            (0, "Loss Landscape for label=0 (outside)"),
        ]

        for ax, (label, title) in zip(axes, labels_and_titles):
            xx, yy, loss_grid = self._compute_loss_on_grid(
                label=label, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, resolution=resolution)

            scatter = ax.scatter(xx.numpy().reshape(-1), yy.numpy().reshape(-1), c=loss_grid.numpy().reshape(-1), cmap="viridis", s=point_size, alpha=0.9)

            vertices = self.vertex_points
            ax.scatter(vertices[:, 0], vertices[:, 1], c="red", s=45, label="Vertices", zorder=3)

            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="upper right")
            fig.colorbar(scatter, ax=ax, label="Loss")

        fig.suptitle(f"Vertex Reconstruction Loss Visualization, temperature={self.temperature}, outside_margin={self.outside_margin}")
        plt.savefig("results/vertex_loss_visualization.png")
        plt.close()

def main() -> None:
    visualizer = VertexLossVisualizer()
    visualizer.plot()

if __name__ == "__main__":
    main()
