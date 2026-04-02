from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
import matplotlib.pyplot as plt
from transform_learning.data import CustomPointDataset

class MetricsVisualizer:
    """Callable visualization utility for trainer metrics.
    """

    def __init__(
        self,
        default_metrics: Optional[Sequence[str]] = None,
        split_order: Optional[Sequence[str]] = None,
        colors: Optional[Dict[str, str]] = None,
        figsize=(12, 4),
    ):

        self.default_metrics = tuple(default_metrics or ("loss", "success"))
        self.split_order = tuple(split_order or ("train", "val"))
        self.colors = {
            "train": "#1f77b4",
            "val": "#ff7f0e"
        }
        if colors:
            self.colors.update(colors)
        self.figsize = figsize

    def __call__(self, stats: Dict[str, list], metrics: Optional[Iterable[str]] = None, save_dir: Optional[str] = None, title: str = "Training Metrics"):
        
        selected_metrics = tuple(metrics or self.default_metrics)
        if not selected_metrics:
            raise ValueError("At least one metric must be provided.")

        fig, axes = plt.subplots(1, len(selected_metrics), figsize=self.figsize)
        if len(selected_metrics) == 1:
            axes = [axes]

        for axis, metric_name in zip(axes, selected_metrics):
            plotted_any_series = False
            for split in self.split_order:
                key = f"{split}_{metric_name}"
                values = stats.get(key, [])
                if not values:
                    continue

                axis.plot(
                    range(1, len(values) + 1),
                    values,
                    label=split,
                    color=self.colors.get(split),
                    linewidth=1,
                )
                plotted_any_series = True

            axis.set_title(metric_name.capitalize())
            axis.set_xlabel("Step")
            axis.set_ylabel(metric_name.capitalize())
            axis.grid(alpha=0.3)
            if plotted_any_series:
                axis.legend()

        # write the test error and test success as text annotations if available
        if "test_loss" in stats and stats["test_loss"]:
            test_loss = stats["test_loss"][-1]
            axes[0].annotate(f"Test Loss: {test_loss:.6f}", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=10, color="black")
        if "test_success" in stats and stats["test_success"]:
            test_success = stats["test_success"][-1]
            axes[1].annotate(f"Test Success: {test_success:.3f}", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=10, color="black")

        fig.suptitle(title)
        fig.tight_layout()

        if save_dir:
            output_path = Path(save_dir) / "training_metrics.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)
    
        plt.close(fig)

        return fig


class EmbeddingsVisualizer:
    """Callable visualization utility for trainer embeddings.
    """

    def __init__(self, figsize=(6, 6)):
        self.figsize = figsize

    def __call__(self, 
                model: torch.nn.Module,
                dataset: CustomPointDataset,
                vertices: torch.Tensor,
                save_dir: Optional[str] = None,
                num_points: int = 200,
                title: Optional[str] = "Embedding Visualization"):
        
        # assume vertices are 1D and normalized to [-1, 1]
        min_point, max_point = -1.0, 1.0

        points = torch.linspace(min_point - dataset.eps * dataset.n, max_point + dataset.eps * dataset.n, num_points).unsqueeze(1)
        with torch.no_grad():
            embeddings = model(points.to(next(model.parameters()).device)).cpu()
        labels = torch.tensor([dataset.get_label(p.item()) for p in points.squeeze()])
        for embedding, label in zip(embeddings, labels):
            plt.scatter(embedding[0].item(), embedding[1].item(), color="red" if label == 0 else "blue", alpha=0.2)

        # Plot vertices
        with torch.no_grad():
            vertices_embeddings = model(vertices.to(next(model.parameters()).device)).cpu()
        for i in range(vertices.shape[0]):
            plt.scatter(vertices_embeddings[i, 0].item(), vertices_embeddings[i, 1].item(), color="black", marker="X", s=100, label="Vertex" if i == 0 else None)
            plt.text(vertices_embeddings[i, 0].item(), vertices_embeddings[i, 1].item(), f"V{i}", fontsize=9, ha="right", va="bottom")

        plt.title(title)
        plt.xlabel("Embedding Dim 1")
        plt.ylabel("Embedding Dim 2")
        plt.grid(alpha=0.3)
        plt.legend()

        if save_dir:
            output_path = Path(save_dir) / "embedding_visualization.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
    

        


        
