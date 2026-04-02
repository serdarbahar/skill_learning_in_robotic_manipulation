from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt


class MetricsVisualizer:
    """Callable visualization utility for trainer metrics.

    Customize output by selecting metric groups, split order, colors, and save path.
    """

    def __init__(
        self,
        default_metrics: Optional[Sequence[str]] = None,
        split_order: Optional[Sequence[str]] = None,
        colors: Optional[Dict[str, str]] = None,
        figsize=(12, 4),
    ):

        self.default_metrics = tuple(default_metrics or ("loss", "error"))
        self.split_order = tuple(split_order or ("train", "val", "test"))
        self.colors = {
            "train": "#1f77b4",
            "val": "#ff7f0e",
            "test": "#2ca02c",
        }
        if colors:
            self.colors.update(colors)
        self.figsize = figsize

    def __call__(
        self,
        stats: Dict[str, list],
        metrics: Optional[Iterable[str]] = None,
        show: bool = False,
        save_path: Optional[str] = None,
        title: str = "Training Metrics",
    ):
        

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
                    linewidth=2,
                )
                plotted_any_series = True

            axis.set_title(metric_name.capitalize())
            axis.set_xlabel("Step")
            axis.set_ylabel(metric_name.capitalize())
            axis.grid(alpha=0.3)
            if plotted_any_series:
                axis.legend()

        fig.suptitle(title)
        fig.tight_layout()

        if save_path:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
