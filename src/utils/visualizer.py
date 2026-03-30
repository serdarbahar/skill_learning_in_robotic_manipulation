import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv

class Visualizer:
    def __init__(self, x, y):
        # x: (n, num_points) (1 -> num feat)
        # y: (n, num_points)

        # type check
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be numpy arrays.")
        if len(x.shape) != 2 or len(y.shape) != 2 or x.shape[1:] != y.shape[1:]:
            raise ValueError("x and y must be 2D arrays with the same last dimension.")

        self.train = x
        self.pred = y

        load_dotenv()
        self.results_dir = os.getenv("RESULTS_DIR")

    def plot(self):
        
        plt.figure(figsize=(10, 6))

        print(self.train.shape)
        T = np.linspace(0, 1, self.train.shape[1])

        for i in range(self.train.shape[0]):
            label_txt = "Train Traj" if i == 0 else None  # Only label the first one for legend
            plt.plot(T, self.train[i, :], label=label_txt, c = 'black', alpha = 0.9)
        for i in range(self.pred.shape[0]):
            plt.plot(T, self.pred[i, :], label=f'Predicted Traj {i+1}', c = 'blue', alpha = 0.5)

        plt.title('Linear Extrapolation Visualization')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, "extrapolation_plot.png"))
        plt.close()

    def save_external_plot(self, fig: plt.Figure, filename: str):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        fig.savefig(os.path.join(self.results_dir, filename))
