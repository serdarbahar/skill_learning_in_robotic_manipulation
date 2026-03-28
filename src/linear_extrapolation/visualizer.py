import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv

class Visualizer:
    def __init__(self, x, y):
        # x: (n, num_points, 1) (1 -> num feat)
        # y: (n, num_points, 1)

        # type check
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("x and y must be numpy arrays.")
        if len(x.shape) != 3 or len(y.shape) != 3 or x.shape[1:] != y.shape[1:]:
            raise ValueError("x and y must be 3D arrays with the same two last dimensions.")

        self.train = x
        self.pred = y

        load_dotenv()
        self.results_dir = os.getenv("RESULTS_DIR")

    def plot(self):
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train[:,:,0], label='Train Data', marker='o')
        plt.plot(self.pred[:,:,0], label='Extrapolated Data', marker='x')
        plt.title('Linear Extrapolation Visualization')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid()
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(self.results_dir, "extrapolation_plot.png"))
