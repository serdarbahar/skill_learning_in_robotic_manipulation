import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

class DummyVisualizer:
    def __init__(self):

        load_dotenv()
        self.results_dir = os.getenv("RESULTS_DIR")

    def save_external_plot(self, fig: plt.Figure, filename: str):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        fig.savefig(os.path.join(self.results_dir, filename))
