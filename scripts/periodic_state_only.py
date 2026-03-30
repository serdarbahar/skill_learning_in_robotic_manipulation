import numpy as np
from periodic_warping.periodic_emb_state_model import PeriodicEmbStateModel
import matplotlib.pyplot as plt
from src.utils.dummy_visualizer import DummyVisualizer

def run_experiment():

    viz = DummyVisualizer()
 
    # --- Setup ---
    N = 100  # phase steps
    s = np.linspace(0, 1, N)
 
    def trajectory(s, c):
        return c * np.sin(np.pi * s)
 
    # Training conditioning values
    c_train_vals = np.array([-1.0,
                            1.0,],          
                    dtype=np.float32)[:, None]  # (M, 1) for broadcasting
    M = len(c_train_vals)
 
    # Generate training trajectories: (M, N)
    q_train = np.array([trajectory(s, c) for c in c_train_vals], dtype=np.float32)[:, :, None]  # Add feature dim -> (M, N, 1)
 
    # Query conditioning values (includes extrapolation)
    c_query_vals = np.array([
        0.0,
        0.5,
        1.0,
        2.0,
    ], dtype=np.float32).reshape(-1, 1)  # (K, 1) for broadcasting
    K = len(c_query_vals)
 
    # Ground truth
    q_true = np.array([trajectory(s, c) for c in c_query_vals], dtype=np.float32)

    model_pw = PeriodicEmbStateModel(
        period_c = np.array([2 * 2], dtype=np.float32),  
        length_scale = 1.5,
    )
    model_pw.fit(c_train_vals, q_train, s)

    q_pred_pw = model_pw.predict(c_query_vals)

    # --- Plotting ---
    fig, axes = plt.subplots(1, K, figsize=(6.4 * K, 8), squeeze=False)
 
    col_title = "Periodic Warping (Toroidal)"
    for ax in axes[0, :]:
        ax.set_title(col_title, fontsize=12, fontweight="bold")
 
    for i in range(K):
        #c_q = c_query_vals[i]
 
        # Periodic Warping
        ax1 = axes[0, i]
        for j in range(M):
            ax1.plot(s, q_train[j], "k--", alpha=0.3)
        ax1.plot(s, q_true[i], "b-", linewidth=1.5, alpha=0.4, label="true")
        ax1.plot(s, q_pred_pw[i, :, 0], "g-", linewidth=2, label="predicted")
        ax1.set_ylim(-2.0, 2.0)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
 
    for ax in axes[-1]:
        ax.set_xlabel("Phase s")
 
    plt.tight_layout()
    #plt.show()
    viz.save_external_plot(fig, "periodic_state_only_results.png")
    plt.close()

if __name__ == "__main__":
    run_experiment()