import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-2, 2, 100)

def func(x: np.float32) -> np.ndarray:
    values = np.array((2, 1), dtype=np.float32)
    values[0] = x * np.exp(-(x-1)**2)
    values[1] = np.exp(-x**2)
    return values

scatters = [-1.0, -0.5, 0.0, 0.5, 1.0]
if __name__ == "__main__":
    Y = np.array([func(x) for x in X])

    plt.plot(Y[:, 0], Y[:, 1])

    for scatter in scatters:
        scatter_point = func(np.float32(scatter))
        plt.scatter(scatter_point[0], scatter_point[1], color="red", s=100, label=f"scatter at x={scatter:.1f}")

    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.grid()
    plt.axis("equal")
    plt.show()

