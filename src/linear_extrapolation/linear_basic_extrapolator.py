import numpy as np

class LinearBasicExtrapolator:

    def __init__(self, num_points):

        self.num_points = num_points
        self.min_values = np.zeros(num_points)
        self.max_values = np.zeros(num_points)

    def fit(self, X):
        # X: (n, num_points)
        self.min_values = np.min(X, axis=0)
        self.max_values = np.max(X, axis=0)
    
    def extrapolate(self, s):
        # s: (m,) normalized scalar
        return self.min_values + s[:, np.newaxis] * (self.max_values - self.min_values)
    
if __name__ == "__main__":
    # Example usage
    X = np.array([[1, 2], [3, 4], [5, 6]])
    extrapolator = LinearBasicExtrapolator(num_points=2)
    extrapolator.fit(X)
    
    s = np.array([-0.5, 0.5, 1.5])  # Normalized scalar values
    extrapolated_points = extrapolator.extrapolate(s)
    print(extrapolated_points)