import numpy as np

class LinearExtrapolator:

    def __init__(self, num_points):

        self.num_points = num_points
        self.init_values = np.zeros(num_points)
        self.final_values = np.zeros(num_points)

    def fit(self, X, S):
        
        assert X.shape[1] == self.num_points, "Input data must have the same number of points as specified."
        assert S.shape[0] == X.shape[0], "The number of scalar values must match the number of data points."
        
        min_index = np.argmin(S)
        self.init_values = X[min_index]
        max_index = np.argmax(S)
        self.final_values = X[max_index]
    
    def extrapolate(self, s):
        # s: (m,) normalized scalar
        return self.init_values + s[:, np.newaxis] * (self.final_values - self.init_values)
