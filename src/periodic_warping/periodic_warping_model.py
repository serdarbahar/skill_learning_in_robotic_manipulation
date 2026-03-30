import numpy as np
from typing import Tuple

 
class PeriodicWarpingModel:
    
    def __init__(
        self,
        period_c: float = 2 * np.pi,
        period_q: float = 2 * np.pi,
        length_scale: float = 1.0,
    ):
        
        self.period_c = np.array(period_c, dtype=np.float32)
        self.period_q = np.array(period_q, dtype=np.float32)
        self.length_scale = length_scale
 
        self._c_train_embedded = None  # (M, 2)
        self._q_train_embedded = None  # (M, N, 2)
        self._s_train = None           # (N,)
        self._num_traj = None
    
    def _embed_to_circle(x: np.ndarray, period: np.ndarray) -> np.ndarray:

        assert x.ndim == 2, "Input x must be 2D (M, d)"
        assert x.shape[1] == len(period), "Input dimension must match period dimension"
 
        theta = 2 * np.pi * x / period  # (*, d)
        cos_vals = np.cos(theta)
        sin_vals = np.sin(theta)
        z = np.stack([cos_vals, sin_vals], axis=-1).reshape(*x.shape[:-1], -1) # Stack and reshape: (*, d, 2) -> (*, 2d)
        return z

    def _unembed_from_circle(z: np.ndarray, period: np.ndarray) -> np.ndarray:

        d = len(period)
        # Reshape to (*, d, 2)
        z_pairs = z.reshape(*z.shape[:-1], d, 2)
        cos_vals = z_pairs[..., 0]
        sin_vals = z_pairs[..., 1]
 
        # atan2 recovers the angle, result in [-pi, pi]
        theta = np.arctan2(sin_vals, cos_vals)
 
        x = theta * period / (2 * np.pi)
        return x

    def _rbf_kernel(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray: # radial basis function kernel for embedded inputs

        # Squared Euclidean distance in embedded space
        diff = z1[:, None, :] - z2[None, :, :]  # (M, K, D)
        sq_dist = np.sum(diff ** 2, axis=-1)      # (M, K)
        return np.exp(-sq_dist / (2 * self.length_scale ** 2))

    def fit(self, c_train: np.ndarray, q_train: np.ndarray, s: np.ndarray):

        assert len(c_train) == len(q_train) == len(s), "Training data length mismatch"
        assert c_train.dtype == np.float32 and q_train.dtype == np.float32, "Training data must be float32"
        assert c_train.ndim == 2 and q_train.ndim == 3, "c_train must be (M, d_c) and q_train must be (M, N, d_q)"
 
        M, N, d_q = q_train.shape
 
        # Embed conditioning variables: (M, d_c) -> (M, 2*d_c)
        self._c_train_embedded = self._embed_to_circle(c_train, self.period_c)
 
        # Embed output trajectories: (M, N, d_q) -> (M, N, 2*d_q)
        self._q_train_embedded = np.zeros((M, N, 2 * d_q))
        for j in range(N):
            self._q_train_embedded[:, j, :] = self._embed_to_circle(
                q_train[:, j, :], self.period_q
            )
 
        self._s_train = s.copy()
        self._num_traj = N
 
    def predict(self, c_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        assert c_query.dtype == np.float32, "Query data must be float32"
        assert c_query.ndim == 2, "Query data must be 2D (K, d_c)"
        assert c_query.shape[1] == len(self.period_c), "Query dimension must match period_c dimension"
 
        K = c_query.shape[0]
        N = self._num_traj
        d_q = len(self.period_q)
 
        c_query_emb = self._embed_to_circle(c_query, self.period_c) # Embed query conditioning: (K, d_c) -> (K, 2*d_c)
 
        weights = self._rbf_kernel(c_query_emb, self._c_train_embedded) # Kernel weights: (K, M)

        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12) # Normalize weights
 
        q_pred = np.zeros((K, N, d_q))
        confidence = np.zeros((K, N, d_q))
 
        for j in range(N):
            
            # Weighted average in embedded output space: (K, 2*d_q)
            q_emb_j = self._q_train_embedded[:, j, :]  # (M, 2*d_q)
            q_avg = weights @ q_emb_j                    # (K, 2*d_q)
 
            # Compute confidence as norm of each (cos, sin) pair before normalization
            q_pairs = q_avg.reshape(K, d_q, 2)           # (K, d_q, 2)
            norms = np.linalg.norm(q_pairs, axis=-1)      # (K, d_q)
            confidence[:, j, :] = norms
 
            # Unembed (atan2 handles the normalization implicitly)
            q_pred[:, j, :] = self._unembed_from_circle(q_avg, self.period_q)
 
        return q_pred, confidence


        
