import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple
from .base import BaseRegression

class ParallelGradientDescentRegression(BaseRegression):
    def __init__(self, learning_rate: float = 0.01,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 n_jobs: Optional[int] = None):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_jobs = n_jobs if n_jobs is not None else cpu_count()
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        
    def _compute_gradient_chunk(self, args: Tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> Tuple[np.ndarray, float]:
        X_chunk, y_chunk, coefficients, intercept = args
        n_samples = len(X_chunk)
        
        # Compute predictions for this chunk
        y_pred = np.dot(X_chunk, coefficients) + intercept
        
        # Compute gradients
        gradients = 2/n_samples * np.dot(X_chunk.T, (y_pred - y_chunk))
        intercept_gradient = 2/n_samples * np.sum(y_pred - y_chunk)
        
        return gradients, intercept_gradient
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ParallelGradientDescentRegression':
        X, y = self._validate_input(X, y)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        # Split data into chunks for parallel processing
        chunk_size = n_samples // self.n_jobs
        X_chunks = [X[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]
        y_chunks = [y[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]
        
        with Pool(self.n_jobs) as pool:
            for _ in range(self.max_iterations):
                # Prepare arguments for parallel processing
                args = [(X_chunk, y_chunk, self.coefficients, self.intercept)
                       for X_chunk, y_chunk in zip(X_chunks, y_chunks)]
                
                # Compute gradients in parallel
                results = pool.map(self._compute_gradient_chunk, args)
                
                # Aggregate gradients
                total_gradients = np.zeros_like(self.coefficients)
                total_intercept_gradient = 0
                
                for gradients, intercept_gradient in results:
                    total_gradients += gradients
                    total_intercept_gradient += intercept_gradient
                
                # Average gradients
                total_gradients /= len(X_chunks)
                total_intercept_gradient /= len(X_chunks)
                
                # Update parameters
                prev_coefficients = self.coefficients.copy()
                self.coefficients -= self.learning_rate * total_gradients
                self.intercept -= self.learning_rate * total_intercept_gradient
                
                # Check convergence
                if np.all(np.abs(self.coefficients - prev_coefficients) < self.tolerance):
                    break
        
        self.is_fitted = True
        return self

    def warmup(self, n_samples: int = 1000, n_features: int = 10) -> None:
        """No warmup needed for parallel gradient descent"""
        pass 