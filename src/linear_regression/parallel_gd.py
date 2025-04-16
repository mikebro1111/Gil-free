import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Tuple, List
from .base import BaseRegression

def _compute_gradients(args: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Compute gradients for a chunk of data"""
    X_chunk, y_chunk, theta = args
    m = len(y_chunk)
    prediction = X_chunk.dot(theta)
    error = prediction - y_chunk
    return 2/m * X_chunk.T.dot(error)

class ParallelGradientDescentRegression(BaseRegression):
    """Linear regression using parallel gradient descent"""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 tol: float = 1e-7, n_processes: int = None):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.n_processes = n_processes or cpu_count()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ParallelGradientDescentRegression':
        """Fit using parallel gradient descent"""
        X, y = self._validate_input(X, y)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.zeros(X_b.shape[1])
        
        # Split data into chunks
        chunk_size = len(y) // self.n_processes
        X_chunks = np.array_split(X_b, self.n_processes)
        y_chunks = np.array_split(y, self.n_processes)
        
        with Pool(processes=self.n_processes) as pool:
            for _ in range(self.n_iterations):
                # Compute gradients in parallel
                chunk_args = [(X_chunk, y_chunk, theta) 
                            for X_chunk, y_chunk in zip(X_chunks, y_chunks)]
                gradients_list = pool.map(_compute_gradients, chunk_args)
                
                # Average gradients
                gradients = np.mean(gradients_list, axis=0)
                
                # Early stopping
                if np.all(np.abs(gradients) < self.tol):
                    break
                    
                # Update parameters
                theta = theta - self.learning_rate * gradients
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        self.is_fitted = True
        
        return self

    def warmup(self, n_samples: int = 1000, n_features: int = 10) -> None:
        """No warmup needed for parallel gradient descent"""
        pass 