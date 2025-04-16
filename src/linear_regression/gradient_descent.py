import numpy as np
from numba import jit
from .base import BaseRegression

class GradientDescentRegression(BaseRegression):
    """Linear regression using gradient descent with Numba acceleration"""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 tol: float = 1e-7):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self._compiled = False

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray,
                         learning_rate: float, n_iterations: int, tol: float) -> np.ndarray:
        m = len(y)
        
        for _ in range(n_iterations):
            prediction = X.dot(theta)
            error = prediction - y
            gradients = 2/m * X.T.dot(error)
            
            # Early stopping if gradients are small
            if np.all(np.abs(gradients) < tol):
                break
                
            theta = theta - learning_rate * gradients
            
        return theta

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientDescentRegression':
        """Fit using gradient descent"""
        X, y = self._validate_input(X, y)
        
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Initialize parameters
        theta = np.zeros(X_b.shape[1])
        
        # Run gradient descent
        if not self._compiled:
            self.warmup()
            
        theta = self._gradient_descent(X_b, y, theta, self.learning_rate,
                                     self.n_iterations, self.tol)
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        self.is_fitted = True
        
        return self

    def warmup(self, n_samples: int = 1000, n_features: int = 10) -> None:
        """Compile Numba function with small arrays"""
        if not self._compiled:
            X = np.random.randn(n_samples, n_features)
            y = np.random.randn(n_samples)
            X_b = np.c_[np.ones((n_samples, 1)), X]
            theta = np.zeros(n_features + 1)
            
            _ = self._gradient_descent(X_b, y, theta, self.learning_rate,
                                     10, self.tol)
            self._compiled = True 