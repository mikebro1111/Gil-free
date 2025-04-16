import numpy as np
from .base import BaseRegression

class NormalEquationRegression(BaseRegression):
    """Linear regression using normal equation method"""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NormalEquationRegression':
        """Fit using normal equation: θ = (X^T X)^(-1) X^T y"""
        X, y = self._validate_input(X, y)
        
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate parameters: θ = (X^T X)^(-1) X^T y
        try:
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for numerical stability
            theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        self.is_fitted = True
        
        return self

    def warmup(self, n_samples: int = 1000, n_features: int = 10) -> None:
        """No warmup needed for normal equation method"""
        pass 