import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from .base import BaseRegression

class SklearnWrapper(BaseRegression):
    """Wrapper for scikit-learn's LinearRegression"""
    
    def __init__(self):
        super().__init__()
        self.model = SklearnLinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SklearnWrapper':
        """Fit using scikit-learn's implementation"""
        X, y = self._validate_input(X, y)
        
        self.model.fit(X, y)
        self.intercept = self.model.intercept_
        self.coefficients = self.model.coef_
        self.is_fitted = True
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using scikit-learn's implementation"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def warmup(self, n_samples: int = 1000, n_features: int = 10) -> None:
        """No warmup needed for scikit-learn"""
        pass 