from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import r2_score

class BaseRegression(ABC):
    """Base class for all linear regression implementations"""
    
    def __init__(self):
        self.coefficients: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseRegression':
        """Fit the model to the data"""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept, self.coefficients])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score"""
        return r2_score(y, self.predict(X))

    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input data"""
        if len(X.shape) != 2:
            raise ValueError("X must be 2-dimensional")
        if len(y.shape) != 1:
            raise ValueError("y must be 1-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. Got {X.shape[0]} and {y.shape[0]}")
        return X, y

    @abstractmethod
    def warmup(self, n_samples: int = 1000, n_features: int = 10) -> None:
        """Warm up the implementation (if needed)"""
        pass 