import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class DataGenerator:
    """Class for generating test data"""
    
    @staticmethod
    def generate_arrays(size: int, random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two random arrays for testing"""
        if random_seed is not None:
            np.random.seed(random_seed)
        return np.random.rand(size), np.random.rand(size)
    
    @staticmethod
    def generate_regression_data(n_samples: int, 
                               n_features: int,
                               test_size: float = 0.2,
                               random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate regression dataset"""
        if random_state is not None:
            np.random.seed(random_state)
            
        X = np.random.randn(n_samples, n_features)
        true_coefficients = np.random.randn(n_features)
        y = np.dot(X, true_coefficients) + np.random.randn(n_samples) * 0.1
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @staticmethod
    def generate_regression_with_correlation(
        n_samples: int,
        n_features: int,
        correlation: float = 0.5,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate dataset with specified feature correlation"""
        if random_state is not None:
            np.random.seed(random_state)
            
        # Generate correlated features
        cov_matrix = np.eye(n_features) * (1 - correlation) + correlation
        X = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=cov_matrix,
            size=n_samples
        )
        
        # Generate target
        true_coefficients = np.random.randn(n_features)
        y = X.dot(true_coefficients) + np.random.randn(n_samples) * 0.1
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state) 