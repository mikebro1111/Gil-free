import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class DataGenerator:
    """Class for generating synthetic datasets"""
    
    @staticmethod
    def generate_arrays(size: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random arrays for basic operations"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.random(size), np.random.random(size)

    @staticmethod
    def generate_regression_data(
        n_samples: int,
        n_features: int,
        noise: float = 0.1,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic regression dataset
        
        Parameters:
        -----------
        n_samples : int
            Number of samples
        n_features : int
            Number of features
        noise : float
            Noise level
        test_size : float
            Proportion of test set
        random_state : int, optional
            Random seed
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : np.ndarray
            Training and test sets
        """
        # Generate regression dataset
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
        
        # Split into train and test sets
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