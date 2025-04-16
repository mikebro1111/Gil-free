import numpy as np
from .base import BaseCalculator

class NumpyCalculator(BaseCalculator):
    """NumPy vectorized implementation"""
    
    def calculate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = self.validate_input(a, b)
        return a + b * 2 - a % 3

    def warmup(self, size: int = 1000) -> None:
        """No warmup needed for NumPy"""
        pass 