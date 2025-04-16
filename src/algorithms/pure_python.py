import numpy as np
from typing import List
from .base import BaseCalculator

class PurePythonCalculator(BaseCalculator):
    """Pure Python implementation using loops"""
    
    def calculate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = self.validate_input(a, b)
        size = len(a)
        result = [0] * size
        
        for i in range(size):
            result[i] = a[i] + b[i] * 2 - a[i] % 3
            
        return np.array(result)

    def warmup(self, size: int = 1000) -> None:
        """No warmup needed for pure Python"""
        pass 