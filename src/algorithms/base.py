from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class BaseCalculator(ABC):
    """Base interface for all calculation implementations"""
    
    @abstractmethod
    def calculate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Perform calculation on input arrays"""
        pass

    @abstractmethod
    def warmup(self, size: int = 1000) -> None:
        """Warm up the implementation (if needed)"""
        pass

    def validate_input(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input arrays"""
        if a.shape != b.shape:
            raise ValueError(f"Arrays must have same shape. Got {a.shape} and {b.shape}")
        return a, b 