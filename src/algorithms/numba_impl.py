import numpy as np
from numba import jit
from .base import BaseCalculator

class NumbaCalculator(BaseCalculator):
    """Numba JIT-compiled implementation"""
    
    def __init__(self):
        self._compiled = False
        
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_calc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        result = np.zeros_like(a)
        for i in range(len(a)):
            result[i] = a[i] + b[i] * 2 - a[i] % 3
        return result

    def calculate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = self.validate_input(a, b)
        if not self._compiled:
            self.warmup()
        return self._numba_calc(a, b)

    def warmup(self, size: int = 1000) -> None:
        """Compile the Numba function with small arrays"""
        if not self._compiled:
            a = np.random.random(size)
            b = np.random.random(size)
            _ = self._numba_calc(a, b)
            self._compiled = True 