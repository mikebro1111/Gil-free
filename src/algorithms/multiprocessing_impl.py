import numpy as np
from multiprocessing import Pool, cpu_count
from typing import Tuple, List
from .base import BaseCalculator

def _process_chunk(args: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Worker function for multiprocessing"""
    a_chunk, b_chunk = args
    return a_chunk + b_chunk * 2 - a_chunk % 3

class MultiprocessingCalculator(BaseCalculator):
    """Multiprocessing implementation"""
    
    def __init__(self, n_processes: int = None):
        self.n_processes = n_processes or cpu_count()

    def calculate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = self.validate_input(a, b)
        
        # Split arrays into chunks
        chunks = list(zip(
            np.array_split(a, self.n_processes),
            np.array_split(b, self.n_processes)
        ))
        
        # Process chunks in parallel
        with Pool(processes=self.n_processes) as pool:
            results = pool.map(_process_chunk, chunks)
            
        # Combine results
        return np.concatenate(results)

    def warmup(self, size: int = 1000) -> None:
        """No warmup needed for multiprocessing"""
        pass 