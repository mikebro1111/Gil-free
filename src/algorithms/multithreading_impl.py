import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Tuple
from .base import BaseCalculator
from multiprocessing import cpu_count

class ThreadSafeArray:
    def __init__(self, size: int):
        self.data = np.zeros(size)
        self.lock = Lock()

    def update_slice(self, start: int, values: np.ndarray) -> None:
        with self.lock:
            self.data[start:start + len(values)] = values

def _process_chunk(args: Tuple[np.ndarray, np.ndarray, int]) -> Tuple[np.ndarray, int]:
    """Worker function for threading"""
    a_chunk, b_chunk, start_idx = args
    result = a_chunk + b_chunk * 2 - a_chunk % 3
    return result, start_idx

class MultithreadingCalculator(BaseCalculator):
    """Multithreading implementation with proper synchronization"""
    
    def __init__(self, n_threads: int = None):
        self.n_threads = n_threads or cpu_count()

    def calculate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = self.validate_input(a, b)
        size = len(a)
        result = ThreadSafeArray(size)
        futures = []

        chunk_size = size // self.n_threads
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # Submit tasks
            for i in range(self.n_threads):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < self.n_threads - 1 else size
                
                future = executor.submit(
                    _process_chunk,
                    (a[start_idx:end_idx], b[start_idx:end_idx], start_idx)
                )
                futures.append(future)

            # Wait for all results and update the array
            for future in as_completed(futures):
                chunk_result, start_idx = future.result()
                result.update_slice(start_idx, chunk_result)

        return result.data

    def warmup(self, size: int = 1000) -> None:
        """No warmup needed for multithreading"""
        pass